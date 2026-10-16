"""End-to-end orchestration of one claim.

Read this file top to bottom to see the workflow. Each step is tagged:
  [DET]  deterministic code
  [LLM]  model judgment, validated before use
  [HITL] a human decides or signs off
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Callable

from . import ai_steps, rules
from .deployment import RolloutConfig, ai_is_visible, effective_route
from .errors import AIStepError, CircuitBreaker, SystemOfRecordError, with_retries
from .integrations import ClaimsDBAdapter, ERPAdapter, ReviewQueueAdapter, TMSAdapter, decision_to_json
from .llm import LLMClient
from .models import Decision, Route
from .router import route_claim

log = logging.getLogger("meridian.pipeline")

LETTER_TEMPLATES = {
    # Legal wording is fixed and reviewed by counsel. The model only writes `{personal}`.
    Route.AUTO_APPROVE: "Dear {name},\n\n{personal}\n\nWe have approved your claim on PRO {pro} for ${amount}. "
    "Payment will be issued within 10 business days.\n\nMeridian Freight Claims",
    Route.AUTO_DENY_DRAFT: "Dear {name},\n\n{personal}\n\nAfter review, we are unable to pay your claim on PRO {pro} "
    "for the following reason: {reason}. You may request reconsideration within 60 days.\n\nMeridian Freight Claims",
}

# Claimant-facing wording for each hard rule; internal rule names never reach a letter.
DENIAL_REASONS = {
    "filing_deadline": "the claim was received after the 9-month filing period ({detail})",
    "commodity_covered": "{detail} are excluded from carrier liability under our tariff",
}


@dataclass
class Pipeline:
    llm: LLMClient
    tms: TMSAdapter
    claims_db: ClaimsDBAdapter
    review_queue: ReviewQueueAdapter
    erp: ERPAdapter
    config: RolloutConfig = field(default_factory=RolloutConfig.from_env)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    retry_sleep: Callable[[float], None] = time.sleep  # injectable so tests don't wait

    def _ai(self, fn):
        """Every model call goes through the circuit breaker and retry policy."""
        return self.breaker.call(lambda: with_retries(fn, sleep=self.retry_sleep))

    def _manual(self, claim_id: str, reason: str, **partial) -> Decision:
        """The fallback for every failure: the claim goes to a person, as it did before AI."""
        decision = Decision(claim_id, Route.ADJUSTER_REVIEW, Decimal("0.00"), [], [reason], **partial)
        return self._commit(decision)

    def process(self, claim_id: str, raw_text: str, received_on: date) -> Decision:
        # 1 [LLM] Intake: email/form text -> structured claim with evidence quotes.
        try:
            extracted = self._ai(lambda: ai_steps.extract_claim(self.llm, raw_text))
        except AIStepError as exc:
            log.warning("%s: intake failed (%s)", claim_id, exc)
            return self._manual(claim_id, f"AI intake failed: {exc}")

        # 2 [DET] PRO number is the join key to everything else; no PRO, no automation.
        if not extracted.pro_number:
            return self._manual(claim_id, "no PRO number found in claim", extracted=extracted)

        # 3 [DET] Shipment lookup in the TMS (system of record).
        try:
            shipment = self.tms.get_shipment(extracted.pro_number)
        except SystemOfRecordError as exc:
            log.error("%s: %s", claim_id, exc)
            return self._manual(claim_id, f"TMS error: {exc}", extracted=extracted)
        if shipment is None:
            return self._manual(claim_id, f"PRO {extracted.pro_number} not found in TMS", extracted=extracted)

        # 4 [LLM] Damage classification and narrative consistency flags.
        try:
            assessment = self._ai(lambda: ai_steps.classify_damage(self.llm, extracted, shipment))
        except AIStepError as exc:
            return self._manual(claim_id, f"AI classification failed: {exc}", extracted=extracted)

        # 5 [DET] Policy rules, fraud screens, and settlement arithmetic.
        history = self.claims_db.history_for(extracted.pro_number, extracted.claimant_email)
        results = [
            rules.check_filing_deadline(shipment, received_on),
            rules.check_commodity(shipment),
            rules.check_pieces(extracted, shipment),
            rules.check_documents(extracted, assessment.damage_type),
            rules.check_exception_noted(shipment, assessment.damage_type, received_on),
            *rules.check_claim_history(extracted.pro_number, extracted.claimant_email, history),
        ]
        payable = rules.settlement_amount(extracted, shipment)

        # 6 [DET] Routing policy decides who acts; rollout stage can only make it more cautious.
        proposed, reasons = route_claim(extracted, assessment, results, payable)
        route = effective_route(proposed, claim_id, self.config)
        if route is not proposed:
            reasons.append(f"rollout {self.config.mode.value}: {proposed.value} downgraded to {route.value}")
        if route is Route.AUTO_DENY_DRAFT:
            payable = Decimal("0.00")

        # 7 [LLM] Personal paragraph for the letter; amounts and legal text come from the template.
        letter = None
        if route in LETTER_TEMPLATES:
            personal = ai_steps.draft_letter_paragraph(self.llm, extracted)
            letter = LETTER_TEMPLATES[route].format(
                name=extracted.claimant_name or "Customer",
                personal=personal,
                pro=extracted.pro_number,
                amount=payable,
                reason="; ".join(
                    DENIAL_REASONS[r.rule].format(detail=r.detail) for r in results if r.rule in DENIAL_REASONS and not r.passed
                ),
            )

        decision = Decision(claim_id, route, payable, results, reasons, extracted, assessment, letter)
        return self._commit(decision)

    def _commit(self, decision: Decision) -> Decision:
        # 8 [DET] Write-back to existing systems; side effects depend on rollout stage.
        self.claims_db.save_decision(decision)
        log.info("audit %s", decision_to_json(decision))

        if not ai_is_visible(self.config):
            return decision  # SHADOW: logged for comparison only; adjusters work as usual

        if decision.route is Route.AUTO_APPROVE:
            # 9 [HITL] Payment is created as *pending*; release stays a human action in the ERP.
            self.erp.create_pending_payment(decision.claim_id, decision.payable_amount)
        else:
            # 9 [HITL] Adjuster or senior adjuster decides, with the AI pre-fill attached.
            queue = "senior" if decision.route is Route.SENIOR_REVIEW else "adjuster"
            self.review_queue.enqueue(decision.claim_id, queue, decision.reasons)
        return decision
