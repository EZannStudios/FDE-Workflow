"""Unit tests for the deterministic core and the AI error-handling paths.

    python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from meridian_claims import ai_steps, rules
from meridian_claims.deployment import Mode, RolloutConfig, effective_route, in_canary
from meridian_claims.errors import (
    CircuitBreaker,
    CircuitOpenError,
    MalformedOutputError,
    SystemOfRecordError,
    TransientModelError,
    with_retries,
)
from meridian_claims.integrations import ClaimsDBAdapter, ERPAdapter, ReviewQueueAdapter, TMSAdapter
from meridian_claims.llm import HeuristicStubClient
from meridian_claims.models import Evidence, ExtractedClaim, Route, ShipmentRecord
from meridian_claims.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent


def shipment(**overrides) -> ShipmentRecord:
    base = dict(
        pro_number="81234567", shipper="S", consignee="C", delivered_on=date(2026, 9, 2),
        weight_lbs=Decimal("1200"), commodity="furniture", released_value_per_lb=Decimal("1.00"),
        declared_value=Decimal("6400"), delivery_exception="1 carton crushed", pieces=6,
    )
    base.update(overrides)
    return ShipmentRecord(**base)


def claim(**overrides) -> ExtractedClaim:
    base = dict(
        pro_number="81234567", claimant_name="Dana", claimant_email="d@x.example",
        claimed_amount=Decimal("340.00"), pieces_affected=1, description="crushed",
    )
    base.update(overrides)
    return ExtractedClaim(**base)


class DeterministicRules(unittest.TestCase):
    def test_filing_deadline_is_nine_months(self):
        s = shipment(delivered_on=date(2026, 1, 15))
        self.assertTrue(rules.check_filing_deadline(s, date(2026, 10, 15)).passed)
        self.assertFalse(rules.check_filing_deadline(s, date(2026, 10, 16)).passed)

    def test_add_months_clamps_end_of_month(self):
        self.assertEqual(rules.add_months(date(2026, 5, 31), 9), date(2027, 2, 28))

    def test_settlement_is_capped_by_released_value_and_prorated(self):
        # $1.00/lb * 1200 lb * (1 of 6 pieces) = $200, below the $340 claimed.
        self.assertEqual(rules.settlement_amount(claim(), shipment()), Decimal("200.00"))

    def test_full_value_declared_pays_claimed_amount(self):
        s = shipment(released_value_per_lb=None)
        self.assertEqual(rules.settlement_amount(claim(), s), Decimal("340.00"))

    def test_more_pieces_than_shipped_fails(self):
        self.assertFalse(rules.check_pieces(claim(pieces_affected=9), shipment()).passed)

    def test_duplicate_pro_is_flagged(self):
        results = rules.check_claim_history("81234567", None, [{"pro_number": "81234567"}])
        self.assertFalse(results[0].passed)


class AIOutputValidation(unittest.TestCase):
    def test_hallucinated_quote_is_caught(self):
        c = claim(evidence=[Evidence("pro_number", "PRO 99999999"), Evidence("claimed_amount", "$340.00")])
        failures = ai_steps.check_grounding(c, "PRO 81234567, claiming $340.00")
        self.assertEqual(failures, ["quote for pro_number not found in source"])

    def test_unsupported_amount_is_caught(self):
        c = claim(evidence=[Evidence("pro_number", "PRO 81234567")])
        self.assertIn("claimed_amount has no supporting quote", ai_steps.check_grounding(c, "PRO 81234567"))

    def test_schema_violation_raises_malformed(self):
        class BadClient:
            def complete_json(self, system, user, max_tokens=1024):
                return {"claimed_amount": "three hundred", "confidence": 0.9}

        with self.assertRaises(MalformedOutputError):
            ai_steps.extract_claim(BadClient(), "anything")

    def test_letter_falls_back_when_model_mentions_money(self):
        class Chatty:
            def complete_json(self, system, user, max_tokens=1024):
                return {"body": "Great news, we're paying you $5,000!"}

        self.assertNotIn("$", ai_steps.draft_letter_paragraph(Chatty(), claim()))


class RetriesAndCircuitBreaker(unittest.TestCase):
    def test_transient_errors_are_retried(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise TransientModelError("429")
            return "ok"

        self.assertEqual(with_retries(flaky, sleep=lambda _: None), "ok")
        self.assertEqual(len(calls), 3)

    def test_system_of_record_errors_are_not_retried(self):
        calls = []

        def bad_tms_row():
            calls.append(1)
            raise SystemOfRecordError("malformed row")

        with self.assertRaises(SystemOfRecordError):
            with_retries(bad_tms_row, sleep=lambda _: None)
        self.assertEqual(len(calls), 1)

    def test_breaker_opens_then_recovers(self):
        now = [0.0]
        breaker = CircuitBreaker(threshold=2, cooldown_s=10, clock=lambda: now[0])

        def fail():
            raise TransientModelError("down")

        for _ in range(2):
            with self.assertRaises(TransientModelError):
                breaker.call(fail)
        with self.assertRaises(CircuitOpenError):
            breaker.call(lambda: "ok")
        now[0] = 11
        self.assertEqual(breaker.call(lambda: "ok"), "ok")


class Rollout(unittest.TestCase):
    def test_canary_bucket_is_stable(self):
        self.assertEqual(in_canary("C-1", 50), in_canary("C-1", 50))

    def test_auto_approve_is_downgraded_outside_auto_mode(self):
        cfg = RolloutConfig(mode=Mode.ASSIST, canary_percent=100)
        self.assertIs(effective_route(Route.AUTO_APPROVE, "C-1", cfg), Route.ADJUSTER_REVIEW)

    def test_denials_are_never_upgraded(self):
        cfg = RolloutConfig(mode=Mode.AUTO, canary_percent=100)
        self.assertIs(effective_route(Route.AUTO_DENY_DRAFT, "C-1", cfg), Route.AUTO_DENY_DRAFT)


class PipelineFallbacks(unittest.TestCase):
    def make(self, llm, mode=Mode.ASSIST):
        return Pipeline(
            llm=llm,
            tms=TMSAdapter(ROOT / "data" / "shipments.json"),
            claims_db=ClaimsDBAdapter(json.loads((ROOT / "data" / "claim_history.json").read_text())),
            review_queue=ReviewQueueAdapter(),
            erp=ERPAdapter(),
            config=RolloutConfig(mode=mode, canary_percent=100),
        )

    def test_model_outage_routes_to_human_not_crash(self):
        class Down:
            def complete_json(self, system, user, max_tokens=1024):
                raise TimeoutError("model timed out")

        p = self.make(Down())
        p.retry_sleep = lambda _: None
        decision = p.process("C-T1", "PRO 81234567 claim $340.00", date(2026, 9, 5))
        self.assertIs(decision.route, Route.ADJUSTER_REVIEW)
        self.assertIn("AI intake failed", decision.reasons[0])
        self.assertEqual(p.review_queue.items[0][0], "C-T1")
        self.assertEqual(p.erp.pending_payments, {})

    def test_shadow_mode_has_no_side_effects(self):
        p = self.make(HeuristicStubClient(), mode=Mode.SHADOW)
        text = "PRO 81234567. 1 carton crushed. Claim $340.00. invoice_1.pdf, a.jpg\n\nRegards,\nDana"
        p.process("C-S1", text, date(2026, 9, 5))
        self.assertEqual(p.review_queue.items, [])
        self.assertEqual(p.erp.pending_payments, {})
        self.assertIn("C-S1", p.claims_db.decisions)  # but it is logged for comparison

    def test_unknown_pro_goes_to_manual(self):
        p = self.make(HeuristicStubClient())
        decision = p.process("C-U1", "PRO 10000000 crushed box, claim $50.00", date(2026, 9, 5))
        self.assertIs(decision.route, Route.ADJUSTER_REVIEW)
        self.assertIn("not found in TMS", decision.reasons[0])


if __name__ == "__main__":
    unittest.main()
