"""Human-in-the-loop routing: a deterministic policy over AI and rule outputs.

The model reports what it saw and how sure it is. This table, owned by the
claims manager, decides who acts. Thresholds come from the eval report
(see evals/README.md), not from intuition.
"""

from __future__ import annotations

from decimal import Decimal

from .models import DamageAssessment, ExtractedClaim, Route, RuleResult
from .rules import AUTO_APPROVE_LIMIT, HARD_RULES, SENIOR_REVIEW_THRESHOLD

MIN_EXTRACTION_CONFIDENCE = 0.85
MIN_CLASSIFICATION_CONFIDENCE = 0.80
FRAUD_RULES = {"not_duplicate", "claimant_frequency"}


def route_claim(
    extracted: ExtractedClaim,
    assessment: DamageAssessment,
    results: list[RuleResult],
    payable: Decimal,
) -> tuple[Route, list[str]]:
    failed = {r.rule for r in results if not r.passed}
    reasons = [f"rule failed: {r.rule} ({r.detail})" for r in results if not r.passed]

    # 1. A hard policy rule failed: draft a denial, a human signs it.
    if failed & HARD_RULES:
        return Route.AUTO_DENY_DRAFT, reasons

    # 2. Fraud screens, AI narrative flags, or large amounts: senior adjuster.
    if failed & FRAUD_RULES or assessment.narrative_flags or payable > SENIOR_REVIEW_THRESHOLD:
        reasons += [f"AI flag: {f}" for f in assessment.narrative_flags]
        if payable > SENIOR_REVIEW_THRESHOLD:
            reasons.append(f"payable {payable} above senior threshold")
        return Route.SENIOR_REVIEW, reasons

    # 3. Anything uncertain goes to an adjuster, with the AI pre-fill attached.
    if extracted.grounding_failures:
        reasons += [f"grounding: {g}" for g in extracted.grounding_failures]
    if extracted.confidence < MIN_EXTRACTION_CONFIDENCE:
        reasons.append(f"extraction confidence {extracted.confidence:.2f}")
    if assessment.confidence < MIN_CLASSIFICATION_CONFIDENCE:
        reasons.append(f"classification confidence {assessment.confidence:.2f}")
    if assessment.consistent_with_exception is False:
        reasons.append("story does not match delivery exception")
    if payable > AUTO_APPROVE_LIMIT:
        reasons.append(f"payable {payable} above auto-approve limit")
    if failed or reasons:
        return Route.ADJUSTER_REVIEW, reasons

    # 4. Small, clean, confident, fully grounded: straight-through.
    return Route.AUTO_APPROVE, ["all checks passed"]
