"""Deterministic steps: policy, law, and arithmetic.

These are plain functions on purpose. Same input, same output, unit-testable,
and explainable to a claimant or a regulator line by line. An LLM adds risk
here and nothing else, so none is used.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .models import DamageType, ExtractedClaim, RuleResult, ShipmentRecord

# Policy constants live in one place so the claims manager can review them.
FILING_WINDOW_MONTHS = 9  # Carmack Amendment minimum filing period
CONCEALED_DAMAGE_REPORT_DAYS = 5  # Meridian tariff: report concealed damage within 5 days
EXCLUDED_COMMODITIES = {"currency", "live animals", "live plants", "artwork"}
AUTO_APPROVE_LIMIT = Decimal("500.00")
SENIOR_REVIEW_THRESHOLD = Decimal("5000.00")
MAX_CLAIMS_PER_CLAIMANT_90D = 3


def add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year, month = d.year + month // 12, month % 12 + 1
    # Clamp to the last valid day (e.g. May 31 + 9 months -> Feb 28/29).
    for day in (d.day, 30, 29, 28):
        try:
            return date(year, month, day)
        except ValueError:
            continue
    raise AssertionError("unreachable")


def check_filing_deadline(shipment: ShipmentRecord, received_on: date) -> RuleResult:
    deadline = add_months(shipment.delivered_on, FILING_WINDOW_MONTHS)
    return RuleResult("filing_deadline", received_on <= deadline, f"deadline {deadline.isoformat()}")


def check_commodity(shipment: ShipmentRecord) -> RuleResult:
    excluded = shipment.commodity.lower() in EXCLUDED_COMMODITIES
    return RuleResult("commodity_covered", not excluded, shipment.commodity)


def check_pieces(claim: ExtractedClaim, shipment: ShipmentRecord) -> RuleResult:
    if claim.pieces_affected is None:
        return RuleResult("pieces_plausible", True, "pieces not stated; prorate on full shipment")
    ok = 0 < claim.pieces_affected <= shipment.pieces
    return RuleResult("pieces_plausible", ok, f"{claim.pieces_affected} claimed of {shipment.pieces} shipped")


def required_documents(damage: DamageType) -> set[str]:
    docs = {"invoice"}
    if damage != DamageType.SHORTAGE:
        docs.add("photo")
    return docs


def check_documents(claim: ExtractedClaim, damage: DamageType) -> RuleResult:
    """Attachment *names* are classified by keyword here. In production the doc
    store already tags uploads by type, so this stays deterministic."""
    found = set()
    for name in claim.attachments:
        lowered = name.lower()
        if "invoice" in lowered or lowered.startswith("inv"):
            found.add("invoice")
        if lowered.endswith((".jpg", ".jpeg", ".png")):
            found.add("photo")
    missing = required_documents(damage) - found
    return RuleResult("documents_complete", not missing, f"missing: {sorted(missing)}" if missing else "")


def check_exception_noted(shipment: ShipmentRecord, damage: DamageType, received_on: date) -> RuleResult:
    """Visible damage should be on the delivery receipt. Concealed damage must be
    reported quickly. Failing this doesn't deny the claim; it sends it to a human."""
    if damage == DamageType.CONCEALED:
        late = received_on > shipment.delivered_on + timedelta(days=CONCEALED_DAMAGE_REPORT_DAYS)
        return RuleResult("exception_noted", not late, "concealed damage reported late" if late else "")
    noted = bool(shipment.delivery_exception)
    return RuleResult("exception_noted", noted, "" if noted else "clean delivery receipt")


def check_claim_history(pro_number: str, claimant_email: str | None, history: list[dict]) -> list[RuleResult]:
    """Deterministic fraud screens. These flag; they never deny on their own."""
    duplicate = any(h["pro_number"] == pro_number for h in history)
    recent = sum(1 for h in history if claimant_email and h.get("claimant_email") == claimant_email)
    return [
        RuleResult("not_duplicate", not duplicate, "existing claim on this PRO" if duplicate else ""),
        RuleResult("claimant_frequency", recent < MAX_CLAIMS_PER_CLAIMANT_90D, f"{recent} claims in 90 days"),
    ]


def liability_limit(claim: ExtractedClaim, shipment: ShipmentRecord) -> Decimal:
    """Carrier's maximum liability under the bill of lading."""
    if shipment.released_value_per_lb is None:
        limit = shipment.declared_value
    else:
        limit = shipment.released_value_per_lb * shipment.weight_lbs
    if claim.pieces_affected:
        limit = limit * Decimal(claim.pieces_affected) / Decimal(shipment.pieces)
    return limit.quantize(Decimal("0.01"), ROUND_HALF_UP)


def settlement_amount(claim: ExtractedClaim, shipment: ShipmentRecord) -> Decimal:
    if claim.claimed_amount is None:
        return Decimal("0.00")
    return min(claim.claimed_amount, liability_limit(claim, shipment)).quantize(Decimal("0.01"))


HARD_RULES = {"filing_deadline", "commodity_covered"}  # failing these means denial
