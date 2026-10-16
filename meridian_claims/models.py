"""Typed records that flow through the claims workflow.

Every AI step returns one of these dataclasses, never free text. That is the
contract that lets deterministic code downstream trust (and check) its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum


class DamageType(str, Enum):
    CRUSHED = "crushed"
    PUNCTURED = "punctured"
    WATER = "water"
    SHORTAGE = "shortage"  # pieces missing, not damaged
    CONCEALED = "concealed"  # damage found after a clean delivery receipt
    OTHER = "other"


class Route(str, Enum):
    AUTO_APPROVE = "auto_approve"  # pay without a human touch
    ADJUSTER_REVIEW = "adjuster_review"  # human makes the call, AI pre-filled the file
    SENIOR_REVIEW = "senior_review"  # fraud flags, high value, or policy edge case
    AUTO_DENY_DRAFT = "auto_deny_draft"  # a rule failed; human still signs the letter


@dataclass
class Evidence:
    """A verbatim quote from the source document backing an extracted field."""

    field: str
    quote: str


@dataclass
class ExtractedClaim:
    """Output of the AI intake step (non-deterministic)."""

    pro_number: str | None
    claimant_name: str | None
    claimant_email: str | None
    claimed_amount: Decimal | None
    pieces_affected: int | None
    description: str
    attachments: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0  # model self-report, recalibrated by evals
    grounding_failures: list[str] = field(default_factory=list)


@dataclass
class DamageAssessment:
    """Output of the AI damage-classification step (non-deterministic)."""

    damage_type: DamageType
    consistent_with_exception: bool | None  # does the story match the delivery notes?
    narrative_flags: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ShipmentRecord:
    """System-of-record data from the TMS. Ground truth, never generated."""

    pro_number: str
    shipper: str
    consignee: str
    delivered_on: date
    weight_lbs: Decimal
    commodity: str
    released_value_per_lb: Decimal | None  # None = full value declared
    declared_value: Decimal
    delivery_exception: str | None  # what the driver wrote on the delivery receipt
    pieces: int


@dataclass
class RuleResult:
    rule: str
    passed: bool
    detail: str = ""


@dataclass
class Decision:
    """Final, auditable outcome for one claim."""

    claim_id: str
    route: Route
    payable_amount: Decimal
    rule_results: list[RuleResult]
    reasons: list[str]
    extracted: ExtractedClaim | None = None
    assessment: DamageAssessment | None = None
    letter_draft: str | None = None
