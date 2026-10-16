"""The three steps where Meridian uses an LLM, and only these three.

Each step follows the same pattern:
  1. Prompt the model for JSON that matches a fixed schema.
  2. Validate the JSON (types, enums, ranges). Bad output -> MalformedOutputError.
  3. Check the output against the source (grounding). Made-up evidence is recorded
     on the record and the router sends the claim to a person.
  4. Return a typed record. Callers decide what to do with low confidence.

Nothing in this module approves, denies, or prices a claim.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation

from .errors import MalformedOutputError, TransientModelError
from .llm import LLMClient
from .models import DamageAssessment, DamageType, Evidence, ExtractedClaim, ShipmentRecord

EXTRACT_SYSTEM = """TASK: extract_claim
You read freight damage claims sent to Meridian Freight by email or web form.
Return ONLY a JSON object with these keys:
  pro_number (string of digits or null), claimant_name (string or null),
  claimant_email (string or null), claimed_amount (decimal string, no $ or commas, or null),
  pieces_affected (integer or null), description (one-paragraph summary),
  attachments (list of filenames), confidence (0-1),
  evidence (list of {"field": <key>, "quote": <exact text copied from the claim>}).
Rules: never guess. If a value isn't stated, use null. Every non-null pro_number and
claimed_amount MUST have an evidence quote copied character-for-character from the claim."""

CLASSIFY_SYSTEM = """TASK: classify_damage
Classify the damage described in a freight claim. Return ONLY JSON:
  damage_type: one of crushed | punctured | water | shortage | concealed | other
  consistent_with_exception: true/false/null. Does the claimant's story match the
      driver's delivery exception note? null if there is no note.
  narrative_flags: list of short strings for anything an adjuster should look at
      (e.g. "claimed pieces exceed pieces shipped", "story changed between emails").
  confidence: 0-1
You are flagging, not deciding. Do not recommend approval or denial."""

LETTER_SYSTEM = """TASK: draft_letter
Write the personal paragraph of a claim decision letter: plain, warm, specific to
the claimant's situation, max 120 words. Do NOT state amounts, legal terms, or
the decision itself; those are inserted by the template. Return ONLY JSON: {"body": "..."}"""


def _call(client: LLMClient, system: str, user: str) -> dict:
    """Normalise client failures into our error taxonomy."""
    try:
        return client.complete_json(system, user)
    except (TimeoutError, ConnectionError) as exc:
        raise TransientModelError(str(exc)) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise MalformedOutputError(str(exc)) from exc


def extract_claim(client: LLMClient, raw_text: str) -> ExtractedClaim:
    """LLM judgment: turn a messy email into structured fields."""
    data = _call(client, EXTRACT_SYSTEM, raw_text)

    try:
        amount = Decimal(data["claimed_amount"]) if data.get("claimed_amount") not in (None, "") else None
        pieces = int(data["pieces_affected"]) if data.get("pieces_affected") is not None else None
        confidence = float(data.get("confidence", 0))
        evidence = [Evidence(e["field"], e["quote"]) for e in data.get("evidence", [])]
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise MalformedOutputError(f"extraction schema violation: {exc}") from exc

    if amount is not None and amount < 0:
        raise MalformedOutputError("negative claimed_amount")
    if not 0 <= confidence <= 1:
        raise MalformedOutputError("confidence out of range")

    claim = ExtractedClaim(
        pro_number=_digits_or_none(data.get("pro_number")),
        claimant_name=data.get("claimant_name"),
        claimant_email=data.get("claimant_email"),
        claimed_amount=amount,
        pieces_affected=pieces,
        description=str(data.get("description", "")),
        attachments=list(data.get("attachments", [])),
        evidence=evidence,
        confidence=confidence,
    )
    claim.grounding_failures = check_grounding(claim, raw_text)
    return claim


def check_grounding(claim: ExtractedClaim, raw_text: str) -> list[str]:
    """Deterministic check on an AI output: every quote must appear in the source,
    and the two money-critical fields must be backed by a quote."""
    failures = []
    normalized_source = " ".join(raw_text.split())
    cited = {e.field for e in claim.evidence}
    for ev in claim.evidence:
        if " ".join(ev.quote.split()) not in normalized_source:
            failures.append(f"quote for {ev.field} not found in source")
    for required in ("pro_number", "claimed_amount"):
        if getattr(claim, required) is not None and required not in cited:
            failures.append(f"{required} has no supporting quote")
    return failures


def classify_damage(client: LLMClient, claim: ExtractedClaim, shipment: ShipmentRecord) -> DamageAssessment:
    """LLM judgment: what kind of damage is this, and does the story hang together?"""
    user = (
        f"Claimant description:\n{claim.description}\n\n"
        f"Pieces claimed: {claim.pieces_affected}  |  Pieces shipped: {shipment.pieces}\n"
        f"Commodity: {shipment.commodity}\n"
        f"Driver delivery exception note: {shipment.delivery_exception or 'none (clean delivery)'}"
    )
    data = _call(client, CLASSIFY_SYSTEM, user)
    try:
        damage = DamageType(data["damage_type"])
        consistent = data.get("consistent_with_exception")
        if consistent not in (True, False, None):
            raise ValueError("consistent_with_exception must be bool or null")
        assessment = DamageAssessment(
            damage_type=damage,
            consistent_with_exception=consistent,
            narrative_flags=[str(f) for f in data.get("narrative_flags", [])],
            confidence=float(data.get("confidence", 0)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise MalformedOutputError(f"classification schema violation: {exc}") from exc
    return assessment


def draft_letter_paragraph(client: LLMClient, claim: ExtractedClaim) -> str:
    """LLM judgment for tone only. Falls back to a fixed sentence if the model fails,
    because a plain letter is always better than a late one."""
    try:
        data = _call(client, LETTER_SYSTEM, claim.description)
        body = str(data["body"]).strip()
    except (MalformedOutputError, TransientModelError, KeyError):
        return "Thank you for your patience while we reviewed your claim."
    if "$" in body or len(body.split()) > 150:
        # The model strayed into amounts or rambled; the template owns those.
        return "Thank you for your patience while we reviewed your claim."
    return body


def _digits_or_none(value: object) -> str | None:
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits or None
