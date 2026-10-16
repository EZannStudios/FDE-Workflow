"""Model access behind one small interface.

The workflow never imports a vendor SDK directly. It talks to `LLMClient`, so
we can swap models, replay recorded outputs in evals, or run fully offline.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str, *, max_tokens: int = 1024) -> dict:
        """Return the model's answer parsed as a JSON object."""
        ...


class AnthropicClient:
    """Production client. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str | None = None):
        import anthropic  # imported lazily so the offline demo has no dependencies

        self._client = anthropic.Anthropic()
        self._model = model or os.environ.get("MERIDIAN_MODEL", "claude-sonnet-5-5")

    def complete_json(self, system: str, user: str, *, max_tokens: int = 1024) -> dict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=0,  # we want repeatable extractions, not creativity
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _parse_json_object(text)


class HeuristicStubClient:
    """Offline stand-in that imitates a model with regexes.

    It is deliberately imperfect (it misses things a real model would catch),
    which makes it useful for showing what the eval harness reports when an AI
    step is not good enough to ship.
    """

    def complete_json(self, system: str, user: str, *, max_tokens: int = 1024) -> dict:
        if "TASK: extract_claim" in system:
            return self._extract(user)
        if "TASK: classify_damage" in system:
            return self._classify(user)
        if "TASK: draft_letter" in system:
            return {"body": "Thank you for your patience while we reviewed your claim."}
        raise ValueError("Stub does not know this task")

    def _extract(self, text: str) -> dict:
        pro = re.search(r"\bPRO\s*(?:#|number|no\.?)?\s*:?\s*(\d{7,10})", text, re.I)
        amount = re.search(r"\$\s?([\d,]+(?:\.\d{2})?)", text)
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
        name = re.search(r"^(?:Regards|Thanks|Sincerely),?\s*\n\s*(.+)$", text, re.I | re.M)
        pieces = re.search(r"(\d+)\s+(?:of\s+\d+\s+)?(?:pallets?|cartons?|box(?:es)?|pieces?|units?)", text, re.I)
        attachments = re.findall(r"[\w-]+\.(?:pdf|jpg|jpeg|png)", text, re.I)
        evidence = []
        for field_name, match in (("pro_number", pro), ("claimed_amount", amount)):
            if match:
                evidence.append({"field": field_name, "quote": match.group(0)})
        return {
            "pro_number": pro.group(1) if pro else None,
            "claimant_name": name.group(1).strip() if name else None,
            "claimant_email": email.group(0) if email else None,
            "claimed_amount": amount.group(1).replace(",", "") if amount else None,
            "pieces_affected": int(pieces.group(1)) if pieces else None,
            "description": text.strip()[:400],
            "attachments": attachments,
            "evidence": evidence,
            "confidence": 0.9 if pro and amount else 0.5,
        }

    def _classify(self, text: str) -> dict:
        lowered = text.lower()
        keywords = [
            ("water", ("wet", "water", "soaked", "moisture")),
            ("punctured", ("forklift", "puncture", "hole", "pierced")),
            ("shortage", ("missing", "short", "only received")),
            ("crushed", ("crushed", "collapsed", "dented", "smashed")),
        ]
        damage = next((label for label, words in keywords if any(w in lowered for w in words)), "other")
        if "clean" in lowered and "unpack" in lowered:
            damage = "concealed"
        return {
            "damage_type": damage,
            "consistent_with_exception": None,
            "narrative_flags": [],
            "confidence": 0.85 if damage != "other" else 0.3,
        }


def default_client() -> LLMClient:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    return HeuristicStubClient()


def _parse_json_object(text: str) -> dict:
    """Models sometimes wrap JSON in prose or code fences; take the first object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])
