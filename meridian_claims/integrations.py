"""Adapters for the systems Meridian already runs. The AI layer sits between
them; it does not replace any of them.

  Shared claims inbox (Microsoft 365)  -> InboxAdapter      read new claims, label them
  TMS (shipment system of record)      -> TMSAdapter        look up the shipment by PRO
  Claims module in the TMS             -> ClaimsDBAdapter   claim history, write decisions
  Document store (SharePoint)          -> DocStoreAdapter   attachments, tagged by type
  ERP (NetSuite) accounts payable      -> ERPAdapter        create payment, human-released
  Adjuster work queue (TMS claims UI)  -> ReviewQueueAdapter put a pre-filled file in front of a person

The in-memory versions below back the demo and tests. Each method documents
the real call it stands for.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from .errors import SystemOfRecordError
from .models import Decision, ShipmentRecord


class TMSAdapter:
    """Real call: GET /api/v2/shipments/{pro} on the TMS REST API (read-only key)."""

    def __init__(self, shipments_path: Path):
        raw = json.loads(shipments_path.read_text())
        self._by_pro = {s["pro_number"]: s for s in raw}

    def get_shipment(self, pro_number: str) -> ShipmentRecord | None:
        row = self._by_pro.get(pro_number)
        if row is None:
            return None
        try:
            return ShipmentRecord(
                pro_number=row["pro_number"],
                shipper=row["shipper"],
                consignee=row["consignee"],
                delivered_on=date.fromisoformat(row["delivered_on"]),
                weight_lbs=Decimal(row["weight_lbs"]),
                commodity=row["commodity"],
                released_value_per_lb=Decimal(row["released_value_per_lb"]) if row.get("released_value_per_lb") else None,
                declared_value=Decimal(row["declared_value"]),
                delivery_exception=row.get("delivery_exception"),
                pieces=int(row["pieces"]),
            )
        except (KeyError, ValueError) as exc:
            # Bad data in the system of record is escalated, never patched by a model.
            raise SystemOfRecordError(f"TMS record {pro_number} malformed: {exc}") from exc


class ClaimsDBAdapter:
    """Real calls: SELECT from tms.claims; INSERT into tms.claim_decisions (idempotent on claim_id)."""

    def __init__(self, history: list[dict] | None = None):
        self._history = history or []
        self.decisions: dict[str, Decision] = {}

    def history_for(self, pro_number: str | None, claimant_email: str | None) -> list[dict]:
        return [
            h for h in self._history
            if h["pro_number"] == pro_number or (claimant_email and h.get("claimant_email") == claimant_email)
        ]

    def save_decision(self, decision: Decision) -> None:
        # Upsert by claim_id so a retried pipeline run never creates a second record.
        self.decisions[decision.claim_id] = decision


class ReviewQueueAdapter:
    """Real call: POST /api/v2/claims/{id}/tasks to the adjuster queue, with the AI
    pre-fill attached as a side panel the adjuster can accept, edit, or reject."""

    def __init__(self):
        self.items: list[tuple[str, str, list[str]]] = []

    def enqueue(self, claim_id: str, queue: str, reasons: list[str]) -> None:
        self.items.append((claim_id, queue, reasons))


class ERPAdapter:
    """Real call: NetSuite vendor-bill API. Creates a bill in *pending approval*
    status; release to payment is a human click inside NetSuite, unchanged."""

    def __init__(self):
        self.pending_payments: dict[str, Decimal] = {}

    def create_pending_payment(self, claim_id: str, amount: Decimal) -> None:
        if amount <= 0:
            raise SystemOfRecordError("refusing to create a non-positive payment")
        self.pending_payments.setdefault(claim_id, amount)  # idempotent


def decision_to_json(decision: Decision) -> str:
    """Audit log line written for every claim, whichever route it took."""
    return json.dumps(asdict(decision), default=str)
