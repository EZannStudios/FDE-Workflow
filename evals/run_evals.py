"""Eval harness: score each AI step and the end-to-end routing against
adjuster-labelled historical claims, then apply ship gates.

    python -m evals.run_evals                 # offline stub model
    ANTHROPIC_API_KEY=... python -m evals.run_evals   # real model

Exit code is non-zero when a gate fails, so this runs in CI before any
prompt, model, or threshold change is deployed.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

from meridian_claims.deployment import Mode, RolloutConfig
from meridian_claims.integrations import ClaimsDBAdapter, ERPAdapter, ReviewQueueAdapter, TMSAdapter
from meridian_claims.llm import default_client
from meridian_claims.models import Route
from meridian_claims.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent

# Ship gates, agreed with the claims manager during the audit.
GATES = {
    "unsafe_auto_approvals": ("max", 0),  # auto-paid a claim a human would not have: never
    "pro_number_accuracy": ("min", 0.95),  # wrong join key = wrong shipment = wrong money
    "claimed_amount_accuracy": ("min", 0.95),
    "damage_type_accuracy": ("min", 0.80),  # adjusters review most of these anyway
    "route_agreement": ("min", 0.80),
}


def run() -> dict:
    cases = [json.loads(line) for line in (ROOT / "evals" / "golden_set.jsonl").read_text().splitlines() if line]
    pipeline = Pipeline(
        llm=default_client(),
        tms=TMSAdapter(ROOT / "data" / "shipments.json"),
        claims_db=ClaimsDBAdapter(json.loads((ROOT / "data" / "claim_history.json").read_text())),
        review_queue=ReviewQueueAdapter(),
        erp=ERPAdapter(),
        # Evaluate the policy as if fully rolled out, so we see what AUTO *would* do.
        config=RolloutConfig(mode=Mode.AUTO, canary_percent=100),
    )

    tallies = {"pro": 0, "amount": 0, "damage": 0, "damage_n": 0, "route": 0, "unsafe": 0, "auto": 0}
    rows = []
    for case in cases:
        exp = case["expected"]
        decision = pipeline.process(case["claim_id"], case["raw_text"], date.fromisoformat(case["received_on"]))
        got_pro = decision.extracted.pro_number if decision.extracted else None
        got_amount = decision.extracted.claimed_amount if decision.extracted else None
        got_damage = decision.assessment.damage_type.value if decision.assessment else None

        tallies["pro"] += got_pro == exp["pro_number"]
        tallies["amount"] += got_amount == (Decimal(exp["claimed_amount"]) if exp["claimed_amount"] else None)
        if decision.assessment:  # only scored when the step ran
            tallies["damage_n"] += 1
            tallies["damage"] += got_damage == exp["damage_type"]
        tallies["route"] += decision.route.value == exp["route"]
        if decision.route is Route.AUTO_APPROVE:
            tallies["auto"] += 1
            tallies["unsafe"] += exp["route"] != Route.AUTO_APPROVE.value
        rows.append((case["claim_id"], exp["route"], decision.route.value, got_pro, got_damage, decision.reasons[:2]))

    n = len(cases)
    metrics = {
        "cases": n,
        "pro_number_accuracy": tallies["pro"] / n,
        "claimed_amount_accuracy": tallies["amount"] / n,
        "damage_type_accuracy": tallies["damage"] / max(tallies["damage_n"], 1),
        "route_agreement": tallies["route"] / n,
        "unsafe_auto_approvals": tallies["unsafe"],
        "automation_rate": tallies["auto"] / n,
    }

    print(f"{'case':<5} {'expected':<16} {'got':<16} {'pro':<10} {'damage':<10} reasons")
    for cid, exp_route, got_route, pro, dmg, reasons in rows:
        mark = " " if exp_route == got_route else "x"
        print(f"{cid:<5} {exp_route:<16} {got_route:<16} {str(pro):<10} {str(dmg):<10} {mark} {'; '.join(reasons)}")
    print()
    for key, value in metrics.items():
        print(f"{key:<26} {value:.2f}" if isinstance(value, float) else f"{key:<26} {value}")
    return metrics


def check_gates(metrics: dict) -> list[str]:
    failures = []
    for name, (kind, threshold) in GATES.items():
        value = metrics[name]
        if (kind == "min" and value < threshold) or (kind == "max" and value > threshold):
            failures.append(f"{name}={value:.2f} (needs {kind} {threshold})")
    return failures


if __name__ == "__main__":
    failed = check_gates(run())
    print()
    if failed:
        print("SHIP GATES FAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print("All ship gates passed.")
