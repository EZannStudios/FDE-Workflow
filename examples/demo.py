"""Run three claims through the pipeline in ASSIST mode and show what each
existing system receives.

    python -m examples.demo
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from meridian_claims.deployment import Mode, RolloutConfig
from meridian_claims.integrations import ClaimsDBAdapter, ERPAdapter, ReviewQueueAdapter, TMSAdapter
from meridian_claims.llm import default_client
from meridian_claims.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    pipeline = Pipeline(
        llm=default_client(),
        tms=TMSAdapter(ROOT / "data" / "shipments.json"),
        claims_db=ClaimsDBAdapter(json.loads((ROOT / "data" / "claim_history.json").read_text())),
        review_queue=ReviewQueueAdapter(),
        erp=ERPAdapter(),
        config=RolloutConfig(mode=Mode.AUTO, canary_percent=100),
    )
    cases = [json.loads(line) for line in (ROOT / "evals" / "golden_set.jsonl").read_text().splitlines()]
    for case in (cases[0], cases[2], cases[4]):  # clean, late filing, duplicate
        d = pipeline.process(case["claim_id"], case["raw_text"], date.fromisoformat(case["received_on"]))
        print(f"== {d.claim_id}: {d.route.value}, payable ${d.payable_amount}")
        for r in d.rule_results:
            print(f"   [{'pass' if r.passed else 'FAIL'}] {r.rule} {r.detail}")
        print(f"   reasons: {'; '.join(d.reasons)}")
        if d.letter_draft:
            print("   letter draft:\n      " + d.letter_draft.replace("\n", "\n      "))
        print()

    print("ERP pending payments (released by a person in NetSuite):", dict(pipeline.erp.pending_payments))
    print("Adjuster queue:", [(cid, q) for cid, q, _ in pipeline.review_queue.items])


if __name__ == "__main__":
    main()
