from __future__ import annotations

import json
from typing import Any

from agents.supervisor_agent import (
    run_supervisor_agent,
)


def print_json(
    data: Any,
) -> None:
    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def print_summary(
    result: dict[str, Any],
) -> None:
    summary = {
        "status":
            result.get("status"),
        "intent":
            result.get(
                "intent",
                {},
            ).get("intent"),
        "routing_mode":
            result.get(
                "route",
                {},
            ).get("routing_mode"),
        "planned_agents":
            result.get(
                "route",
                {},
            ).get("planned_agents"),
        "executed_agents":
            result.get(
                "executed_agents"
            ),
        "transaction_id":
            result.get(
                "transaction_id"
            ),
        "application_id":
            result.get(
                "application_id"
            ),
        "human_review_required":
            result.get(
                "human_review_required"
            ),
        "final_response":
            result.get(
                "final_response"
            ),
        "execution_errors":
            result.get(
                "execution_errors"
            ),
    }

    print_json(summary)


def test_single_agent() -> None:
    print("\n" + "=" * 70)
    print("TEST SUPERVISOR: SINGLE AGENT")
    print("=" * 70)

    result = run_supervisor_agent(
        user_message=(
            "Bagaimana cara mengganti PIN?"
        )
    )

    print_summary(result)


def test_three_agents() -> None:
    print("\n" + "=" * 70)
    print("TEST SUPERVISOR: THREE AGENTS")
    print("=" * 70)

    result = run_supervisor_agent(
        user_message=(
            "Verifikasi identitas KYC0153 "
            "masih ditinjau karena nama berbeda. "
            "Selain itu, transfer TRX0002 masih "
            "pending dan tidak saya kenali."
        )
    )

    print_summary(result)


def test_missing_identifiers() -> None:
    print("\n" + "=" * 70)
    print("TEST SUPERVISOR: IDENTIFIER KOSONG")
    print("=" * 70)

    result = run_supervisor_agent(
        user_message=(
            "Verifikasi identitas saya gagal "
            "dan ada transaksi yang tidak "
            "saya kenali."
        )
    )

    print_summary(result)


def main() -> None:
    print("=" * 70)
    print("PENGUJIAN SUPERVISOR AGENT FINSECURE")
    print("=" * 70)

    test_single_agent()
    test_three_agents()
    test_missing_identifiers()

    print("\n" + "=" * 70)
    print("SEMUA PENGUJIAN SUPERVISOR SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()