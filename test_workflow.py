from __future__ import annotations

import json
from typing import Any

from workflow import (
    run_finsecure_workflow,
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
        "workflow":
            result.get("agent"),
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
        "synthesis_llm":
            result.get(
                "synthesis_llm"
            ),
        "execution_errors":
            result.get(
                "execution_errors"
            ),
    }

    print_json(summary)


def test_single_agent() -> None:
    print("\n" + "=" * 70)
    print("TEST LANGGRAPH: SINGLE AGENT")
    print("=" * 70)

    result = run_finsecure_workflow(
        user_message=(
            "Bagaimana cara mengganti PIN?"
        )
    )

    assert (
        result["route"]["routing_mode"]
        == "single_agent"
    )

    assert result[
        "executed_agents"
    ] == [
        "customer_service_agent"
    ]

    print_summary(result)


def test_three_agents() -> None:
    print("\n" + "=" * 70)
    print("TEST LANGGRAPH: THREE AGENTS")
    print("=" * 70)

    result = run_finsecure_workflow(
        user_message=(
            "Verifikasi identitas KYC0153 "
            "masih ditinjau karena nama berbeda. "
            "Selain itu, transfer TRX0002 masih "
            "pending dan tidak saya kenali."
        )
    )

    assert (
        result["route"]["routing_mode"]
        == "multi_agent_3"
    )

    assert result[
        "executed_agents"
    ] == [
        "customer_service_agent",
        "fraud_risk_agent",
        "kyc_compliance_agent",
    ]

    assert (
        result["transaction_id"]
        == "TRX0002"
    )

    assert (
        result["application_id"]
        == "KYC0153"
    )

    assert (
        result["human_review_required"]
        is True
    )

    print_summary(result)


def test_missing_identifiers() -> None:
    print("\n" + "=" * 70)
    print("TEST LANGGRAPH: IDENTIFIER KOSONG")
    print("=" * 70)

    result = run_finsecure_workflow(
        user_message=(
            "Verifikasi identitas saya gagal "
            "dan ada transaksi yang tidak "
            "saya kenali."
        )
    )

    assert (
        result["route"]["routing_mode"]
        == "multi_agent_2"
    )

    assert (
        result["transaction_id"]
        is None
    )

    assert (
        result["application_id"]
        is None
    )

    print_summary(result)


def main() -> None:
    print("=" * 70)
    print("PENGUJIAN LANGGRAPH FINSECURE")
    print("=" * 70)

    test_single_agent()
    test_three_agents()
    test_missing_identifiers()

    print("\n" + "=" * 70)
    print("SEMUA PENGUJIAN LANGGRAPH SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()