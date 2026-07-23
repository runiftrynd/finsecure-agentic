from __future__ import annotations

import json
from typing import Any

from agents.customer_service_agent import (
    run_customer_service_agent,
)
from agents.fraud_risk_agent import (
    run_fraud_risk_agent,
)
from agents.kyc_compliance_agent import (
    run_kyc_compliance_agent,
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


def test_customer_service_agent() -> None:
    print("\n" + "=" * 70)
    print("TEST CUSTOMER SERVICE AGENT")
    print("=" * 70)

    result = run_customer_service_agent(
        user_message=(
            "Transfer saya masih pending. "
            "Apa yang harus saya lakukan?"
        ),
        top_k=2,
    )

    print_json(result)


def test_fraud_risk_agent() -> None:
    print("\n" + "=" * 70)
    print("TEST FRAUD & RISK AGENT")
    print("=" * 70)

    result = run_fraud_risk_agent(
        user_message=(
            "Saya tidak mengenali transaksi "
            "ini dan dilakukan pada malam hari."
        ),
        transaction_id="TRX0002",
        top_k=2,
    )

    print_json(result)


def test_kyc_compliance_agent() -> None:
    print("\n" + "=" * 70)
    print("TEST KYC & COMPLIANCE AGENT")
    print("=" * 70)

    result = run_kyc_compliance_agent(
        user_message=(
            "Verifikasi identitas saya perlu "
            "ditinjau karena nama sedikit berbeda."
        ),
        application_id="KYC0153",
        top_k=2,
    )

    print_json(result)


def test_missing_identifiers() -> None:
    print("\n" + "=" * 70)
    print("TEST IDENTIFIER TIDAK DIBERIKAN")
    print("=" * 70)

    fraud_result = run_fraud_risk_agent(
        user_message=(
            "Ada transaksi yang tidak "
            "saya kenali."
        ),
        transaction_id=None,
        top_k=1,
    )

    kyc_result = (
        run_kyc_compliance_agent(
            user_message=(
                "Mengapa verifikasi "
                "identitas saya gagal?"
            ),
            application_id=None,
            top_k=1,
        )
    )

    print("\nFraud tanpa Transaction ID:")
    print_json(fraud_result)

    print("\nKYC tanpa Application ID:")
    print_json(kyc_result)


def main() -> None:
    print("=" * 70)
    print("PENGUJIAN WORKER AGENTS FINSECURE")
    print("=" * 70)

    test_customer_service_agent()
    test_fraud_risk_agent()
    test_kyc_compliance_agent()
    test_missing_identifiers()

    print("\n" + "=" * 70)
    print("SEMUA PENGUJIAN WORKER AGENTS SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()