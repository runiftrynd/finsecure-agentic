from __future__ import annotations

import json

from backend import process_finsecure_request


def main() -> None:
    print("=" * 70)
    print("PENGUJIAN BACKEND FINSECURE")
    print("=" * 70)

    result = process_finsecure_request(
        user_message=(
            "Verifikasi identitas KYC0153 "
            "masih ditinjau karena nama berbeda. "
            "Transfer TRX0002 juga pending "
            "dan tidak saya kenali."
        ),
        include_debug=True,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    assert result[
        "status"
    ] in {
        "success",
        "partial_failure",
    }

    assert result[
        "routing"
    ]["mode"] == "multi_agent_3"

    assert result[
        "identifiers"
    ]["transaction_id"] == "TRX0002"

    assert result[
        "identifiers"
    ]["application_id"] == "KYC0153"

    assert result[
        "response"
    ]

    assert result[
        "human_review_required"
    ] is True

    print("\n" + "=" * 70)
    print("PENGUJIAN BACKEND BERHASIL")
    print("=" * 70)


if __name__ == "__main__":
    main()