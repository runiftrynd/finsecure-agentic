from __future__ import annotations

from typing import Any

from model_loader import (
    load_fraud_resources,
    load_transaction_dataframe,
)

from tools.fraud_tool import (
    analyze_fraud,
)


def get_transaction(
    transaction_id: str,
) -> dict[str, Any] | None:
    """
    Mengambil transaksi berdasarkan transaction_id.
    """

    clean_transaction_id = str(
        transaction_id
    ).strip().upper()

    if not clean_transaction_id:
        raise ValueError(
            "Transaction ID tidak boleh kosong."
        )

    dataframe = load_transaction_dataframe()

    selected_rows = dataframe[
        dataframe["transaction_id"]
        == clean_transaction_id
    ]

    if selected_rows.empty:
        return None

    return selected_rows.iloc[0].to_dict()


def list_transaction_ids() -> list[str]:
    """
    Mengambil seluruh ID transaksi demo.
    """

    dataframe = load_transaction_dataframe()

    return (
        dataframe["transaction_id"]
        .astype(str)
        .tolist()
    )


def get_customer_transactions(
    customer_id: str,
) -> list[dict[str, Any]]:
    """
    Mengambil semua transaksi milik satu nasabah.
    """

    clean_customer_id = str(
        customer_id
    ).strip().upper()

    if not clean_customer_id:
        raise ValueError(
            "Customer ID tidak boleh kosong."
        )

    dataframe = load_transaction_dataframe()

    selected_rows = dataframe[
        dataframe["customer_id"]
        == clean_customer_id
    ]

    return selected_rows.to_dict(
        orient="records"
    )


def analyze_transaction(
    transaction_id: str,
) -> dict[str, Any]:
    """
    Mengambil transaksi dan menjalankan
    analisis fraud otomatis.
    """

    clean_transaction_id = str(
        transaction_id
    ).strip().upper()

    transaction = get_transaction(
        clean_transaction_id
    )

    if transaction is None:
        return {
            "status": "not_found",
            "transaction_id":
                clean_transaction_id,
            "message":
                "Transaction ID tidak ditemukan.",
            "human_review_required":
                False,
        }

    (
        _,
        fraud_features,
        _,
        _,
    ) = load_fraud_resources()

    input_features = {
        feature: transaction[feature]
        for feature in fraud_features
    }

    fraud_result = analyze_fraud(
        input_features=input_features,
        transaction_id=clean_transaction_id,
    )

    return {
        "status": "success",
        "transaction": {
            "transaction_id":
                transaction["transaction_id"],
            "customer_id":
                transaction["customer_id"],
            "amount":
                float(transaction["TX_AMOUNT"]),
            "transaction_status":
                transaction[
                    "transaction_status"
                ],
            "description":
                transaction["description"],
        },
        "fraud_analysis":
            fraud_result,
    }