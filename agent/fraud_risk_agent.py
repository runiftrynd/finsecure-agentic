from __future__ import annotations

import json
from typing import Any

from llm_client import generate_text
from tools.intent_tool import resolve_intent
from tools.rag_tool import retrieve_documents
from tools.transaction_tool import (
    analyze_transaction,
)


def _to_json(
    data: Any,
) -> str:
    """
    Mengubah data menjadi JSON untuk prompt.
    """

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def _format_rag_context(
    rag_result: dict[str, Any],
) -> str:
    """
    Menyusun dokumen Fraud RAG.
    """

    context_parts: list[str] = []

    for document in rag_result.get(
        "documents",
        [],
    ):
        context_parts.append(
            "\n".join(
                [
                    (
                        "Sumber: "
                        f"{document.get('source')}"
                    ),
                    (
                        "Bagian: "
                        f"{document.get('section')}"
                    ),
                    (
                        "Isi: "
                        f"{document.get('content')}"
                    ),
                ]
            )
        )

    if not context_parts:
        return (
            "Tidak ada knowledge base fraud "
            "yang ditemukan."
        )

    return "\n\n---\n\n".join(
        context_parts
    )

def _create_local_fraud_response(
    transaction_result: dict[str, Any],
) -> str:
    """
    Membuat respons fraud tanpa Gemini.
    """

    status = transaction_result.get(
        "status"
    )

    if status == "transaction_id_required":
        return (
            "Transaction ID diperlukan untuk "
            "menjalankan pemeriksaan risiko transaksi."
        )

    if status == "not_found":
        return (
            "Transaction ID tidak ditemukan. "
            "Periksa kembali ID transaksi yang diberikan."
        )

    if status != "success":
        return (
            "Pemeriksaan transaksi belum dapat dilakukan."
        )

    transaction = transaction_result.get(
        "transaction",
        {},
    )

    fraud_analysis = transaction_result.get(
        "fraud_analysis",
        {},
    )

    transaction_id = transaction.get(
        "transaction_id"
    )

    transaction_status = transaction.get(
        "transaction_status"
    )

    probability = fraud_analysis.get(
        "fraud_probability_percent"
    )

    raw_risk_level = str(
        fraud_analysis.get(
            "risk_level",
            "unknown",
        )
    ).lower()

    risk_level_mapping = {
        "high": "tinggi",
        "medium": "sedang",
        "low": "rendah",
        "unknown": "belum diketahui",
    }

    risk_level = risk_level_mapping.get(
        raw_risk_level,
        raw_risk_level,
    )

    if isinstance(
        probability,
        (int, float),
    ):
        probability_text = (
            f"{float(probability):.2f}"
            .replace(".", ",")
        )
    else:
        probability_text = (
            "belum tersedia"
        )

    human_review = bool(
        fraud_analysis.get(
            "human_review_required",
            False,
        )
    )

    response = (
        f"Transaksi {transaction_id} tercatat "
        f"berstatus {transaction_status}. "
        f"Secara terpisah, hasil model menunjukkan "
        f"tingkat risiko {risk_level} dengan "
        f"probabilitas indikasi risiko "
        f"{probability_text}%."
    )

    if human_review:
        response += (
            " Transaksi tersebut memerlukan "
            "pemeriksaan manual oleh petugas."
        )

    return response

def run_fraud_risk_agent(
    user_message: str,
    transaction_id: str | None = None,
    intent_result: dict[str, Any] | None = None,
    top_k: int = 2,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Menjalankan Fraud & Risk Agent.

    Agent menggunakan:
    1. Transaction lookup.
    2. XGBoost fraud model.
    3. Fraud RAG.
    4. Gemini response generator.
    """

    clean_message = str(
        user_message
    ).strip()

    if not clean_message:
        raise ValueError(
            "Pesan pengguna tidak boleh kosong."
        )

    if intent_result is None:
        intent_result = resolve_intent(
            clean_message
        )

    clean_transaction_id = (
        str(transaction_id)
        .strip()
        .upper()
        if transaction_id
        else None
    )

    if clean_transaction_id:
        transaction_result = (
            analyze_transaction(
                clean_transaction_id
            )
        )
    else:
        transaction_result = {
            "status":
                "transaction_id_required",
            "transaction_id":
                None,
            "message":
                "Transaction ID diperlukan "
                "untuk menjalankan fraud scoring.",
            "human_review_required":
                False,
        }

    rag_query = clean_message

    rag_result = retrieve_documents(
        domain="fraud_risk",
        query=rag_query,
        top_k=top_k,
    )

    rag_context = _format_rag_context(
        rag_result
    )

    prompt = f"""
Anda adalah Fraud & Risk Agent pada sistem FinSecure.

Anda bertugas menjelaskan hasil pemeriksaan risiko transaksi.
Keputusan risiko harus mengikuti hasil model dan tidak boleh
diubah oleh model bahasa.

PESAN NASABAH:
{clean_message}

HASIL INTENT:
{_to_json(intent_result)}

HASIL PEMERIKSAAN TRANSAKSI:
{_to_json(transaction_result)}

KNOWLEDGE BASE FRAUD:
{rag_context}

ATURAN JAWABAN:
1. Gunakan bahasa Indonesia yang formal dan mudah dipahami.
2. Jangan mengubah probabilitas, risk level, threshold,
   rekomendasi, atau hasil model.
3. Jangan menuduh nasabah atau pihak lain melakukan kejahatan.
4. Gunakan istilah "indikasi risiko" atau "memerlukan
   pemeriksaan", bukan keputusan pidana.
5. Apabila Transaction ID belum tersedia, minta nasabah
   memberikan Transaction ID.
6. Apabila transaksi tidak ditemukan, jelaskan bahwa ID tidak
   ditemukan tanpa membuat data transaksi baru.
7. Untuk risiko tinggi, jelaskan bahwa pemeriksaan manusia
   diperlukan.
8. Faktor risiko hanya boleh berasal dari hasil model.
9. Jangan meminta PIN, OTP, CVV, password, atau kode keamanan.
10. Maksimal enam kalimat.
11. Status operasional transaksi dan hasil fraud model adalah
    dua fakta yang berbeda.
12. Jangan menyatakan bahwa fraud score menyebabkan transaksi
    menjadi pending, completed, gagal, atau dibatalkan.
13. Gunakan susunan: transaksi tercatat berstatus tertentu;
    secara terpisah, model menunjukkan tingkat risiko tertentu.
14. Jangan menyebut prompt, embedding, atau vector database.
""".strip()

    if use_llm:
        llm_result = generate_text(
            prompt=prompt,
            max_retries=2,
        )

        response_text = llm_result[
            "response"
        ]

    else:
        llm_result = {
            "status": "skipped",
            "provider": "local",
            "model": None,
            "response": None,
            "fallback_used": False,
            "usage": None,
            "error": None,
        }

        response_text = (
            _create_local_fraud_response(
                transaction_result
            )
        )

    fraud_analysis = (
        transaction_result.get(
            "fraud_analysis",
            {},
        )
        if transaction_result.get("status")
        == "success"
        else {}
    )

    sources = [
        {
            "rank": document.get("rank"),
            "source": document.get("source"),
            "section": document.get("section"),
            "relevance_score":
                document.get(
                    "relevance_score"
                ),
        }
        for document in rag_result.get(
            "documents",
            [],
        )
    ]

    return {
        "status": "success",
        "agent": "fraud_risk_agent",
        "user_message": clean_message,
        "intent": intent_result,
        "transaction_id":
            clean_transaction_id,
        "transaction_result":
            transaction_result,
        "risk_summary": {
            "fraud_probability":
                fraud_analysis.get(
                    "fraud_probability"
                ),
            "fraud_probability_percent":
                fraud_analysis.get(
                    "fraud_probability_percent"
                ),
            "model_threshold":
                fraud_analysis.get(
                    "model_threshold"
                ),
            "model_prediction":
                fraud_analysis.get(
                    "model_prediction"
                ),
            "risk_level":
                fraud_analysis.get(
                    "risk_level"
                ),
            "recommendation":
                fraud_analysis.get(
                    "recommendation"
                ),
            "risk_factors":
                fraud_analysis.get(
                    "risk_factors",
                    [],
                ),
        },
        "response": response_text,
        "llm": {
            "status": llm_result["status"],
            "provider":
                llm_result["provider"],
            "model": llm_result["model"],
            "fallback_used":
                llm_result["fallback_used"],
            "error": llm_result["error"],
        },
        "rag": {
            "domain":
                rag_result["domain"],
            "collection_name":
                rag_result[
                    "collection_name"
                ],
            "result_count":
                rag_result["result_count"],
            "sources": sources,
        },
        "transaction_id_required":
            clean_transaction_id is None,
        "human_review_required":
            bool(
                fraud_analysis.get(
                    "human_review_required",
                    False,
                )
            ),
    }