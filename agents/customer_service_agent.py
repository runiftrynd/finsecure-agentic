from __future__ import annotations

from typing import Any

from llm_client import generate_text
from tools.intent_tool import resolve_intent
from tools.rag_tool import retrieve_documents


CUSTOMER_SERVICE_INTENTS = {
    "account_blocked",
    "balance_not_updated",
    "card_arrival",
    "change_pin",
    "lost_card",
    "transfer_failed",
    "transfer_pending",
}

LOCAL_CUSTOMER_RESPONSES = {
    "card_arrival": (
        "Status pengiriman kartu baru dapat diperiksa melalui "
        "menu kartu pada aplikasi FinSecure. Waktu kedatangan "
        "bergantung pada proses penerbitan, alamat tujuan, dan "
        "layanan kurir. Apabila kartu belum diterima sesuai "
        "estimasi, siapkan nomor permohonan kartu untuk "
        "pemeriksaan lebih lanjut."
    ),
    "change_pin": (
        "PIN dapat diganti melalui menu keamanan pada aplikasi. "
        "Nasabah tidak boleh memberikan PIN atau OTP kepada "
        "pihak lain."
    ),
    "lost_card": (
        "Kartu yang hilang harus segera diblokir melalui "
        "aplikasi atau layanan nasabah. Jangan memberikan PIN, "
        "OTP, CVV, atau password kepada pihak lain."
    ),
    "transfer_failed": (
        "Periksa saldo, status rekening tujuan, koneksi, dan "
        "batas transaksi. Pastikan transaksi sebelumnya tidak "
        "berhasil sebelum mencoba kembali."
    ),
    "transfer_pending": (
        "Transfer berstatus pending masih diproses atau "
        "diverifikasi. Periksa riwayat transaksi dan jangan "
        "mengulangi transfer sebelum statusnya dipastikan."
    ),
    "balance_not_updated": (
        "Periksa riwayat transaksi dan muat ulang aplikasi. "
        "Saldo dapat terlambat diperbarui karena proses "
        "settlement atau gangguan sistem."
    ),
    "account_blocked": (
        "Akun yang diblokir memerlukan verifikasi melalui kanal "
        "resmi FinSecure. Jangan memberikan PIN, OTP, password, "
        "atau kode keamanan kepada pihak lain."
    ),
}

FRAUD_INTENTS = {
    "suspicious_transaction",
}

KYC_INTENTS = {
    "kyc_failed",
    "update_identity",
}


def _format_rag_context(
    rag_result: dict[str, Any],
) -> str:
    """
    Menyusun hasil RAG menjadi konteks prompt.
    """

    context_parts: list[str] = []

    for document in rag_result.get(
        "documents",
        [],
    ):
        source = (
            document.get("source")
            or "Sumber tidak diketahui"
        )

        section = (
            document.get("section")
            or "Bagian tidak diketahui"
        )

        content = (
            document.get("content")
            or ""
        )

        context_parts.append(
            "\n".join(
                [
                    f"Sumber: {source}",
                    f"Bagian: {section}",
                    f"Isi: {content}",
                ]
            )
        )

    if not context_parts:
        return (
            "Tidak ada knowledge base yang "
            "berhasil ditemukan."
        )

    return "\n\n---\n\n".join(
        context_parts
    )


def _get_handoff_target(
    intent: str,
) -> str | None:
    """
    Menentukan apakah permintaan lebih tepat
    diteruskan ke agent domain lain.
    """

    if intent in FRAUD_INTENTS:
        return "fraud_risk_agent"

    if intent in KYC_INTENTS:
        return "kyc_compliance_agent"

    return None

def _create_local_customer_response(
    rag_result: dict[str, Any],
    intent: str,
) -> str:
    """
    Membuat respons lokal berdasarkan intent.
    RAG digunakan sebagai cadangan jika intent tidak memiliki
    respons lokal.
    """

    intent_response = LOCAL_CUSTOMER_RESPONSES.get(
        intent
    )

    if intent_response:
        return intent_response

    documents = rag_result.get(
        "documents",
        [],
    )

    if not documents:
        return (
            "Permintaan layanan telah diterima, "
            "tetapi panduan yang sesuai belum ditemukan."
        )

    primary_content = str(
        documents[0].get(
            "content",
            "",
        )
    ).strip()

    if not primary_content:
        return (
            "Permintaan layanan telah diterima dan "
            "memerlukan pemeriksaan lebih lanjut."
        )

    lines = [
        line.strip()
        for line in primary_content.splitlines()
        if line.strip()
    ]

    if (
        lines
        and lines[0].lower().startswith("sop ")
    ):
        lines = lines[1:]

    return " ".join(
        " ".join(lines).split()
    )

def run_customer_service_agent(
    user_message: str,
    intent_result: dict[str, Any] | None = None,
    top_k: int = 2,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Menjalankan Customer Service Agent.

    Agent menggunakan:
    1. IndoBERT untuk intent.
    2. RAG Customer Service.
    3. Gemini untuk menyusun jawaban.
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

    detected_intent = str(
        intent_result.get(
            "intent",
            "unknown",
        )
    )

    handoff_target = _get_handoff_target(
        detected_intent
    )

    rag_result = retrieve_documents(
        domain="customer_service",
        query=clean_message,
        top_k=top_k,
    )

    rag_context = _format_rag_context(
        rag_result
    )

    prompt = f"""
Anda adalah Customer Service Agent pada sistem FinSecure.

Tugas Anda adalah menjawab pertanyaan layanan keuangan umum
berdasarkan knowledge base yang diberikan.

PESAN NASABAH:
{clean_message}

HASIL KLASIFIKASI INTENT:
- Intent: {detected_intent}
- Confidence: {intent_result.get("confidence")}
- Low confidence: {intent_result.get("low_confidence")}

KNOWLEDGE BASE:
{rag_context}

ATURAN JAWABAN:
1. Gunakan bahasa Indonesia yang formal dan mudah dipahami.
2. Gunakan hanya informasi yang didukung knowledge base.
3. Jangan meminta PIN, OTP, CVV, password, atau kode keamanan.
4. Jangan membuat status transaksi yang tidak tersedia.
5. Jangan menyatakan bahwa suatu transaksi berhasil atau gagal
   tanpa data pendukung.
6. Maksimal lima kalimat.
7. Apabila kasus membutuhkan pemeriksaan agent lain, jelaskan
   bahwa kasus perlu diteruskan tanpa membuat keputusan sendiri.
8. Jangan menyebut istilah internal seperti embedding,
   vectorstore, prompt, atau model bahasa.
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
            _create_local_customer_response(
                rag_result=rag_result,
                intent=detected_intent
            )
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
        "agent": "customer_service_agent",
        "user_message": clean_message,
        "intent": intent_result,
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
        "handoff_required":
            handoff_target is not None,
        "handoff_target":
            handoff_target,
        "human_review_required":
            False,
    }
