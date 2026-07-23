from __future__ import annotations

import json
from typing import Any

from llm_client import generate_text
from tools.intent_tool import resolve_intent
from tools.kyc_tool import (
    validate_kyc_application,
)
from tools.rag_tool import retrieve_documents


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
    Menyusun dokumen KYC RAG.
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
            "Tidak ada knowledge base KYC "
            "yang ditemukan."
        )

    return "\n\n---\n\n".join(
        context_parts
    )

def _create_local_kyc_response(
    kyc_result: dict[str, Any],
) -> str:
    """
    Membuat respons KYC tanpa Gemini.
    """

    status = kyc_result.get(
        "status"
    )

    if status == "application_id_required":
        return (
            "Application ID diperlukan untuk "
            "memeriksa status verifikasi identitas."
        )

    if status == "not_found":
        return (
            "Application ID tidak ditemukan. "
            "Periksa kembali ID pengajuan yang diberikan."
        )

    application_id = kyc_result.get(
        "application_id"
    )

    issues = kyc_result.get(
        "issues",
        [],
    )

    issue_mapping = {
        "minor_name_variation": (
            "terdapat variasi kecil pada "
            "penulisan nama"
        ),
        "significant_name_mismatch": (
            "terdapat perbedaan nama yang signifikan"
        ),
        "nik_mismatch": (
            "NIK pada formulir tidak sesuai "
            "dengan dokumen"
        ),
        "invalid_nik_format": (
            "format NIK tidak valid"
        ),
        "birth_date_mismatch": (
            "tanggal lahir pada formulir tidak sesuai "
            "dengan dokumen"
        ),
        "document_not_active": (
            "dokumen identitas tidak aktif"
        ),
        "missing_required_fields": (
            "terdapat data wajib yang belum tersedia"
        ),
    }

    translated_issues = [
        issue_mapping.get(
            str(issue),
            str(issue).replace(
                "_",
                " ",
            ),
        )
        for issue in issues
    ]

    issue_text = (
        "; ".join(
            translated_issues
        )
        if translated_issues
        else "tidak ditemukan masalah utama"
    )

    human_review = bool(
        kyc_result.get(
            "human_review_required",
            False,
        )
    )

    response = (
        f"Pengajuan {application_id} "
        f"berstatus {status}. "
        f"Hasil pemeriksaan menunjukkan bahwa "
        f"{issue_text}."
    )

    if human_review:
        response += (
            " Pengajuan tersebut memerlukan "
            "pemeriksaan manual oleh petugas."
        )

    return response

def run_kyc_compliance_agent(
    user_message: str,
    application_id: str | None = None,
    intent_result: dict[str, Any] | None = None,
    top_k: int = 2,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Menjalankan KYC & Compliance Agent.

    Agent menggunakan:
    1. KYC validation rules.
    2. KYC RAG.
    3. Gemini response generator.
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

    clean_application_id = (
        str(application_id)
        .strip()
        .upper()
        if application_id
        else None
    )

    if clean_application_id:
        kyc_result = (
            validate_kyc_application(
                clean_application_id
            )
        )
    else:
        kyc_result = {
            "status":
                "application_id_required",
            "application_id":
                None,
            "message":
                "Application ID diperlukan "
                "untuk memeriksa status KYC.",
            "human_review_required":
                False,
        }

    rag_query = clean_message

    kyc_issues = set(
        kyc_result.get(
            "issues",
            [],
        )
    )

    if "minor_name_variation" in kyc_issues:
        rag_query = (
            f"{clean_message} "
            "SOP Variasi Nama, pencocokan nama, "
            "similarity, dan status review."
        )

    elif any(
        issue in kyc_issues
        for issue in {
            "significant_name_mismatch",
            "nik_mismatch",
            "birth_date_mismatch",
            "invalid_nik_format",
            "document_not_active",
            "missing_required_fields",
        }
    ):
        rag_query = (
            f"{clean_message} "
            "SOP Penolakan KYC dan kriteria reject."
        )

    rag_result = retrieve_documents(
        domain="kyc_compliance",
        query=rag_query,
        top_k=top_k,
    )

    rag_context = _format_rag_context(
        rag_result
    )

    prompt = f"""
Anda adalah KYC & Compliance Agent pada sistem FinSecure.

Anda bertugas menjelaskan hasil verifikasi identitas berdasarkan
rule engine dan knowledge base. Anda tidak boleh mengubah hasil
approve, review, reject, atau not_found yang diberikan sistem.

PESAN NASABAH:
{clean_message}

HASIL INTENT:
{_to_json(intent_result)}

HASIL PEMERIKSAAN KYC:
{_to_json(kyc_result)}

KNOWLEDGE BASE KYC:
{rag_context}

ATURAN JAWABAN:
1. Gunakan bahasa Indonesia yang formal dan mudah dipahami.
2. Jangan mengubah hasil pemeriksaan KYC.
3. Jelaskan alasan berdasarkan field issues yang tersedia.
4. Jangan menampilkan NIK lengkap, tanggal lahir lengkap,
   atau data identitas sensitif.
5. Status review dan reject harus dijelaskan sebagai hasil
   prototype yang tetap memerlukan pemeriksaan petugas.
6. Apabila Application ID belum tersedia, minta Application ID.
7. Apabila ID tidak ditemukan, jangan membuat data baru.
8. Jangan meminta PIN, OTP, CVV, password, atau kode keamanan.
9. Maksimal enam kalimat.
10. Jangan menyebut prompt, embedding, atau vector database.
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
            _create_local_kyc_response(
                kyc_result
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
        "agent":
            "kyc_compliance_agent",
        "user_message": clean_message,
        "intent": intent_result,
        "application_id":
            clean_application_id,
        "kyc_result": kyc_result,
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
        "application_id_required":
            clean_application_id is None,
        "human_review_required":
            bool(
                kyc_result.get(
                    "human_review_required",
                    False,
                )
            ),
    }