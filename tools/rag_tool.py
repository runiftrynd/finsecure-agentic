from __future__ import annotations

from typing import Any

from model_loader import (
    EMBEDDING_MODEL_NAME,
    load_embedding_model,
    load_vector_collection,
)


VALID_DOMAINS = {
    "customer_service",
    "fraud_risk",
    "kyc_compliance",
}

QUERY_EXPANSION_RULES = {
    "customer_service": [
        (
            (
                "pin",
                "ganti pin",
                "ubah pin",
            ),
            (
                "SOP Mengganti PIN, perubahan PIN, "
                "verifikasi keamanan"
            ),
        ),
        (
            (
                "transfer",
                "pending",
                "tertunda",
                "gagal",
            ),
            (
                "SOP Transfer Gagal atau Tertunda, "
                "status transfer pending"
            ),
        ),
        (
            (
                "kartu baru",
                "kartu sampai",
                "kartu datang",
                "pengiriman kartu",
                "status kartu",
                "kartu belum diterima",
            ),
            (
                "SOP Pengiriman Kartu Baru, "
                "status pengiriman kartu, "
                "estimasi kedatangan kartu"
            ),
        ),
        (
            (
                "kartu hilang",
            ),
            (
                "SOP Pengiriman Kartu Baru, "
                "status pengiriman kartu, "
                "estimasi kedatangan kartu"
            ),
        ),
        (
            (
                "kartu hilang",
                "kehilangan kartu",
            ),
            (
                "SOP Kartu Hilang dan "
                "pemblokiran sementara"
            ),
        ),
        (
            (
                "saldo",
                "saldo belum masuk",
                "saldo tidak berubah",
            ),
            (
                "SOP Saldo Belum Diperbarui"
            ),
        ),
    ],

    "fraud_risk": [
        (
            (
                "tidak saya kenali",
                "tidak mengenali",
                "tidak dikenali",
                "tidak dikenal",
                "transaksi asing",
                "mencurigakan",
            ),
            (
                "SOP Transaksi Tidak Dikenali "
                "dan fraud scoring"
            ),
        ),
        (
            (
                "malam",
                "terminal",
                "risiko",
                "fraud",
            ),
            (
                "Penjelasan Risiko Transaksi "
                "dan faktor risiko model"
            ),
        ),
        (
            (
                "review",
                "pemeriksaan manual",
            ),
            (
                "Prosedur Human Review Fraud"
            ),
        ),
    ],

    "kyc_compliance": [
        (
            (
                "nama",
                "salah eja",
                "typo",
                "berbeda sedikit",
                "sedikit berbeda",
                "variasi nama",
            ),
            (
                "SOP Variasi Nama, pencocokan nama, "
                "similarity, dan status review"
            ),
        ),
        (
            (
                "nik",
                "tanggal lahir",
                "dokumen tidak aktif",
                "expired",
                "ditolak",
                "gagal",
                "verifikasi gagal",
                "kyc gagal",
            ),
            (
                "SOP Penolakan KYC dan "
                "kriteria reject"
            ),
        ),
        (
            (
                "ubah identitas",
                "perubahan identitas",
                "perbarui identitas",
            ),
            (
                "SOP Perubahan Identitas"
            ),
        ),
    ],
}

def _validate_domain(
    domain: str,
) -> str:
    """
    Membersihkan dan memvalidasi nama domain.
    """

    clean_domain = str(
        domain
    ).strip().lower()

    if clean_domain not in VALID_DOMAINS:
        raise ValueError(
            "Domain RAG tidak valid: "
            f"{clean_domain}. "
            "Gunakan customer_service, "
            "fraud_risk, atau kyc_compliance."
        )

    return clean_domain

def _expand_query(
    domain: str,
    query: str,
) -> str:
    """
    Menambahkan petunjuk domain agar dokumen
    yang lebih spesifik memperoleh prioritas.
    """

    lower_query = query.lower()

    domain_rules = QUERY_EXPANSION_RULES.get(
        domain,
        [],
    )

    for keywords, expansion in domain_rules:
        keyword_found = any(
            keyword in lower_query
            for keyword in keywords
        )

        if keyword_found:
            return (
                f"{query}\n"
                f"Fokus pencarian: {expansion}"
            )

    return query

def _distance_to_score(
    distance: float | None,
) -> float | None:
    """
    Mengubah distance menjadi skor sederhana
    pada rentang nol sampai satu.

    Skor hanya digunakan untuk tampilan,
    bukan untuk keputusan model.
    """

    if distance is None:
        return None

    clean_distance = max(
        float(distance),
        0.0,
    )

    return float(
        1.0
        / (1.0 + clean_distance)
    )


def retrieve_documents(
    domain: str,
    query: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Mengambil dokumen paling relevan dari
    vectorstore berdasarkan domain.

    Embedding query dibuat menggunakan model
    yang sama dengan vectorstore Colab.
    """

    clean_domain = _validate_domain(
        domain
    )

    clean_query = str(
        query
    ).strip()

    if not clean_query:
        raise ValueError(
            "Query RAG tidak boleh kosong."
        )

    try:
        clean_top_k = int(
            top_k
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "top_k harus berupa angka."
        ) from error

    if clean_top_k < 1:
        raise ValueError(
            "top_k minimal bernilai 1."
        )

    embedding_model = (
        load_embedding_model()
    )

    collection = (
        load_vector_collection(
            clean_domain
        )
    )

    document_count = collection.count()

    result_limit = min(
        clean_top_k,
        document_count,
    )

    retrieval_query = _expand_query(
        domain=clean_domain,
        query=clean_query,
    )

    query_embedding = (
        embedding_model.encode(
            retrieval_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )

    query_result = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=result_limit,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = (
        query_result.get(
            "ids",
            [[]],
        )[0]
        or []
    )

    documents = (
        query_result.get(
            "documents",
            [[]],
        )[0]
        or []
    )

    metadatas = (
        query_result.get(
            "metadatas",
            [[]],
        )[0]
        or []
    )

    distances = (
        query_result.get(
            "distances",
            [[]],
        )[0]
        or []
    )

    retrieved_documents = []

    for index, document in enumerate(
        documents
    ):
        metadata = (
            metadatas[index]
            if index < len(metadatas)
            and metadatas[index]
            else {}
        )

        distance = (
            float(distances[index])
            if index < len(distances)
            and distances[index] is not None
            else None
        )

        document_id = (
            str(ids[index])
            if index < len(ids)
            else None
        )

        retrieved_documents.append(
            {
                "rank": index + 1,
                "document_id":
                    document_id,
                "domain":
                    clean_domain,
                "content":
                    str(document).strip(),
                "source":
                    metadata.get(
                        "source"
                    ),
                "section":
                    metadata.get(
                        "section"
                    ),
                "version":
                    metadata.get(
                        "version"
                    ),
                "distance":
                    distance,
                "relevance_score":
                    _distance_to_score(
                        distance
                    ),
                "metadata":
                    metadata,
            }
        )

    return {
        "status": "success",
        "domain": clean_domain,
        "query": clean_query,
        "retrieval_query": retrieval_query,
        "collection_name":
            str(collection.name),
        "embedding_model":
            EMBEDDING_MODEL_NAME,
        "available_documents":
            document_count,
        "result_count":
            len(retrieved_documents),
        "documents":
            retrieved_documents,
    }


def retrieve_context_text(
    domain: str,
    query: str,
    top_k: int = 3,
) -> str:
    """
    Mengubah hasil retrieval menjadi teks
    yang dapat dimasukkan ke prompt Gemini.
    """

    result = retrieve_documents(
        domain=domain,
        query=query,
        top_k=top_k,
    )

    context_parts = []

    for document in result["documents"]:
        source = (
            document.get("source")
            or "Sumber tidak diketahui"
        )

        section = (
            document.get("section")
            or "Bagian tidak diketahui"
        )

        content = document.get(
            "content",
            "",
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
            "Tidak ada dokumen knowledge base "
            "yang ditemukan."
        )

    return "\n\n---\n\n".join(
        context_parts
    )
