from __future__ import annotations

from typing import Any

import streamlit as st

from backend import process_finsecure_request

st.set_page_config(
    page_title="FinSecure Agentic",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.code(
    f"Watcher: {st.get_option('server.fileWatcherType')}"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .finsecure-header {
        padding: 1.3rem 1.5rem;
        border: 1px solid #dbe4f0;
        border-radius: 16px;
        background:
            linear-gradient(
                135deg,
                #f8fbff 0%,
                #eef5ff 100%
            );
        margin-bottom: 1.2rem;
    }

    .finsecure-title {
        font-size: 2rem;
        font-weight: 750;
        margin: 0;
        color: #152238;
    }

    .finsecure-subtitle {
        margin-top: 0.35rem;
        margin-bottom: 0;
        color: #52647a;
        line-height: 1.5;
    }

    .status-box {
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-top: 0.6rem;
        background: #f8fafc;
    }

    .review-box {
        border-left: 4px solid #d97706;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        background: #fff8e6;
        margin-top: 0.75rem;
    }

    .fallback-box {
        border-left: 4px solid #64748b;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        background: #f1f5f9;
        margin-top: 0.75rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


AGENT_DISPLAY_NAMES = {
    "customer_service_agent":
        "Customer Service",
    "fraud_risk_agent":
        "Fraud & Risk",
    "kyc_compliance_agent":
        "KYC & Compliance",
}


def initialize_session_state() -> None:
    """
    Menyiapkan riwayat percakapan.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Selamat datang di FinSecure. "
                    "Sampaikan pertanyaan layanan, "
                    "keluhan transaksi, atau kendala "
                    "verifikasi identitas Anda."
                ),
                "metadata": None,
            }
        ]


def format_confidence(
    value: Any,
) -> str:
    """
    Mengubah confidence menjadi persentase.
    """

    if not isinstance(
        value,
        (int, float),
    ):
        return "-"

    return f"{float(value) * 100:.2f}%"


def display_metadata(
    metadata: dict[str, Any],
    show_debug: bool,
) -> None:
    """
    Menampilkan metadata workflow tanpa
    mengekspos error teknis kepada pengguna.
    """

    routing = metadata.get(
        "routing",
        {},
    )

    intent = metadata.get(
        "intent",
        {},
    )

    identifiers = metadata.get(
        "identifiers",
        {},
    )

    generation = metadata.get(
        "generation",
        {},
    )

    executed_agents = routing.get(
        "executed_agents",
        [],
    )

    agent_labels = [
        AGENT_DISPLAY_NAMES.get(
            agent,
            agent,
        )
        for agent in executed_agents
    ]

    caption_parts = [
        (
            "Intent: "
            f"{intent.get('resolved_intent') or '-'}"
        ),
        (
            "Routing: "
            f"{routing.get('mode') or '-'}"
        ),
        (
            "Agent: "
            f"{', '.join(agent_labels) or '-'}"
        ),
    ]

    st.caption(
        " · ".join(
            caption_parts
        )
    )

    if metadata.get(
        "human_review_required",
        False,
    ):
        st.markdown(
            """
            <div class="review-box">
                <strong>Pemeriksaan manual diperlukan.</strong><br>
                Hasil sistem perlu ditinjau oleh petugas sebelum
                keputusan operasional dibuat.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if generation.get(
        "fallback_used",
        False,
    ):
        st.markdown(
            """
            <div class="fallback-box">
                Jawaban disusun menggunakan hasil model lokal
                karena layanan penyusunan bahasa eksternal
                sedang tidak tersedia.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander(
        "Detail hasil pemeriksaan",
        expanded=False,
    ):
        column_1, column_2 = st.columns(2)

        with column_1:
            st.markdown(
                "**Klasifikasi intent**"
            )

            st.write(
                {
                    "Intent akhir":
                        intent.get(
                            "resolved_intent"
                        ),
                    "Confidence":
                        format_confidence(
                            intent.get(
                                "confidence"
                            )
                        ),
                    "Metode":
                        intent.get(
                            "resolution_method"
                        ),
                    "Keyword override":
                        intent.get(
                            "override_applied"
                        ),
                }
            )

        with column_2:
            st.markdown(
                "**Routing agent**"
            )

            st.write(
                {
                    "Mode":
                        routing.get(
                            "mode"
                        ),
                    "Jumlah agent":
                        routing.get(
                            "agent_count"
                        ),
                    "Transaction ID":
                        identifiers.get(
                            "transaction_id"
                        ),
                    "Application ID":
                        identifiers.get(
                            "application_id"
                        ),
                }
            )

        worker_details = metadata.get(
            "worker_details",
            {},
        )

        fraud_detail = worker_details.get(
            "fraud_risk"
        )

        if fraud_detail:
            st.markdown(
                "**Hasil Fraud & Risk**"
            )

            risk_summary = fraud_detail.get(
                "risk_summary",
                {},
            )

            st.write(
                {
                    "Transaction ID":
                        fraud_detail.get(
                            "transaction_id"
                        ),
                    "Status transaksi":
                        fraud_detail.get(
                            "transaction_status"
                        ),
                    "Probabilitas risiko":
                        risk_summary.get(
                            "fraud_probability_percent"
                        ),
                    "Tingkat risiko":
                        risk_summary.get(
                            "risk_level"
                        ),
                    "Rekomendasi":
                        risk_summary.get(
                            "recommendation"
                        ),
                    "Human review":
                        fraud_detail.get(
                            "human_review_required"
                        ),
                }
            )

            risk_factors = risk_summary.get(
                "risk_factors",
                [],
            )

            if risk_factors:
                st.markdown(
                    "**Faktor risiko utama**"
                )

                st.dataframe(
                    risk_factors,
                    use_container_width=True,
                    hide_index=True,
                )

        kyc_detail = worker_details.get(
            "kyc_compliance"
        )

        if kyc_detail:
            st.markdown(
                "**Hasil KYC & Compliance**"
            )

            kyc_result = kyc_detail.get(
                "kyc_result",
                {},
            )

            st.write(
                {
                    "Application ID":
                        kyc_detail.get(
                            "application_id"
                        ),
                    "Status":
                        kyc_result.get(
                            "status"
                        ),
                    "Kemiripan nama":
                        kyc_result.get(
                            "name_similarity"
                        ),
                    "NIK sesuai":
                        kyc_result.get(
                            "nik_match"
                        ),
                    "Tanggal lahir sesuai":
                        kyc_result.get(
                            "birth_date_match"
                        ),
                    "Status dokumen":
                        kyc_result.get(
                            "document_status"
                        ),
                    "Masalah":
                        kyc_result.get(
                            "issues"
                        ),
                    "Human review":
                        kyc_detail.get(
                            "human_review_required"
                        ),
                }
            )

        if show_debug:
            st.divider()
            st.markdown(
                "**Debug workflow**"
            )

            st.json(
                metadata.get(
                    "debug",
                    {},
                )
            )


initialize_session_state()


with st.sidebar:
    st.header(
        "Pengaturan pemeriksaan"
    )

    st.write(
        "Identifier bersifat opsional. "
        "Sistem juga dapat mendeteksinya "
        "langsung dari pesan."
    )

    manual_transaction_id = st.text_input(
        "Transaction ID",
        placeholder="Contoh: TRX0002",
    )

    manual_application_id = st.text_input(
        "Application ID",
        placeholder="Contoh: KYC0153",
    )

    show_debug = st.toggle(
        "Tampilkan debug",
        value=False,
        help=(
            "Menampilkan routing, prediksi intent, "
            "dan data teknis untuk pengujian."
        ),
    )

    st.divider()

    st.markdown(
        """
        **Contoh pertanyaan**

        - Bagaimana cara mengganti PIN?
        - Saya tidak mengenali transaksi TRX0002.
        - Mengapa verifikasi KYC0153 ditinjau?
        - KYC0153 bermasalah dan transfer TRX0002 pending.
        """
    )

    if st.button(
        "Hapus percakapan",
        use_container_width=True,
    ):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Percakapan telah dihapus. "
                    "Silakan sampaikan pertanyaan baru."
                ),
                "metadata": None,
            }
        ]

        st.rerun()


st.markdown(
    """
    <div class="finsecure-header">
        <p class="finsecure-title">
            FinSecure Agentic
        </p>
        <p class="finsecure-subtitle">
            Sistem layanan fintech berbasis multi-agent untuk
            layanan nasabah, analisis risiko transaksi, serta
            pemeriksaan KYC dan kepatuhan.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("metadata")
        ):
            display_metadata(
                metadata=message[
                    "metadata"
                ],
                show_debug=show_debug,
            )


user_message = st.chat_input(
    "Tuliskan pertanyaan atau keluhan Anda"
)


if user_message:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
            "metadata": None,
        }
    )

    with st.chat_message("user"):
        st.markdown(
            user_message
        )

    with st.chat_message("assistant"):
        with st.spinner(
            "FinSecure sedang memeriksa permintaan..."
        ):
            result = process_finsecure_request(
                user_message=user_message,
                transaction_id=(
                    manual_transaction_id
                    or None
                ),
                application_id=(
                    manual_application_id
                    or None
                ),
                include_debug=show_debug,
            )

        assistant_response = str(
            result.get(
                "response",
                (
                    "Permintaan belum dapat "
                    "diproses."
                ),
            )
        )

        st.markdown(
            assistant_response
        )

        if result.get("status") == "failure":
            st.error(
                "Terjadi kendala ketika memproses permintaan."
            )   

        if show_debug:
            debug_result = result.get(
                "debug",
                {},
            )

            st.markdown(
                "**Detail error pengujian**"
            )

            st.code(
                debug_result.get(
                    "traceback",
                        debug_result.get(
                            "error",
                            "Detail error tidak tersedia.",
                        ),
                    ),
                language="text",
            )

        display_metadata(
            metadata=result,
            show_debug=show_debug,
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content":
                assistant_response,
            "metadata":
                result,
        }
    )
