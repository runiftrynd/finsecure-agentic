from __future__ import annotations

import json
import re
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
from llm_client import generate_text
from tools.intent_tool import resolve_intent


CUSTOMER_SERVICE_AGENT = (
    "customer_service_agent"
)

FRAUD_RISK_AGENT = (
    "fraud_risk_agent"
)

KYC_COMPLIANCE_AGENT = (
    "kyc_compliance_agent"
)


CUSTOMER_SERVICE_INTENTS = {
    "account_blocked",
    "balance_not_updated",
    "card_arrival",
    "change_pin",
    "lost_card",
    "transfer_failed",
    "transfer_pending",
}

FRAUD_INTENTS = {
    "suspicious_transaction",
}

KYC_INTENTS = {
    "kyc_failed",
    "update_identity",
}


CUSTOMER_SERVICE_KEYWORDS = (
    "transfer pending",
    "transfer tertunda",
    "transfer gagal",
    "saldo",
    "ganti pin",
    "ubah pin",
    "kartu hilang",
    "kartu belum tiba",
    "akun diblokir",
    "akun terkunci",
)

FRAUD_KEYWORDS = (
    "tidak saya kenali",
    "tidak mengenali",
    "tidak dikenali",
    "tidak dikenal",
    "transaksi asing",
    "transaksi mencurigakan",
    "fraud",
    "risiko transaksi",
    "transaksi malam",
)

KYC_KEYWORDS = (
    "kyc",
    "verifikasi identitas",
    "identitas gagal",
    "identitas ditolak",
    "perubahan identitas",
    "update identitas",
    "perbarui identitas",
    "dokumen identitas",
    "nik",
    "nama berbeda",
    "variasi nama",
)


TRANSACTION_ID_PATTERN = re.compile(
    r"\bTRX[A-Z0-9-]+\b",
    flags=re.IGNORECASE,
)

APPLICATION_ID_PATTERN = re.compile(
    r"\bKYC[A-Z0-9-]+\b",
    flags=re.IGNORECASE,
)


def _to_json(
    data: Any,
) -> str:
    """
    Mengubah data ke JSON untuk prompt Gemini.
    """

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def _clean_optional_id(
    value: str | None,
) -> str | None:
    """
    Membersihkan identifier opsional.
    """

    if value is None:
        return None

    clean_value = str(
        value
    ).strip().upper()

    return clean_value or None


def extract_transaction_id(
    user_message: str,
) -> str | None:
    """
    Mendeteksi Transaction ID dari pesan.
    """

    match = TRANSACTION_ID_PATTERN.search(
        str(user_message)
    )

    if match is None:
        return None

    return match.group(0).upper()


def extract_application_id(
    user_message: str,
) -> str | None:
    """
    Mendeteksi Application ID dari pesan.
    """

    match = APPLICATION_ID_PATTERN.search(
        str(user_message)
    )

    if match is None:
        return None

    return match.group(0).upper()


def _contains_keyword(
    message: str,
    keywords: tuple[str, ...],
) -> bool:
    """
    Memeriksa sinyal domain pada pesan.
    """

    lower_message = message.lower()

    return any(
        keyword in lower_message
        for keyword in keywords
    )


def _get_primary_agent(
    intent: str,
) -> str:
    """
    Menentukan agent utama dari resolved intent.
    """

    if intent in FRAUD_INTENTS:
        return FRAUD_RISK_AGENT

    if intent in KYC_INTENTS:
        return KYC_COMPLIANCE_AGENT

    return CUSTOMER_SERVICE_AGENT


def plan_agent_route(
    user_message: str,
    intent_result: dict[str, Any],
    transaction_id: str | None = None,
    application_id: str | None = None,
) -> dict[str, Any]:
    """
    Membuat rencana routing satu sampai tiga agent.
    """

    clean_message = str(
        user_message
    ).strip()

    resolved_intent = str(
        intent_result.get(
            "intent",
            "unknown",
        )
    )

    primary_agent = _get_primary_agent(
        resolved_intent
    )

    customer_service_signal = (
        _contains_keyword(
            clean_message,
            CUSTOMER_SERVICE_KEYWORDS,
        )
        or resolved_intent
        in CUSTOMER_SERVICE_INTENTS
    )

    fraud_signal = (
        _contains_keyword(
            clean_message,
            FRAUD_KEYWORDS,
        )
        or resolved_intent in FRAUD_INTENTS
    )

    kyc_signal = (
        _contains_keyword(
            clean_message,
            KYC_KEYWORDS,
        )
        or resolved_intent in KYC_INTENTS
    )

    # Identifier memperkuat sinyal domain,
    # tetapi tidak otomatis memanggil agent
    # tanpa konteks yang relevan.
    if (
        transaction_id
        and (
            "transaksi" in clean_message.lower()
            or "transfer" in clean_message.lower()
        )
    ):
        fraud_signal = (
            fraud_signal
            or _contains_keyword(
                clean_message,
                FRAUD_KEYWORDS,
            )
        )

    if (
        application_id
        and (
            "identitas" in clean_message.lower()
            or "kyc" in clean_message.lower()
            or "nama" in clean_message.lower()
        )
    ):
        kyc_signal = True

    selected_agents: set[str] = {
        primary_agent
    }

    routing_reasons: list[str] = [
        (
            "Agent utama ditentukan dari "
            f"resolved intent: {resolved_intent}."
        )
    ]

    if customer_service_signal:
        selected_agents.add(
            CUSTOMER_SERVICE_AGENT
        )

        routing_reasons.append(
            "Pesan memiliki konteks "
            "layanan nasabah."
        )

    if fraud_signal:
        selected_agents.add(
            FRAUD_RISK_AGENT
        )

        routing_reasons.append(
            "Pesan memiliki indikasi "
            "pemeriksaan risiko transaksi."
        )

    if kyc_signal:
        selected_agents.add(
            KYC_COMPLIANCE_AGENT
        )

        routing_reasons.append(
            "Pesan memiliki konteks "
            "verifikasi identitas."
        )

    execution_order = [
        CUSTOMER_SERVICE_AGENT,
        FRAUD_RISK_AGENT,
        KYC_COMPLIANCE_AGENT,
    ]

    planned_agents = [
        agent
        for agent in execution_order
        if agent in selected_agents
    ]

    agent_count = len(
        planned_agents
    )

    if agent_count == 1:
        routing_mode = "single_agent"

    elif agent_count == 2:
        routing_mode = "multi_agent_2"

    else:
        routing_mode = "multi_agent_3"

    return {
        "primary_agent":
            primary_agent,
        "planned_agents":
            planned_agents,
        "agent_count":
            agent_count,
        "routing_mode":
            routing_mode,
        "signals": {
            "customer_service":
                customer_service_signal,
            "fraud_risk":
                fraud_signal,
            "kyc_compliance":
                kyc_signal,
        },
        "reasons":
            routing_reasons,
    }


def _execute_worker_agent(
    agent_name: str,
    user_message: str,
    intent_result: dict[str, Any],
    transaction_id: str | None,
    application_id: str | None,
) -> dict[str, Any]:
    """
    Menjalankan Worker Agent berdasarkan nama.
    """

    if agent_name == CUSTOMER_SERVICE_AGENT:
        return run_customer_service_agent(
            user_message=user_message,
            intent_result=intent_result,
            top_k=2,
            use_llm=False,
        )

    if agent_name == FRAUD_RISK_AGENT:
        return run_fraud_risk_agent(
            user_message=user_message,
            transaction_id=transaction_id,
            intent_result=intent_result,
            top_k=2,
            use_llm=False,
        )

    if agent_name == KYC_COMPLIANCE_AGENT:
        return run_kyc_compliance_agent(
            user_message=user_message,
            application_id=application_id,
            intent_result=intent_result,
            top_k=2,
            use_llm=False,
        )

    raise ValueError(
        f"Worker Agent tidak dikenal: {agent_name}"
    )


def _compact_worker_result(
    worker_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Mengambil bagian penting Worker Agent
    untuk proses sintesis Supervisor.
    """

    agent_name = worker_result.get(
        "agent"
    )

    compact_result: dict[str, Any] = {
        "agent": agent_name,
        "status":
            worker_result.get("status"),
        "response":
            worker_result.get("response"),
        "human_review_required":
            worker_result.get(
                "human_review_required",
                False,
            ),
    }

    if agent_name == CUSTOMER_SERVICE_AGENT:
        compact_result.update(
            {
                "handoff_required":
                    worker_result.get(
                        "handoff_required"
                    ),
                "handoff_target":
                    worker_result.get(
                        "handoff_target"
                    ),
            }
        )

    elif agent_name == FRAUD_RISK_AGENT:
        transaction_result = (
            worker_result.get(
                "transaction_result",
                {},
            )
        )

        compact_result.update(
            {
                "transaction_id":
                    worker_result.get(
                        "transaction_id"
                    ),
                "transaction_status":
                    transaction_result.get(
                        "transaction",
                        {},
                    ).get(
                        "transaction_status"
                    ),
                "transaction_lookup_status":
                    transaction_result.get(
                        "status"
                    ),
                "risk_summary":
                    worker_result.get(
                        "risk_summary"
                    ),
                "transaction_id_required":
                    worker_result.get(
                        "transaction_id_required"
                    ),
            }
        )

    elif agent_name == KYC_COMPLIANCE_AGENT:
        compact_result.update(
            {
                "application_id":
                    worker_result.get(
                        "application_id"
                    ),
                "kyc_result":
                    worker_result.get(
                        "kyc_result"
                    ),
                "application_id_required":
                    worker_result.get(
                        "application_id_required"
                    ),
            }
        )

    return compact_result


def _create_deterministic_response(
    worker_results: list[dict[str, Any]],
) -> str:
    """
    Membuat respons cadangan tanpa Gemini.
    """

    valid_responses = [
        str(result.get("response")).strip()
        for result in worker_results
        if result.get("response")
    ]

    if not valid_responses:
        return (
            "Permintaan telah diproses, tetapi "
            "sistem belum menghasilkan jawaban."
        )

    if len(valid_responses) == 1:
        return valid_responses[0]

    return "\n\n".join(
        valid_responses
    )


def run_supervisor_agent(
    user_message: str,
    transaction_id: str | None = None,
    application_id: str | None = None,
) -> dict[str, Any]:
    """
    Menjalankan Supervisor Agent FinSecure.
    """

    clean_message = str(
        user_message
    ).strip()

    if not clean_message:
        raise ValueError(
            "Pesan pengguna tidak boleh kosong."
        )

    resolved_transaction_id = (
        _clean_optional_id(
            transaction_id
        )
        or extract_transaction_id(
            clean_message
        )
    )

    resolved_application_id = (
        _clean_optional_id(
            application_id
        )
        or extract_application_id(
            clean_message
        )
    )

    intent_result = resolve_intent(
        clean_message
    )

    route_plan = plan_agent_route(
        user_message=clean_message,
        intent_result=intent_result,
        transaction_id=
            resolved_transaction_id,
        application_id=
            resolved_application_id,
    )

    worker_results: list[
        dict[str, Any]
    ] = []

    execution_errors: list[
        dict[str, str]
    ] = []

    for agent_name in route_plan[
        "planned_agents"
    ]:
        try:
            worker_result = (
                _execute_worker_agent(
                    agent_name=agent_name,
                    user_message=clean_message,
                    intent_result=intent_result,
                    transaction_id=
                        resolved_transaction_id,
                    application_id=
                        resolved_application_id,
                )
            )

            worker_results.append(
                worker_result
            )

        except Exception as error:
            execution_errors.append(
                {
                    "agent": agent_name,
                    "error":
                        str(error),
                    "error_type":
                        type(error).__name__,
                }
            )

    compact_results = [
        _compact_worker_result(
            result
        )
        for result in worker_results
    ]

    human_review_required = any(
        bool(
            result.get(
                "human_review_required",
                False,
            )
        )
        for result in worker_results
    )

    synthesis_prompt = f"""
Anda adalah Supervisor Agent pada sistem FinSecure.

Tugas Anda adalah menggabungkan hasil Worker Agent menjadi satu
jawaban yang koheren untuk nasabah.

PESAN NASABAH:
{clean_message}

HASIL ROUTING:
{_to_json(route_plan)}

HASIL WORKER AGENT:
{_to_json(compact_results)}

ATURAN:
1. Gunakan bahasa Indonesia yang formal dan mudah dipahami.
2. Jangan mengubah status transaksi, hasil KYC, probabilitas,
   tingkat risiko, rekomendasi, atau hasil model.
3. Status operasional transaksi dan fraud score adalah dua
   informasi yang berbeda.
4. Jangan menyatakan fraud score menyebabkan transaksi pending,
   gagal, completed, atau dibatalkan.
5. Jangan menuduh nasabah atau pihak lain melakukan kejahatan.
6. Jelaskan kebutuhan Transaction ID atau Application ID jika
   identifier tersebut belum tersedia.
7. Jika human review diperlukan, sampaikan dengan jelas.
8. Jangan menampilkan NIK lengkap atau data sensitif.
9. Jangan meminta PIN, OTP, CVV, password, atau kode keamanan.
10. Jangan menyebut nama internal agent, prompt, embedding,
    vectorstore, atau arsitektur sistem.
11. Jangan membuat informasi yang tidak tersedia pada hasil
    Worker Agent.
12. Maksimal delapan kalimat.
""".strip()

    if worker_results:
        synthesis_result = generate_text(
            prompt=synthesis_prompt,
            max_retries=2,
        )

        if synthesis_result[
            "fallback_used"
        ]:
            final_response = (
                _create_deterministic_response(
                    worker_results
                )
            )
        else:
            final_response = (
                synthesis_result["response"]
            )

    else:
        synthesis_result = {
            "status": "fallback",
            "provider":
                "local_fallback",
            "model": None,
            "fallback_used": True,
            "error":
                "Tidak ada Worker Agent "
                "yang berhasil dijalankan.",
        }

        final_response = (
            "Permintaan belum dapat diproses "
            "karena Worker Agent gagal dijalankan."
        )

    overall_status = (
        "success"
        if worker_results
        and not execution_errors
        else "partial_failure"
    )

    return {
        "status":
            overall_status,
        "agent":
            "supervisor_agent",
        "user_message":
            clean_message,
        "transaction_id":
            resolved_transaction_id,
        "application_id":
            resolved_application_id,
        "intent":
            intent_result,
        "route":
            route_plan,
        "executed_agents": [
            result.get("agent")
            for result in worker_results
        ],
        "worker_results":
            worker_results,
        "execution_errors":
            execution_errors,
        "final_response":
            final_response,
        "synthesis_llm": {
            "status":
                synthesis_result.get(
                    "status"
                ),
            "provider":
                synthesis_result.get(
                    "provider"
                ),
            "model":
                synthesis_result.get(
                    "model"
                ),
            "fallback_used":
                synthesis_result.get(
                    "fallback_used"
                ),
            "error":
                synthesis_result.get(
                    "error"
                ),
        },
        "human_review_required":
            human_review_required,
    }