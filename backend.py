from __future__ import annotations

from typing import Any

from workflow import run_finsecure_workflow

import traceback

def _classify_llm_error(
    synthesis_llm: dict[str, Any],
) -> str | None:
    """
    Mengubah error teknis LLM menjadi kategori aman.

    Detail error asli tidak dikirim ke antarmuka pengguna.
    """

    raw_error = synthesis_llm.get(
        "error"
    )

    if not raw_error:
        return None

    error_text = str(
        raw_error
    ).lower()

    if (
        "429" in error_text
        or "quota" in error_text
        or "too_many_requests" in error_text
    ):
        return "quota_exceeded"

    if (
        "timeout" in error_text
        or "timed out" in error_text
    ):
        return "timeout"

    if (
        "api key" in error_text
        or "authentication" in error_text
        or "unauthorized" in error_text
    ):
        return "authentication_error"

    return "llm_unavailable"


def _extract_worker_details(
    worker_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Mengambil hasil penting dari masing-masing
    Worker Agent untuk ditampilkan pada dashboard.
    """

    details: dict[str, Any] = {
        "customer_service": None,
        "fraud_risk": None,
        "kyc_compliance": None,
    }

    for worker in worker_results:
        agent_name = worker.get(
            "agent"
        )

        if agent_name == "customer_service_agent":
            details["customer_service"] = {
                "response":
                    worker.get(
                        "response"
                    ),
                "handoff_required":
                    worker.get(
                        "handoff_required",
                        False,
                    ),
                "handoff_target":
                    worker.get(
                        "handoff_target"
                    ),
                "rag_sources":
                    worker.get(
                        "rag",
                        {},
                    ).get(
                        "sources",
                        [],
                    ),
            }

        elif agent_name == "fraud_risk_agent":
            transaction_result = worker.get(
                "transaction_result",
                {},
            )

            details["fraud_risk"] = {
                "transaction_id":
                    worker.get(
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
                "transaction_id_required":
                    worker.get(
                        "transaction_id_required",
                        False,
                    ),
                "risk_summary":
                    worker.get(
                        "risk_summary",
                        {},
                    ),
                "human_review_required":
                    worker.get(
                        "human_review_required",
                        False,
                    ),
                "rag_sources":
                    worker.get(
                        "rag",
                        {},
                    ).get(
                        "sources",
                        [],
                    ),
            }

        elif agent_name == "kyc_compliance_agent":
            details["kyc_compliance"] = {
                "application_id":
                    worker.get(
                        "application_id"
                    ),
                "application_id_required":
                    worker.get(
                        "application_id_required",
                        False,
                    ),
                "kyc_result":
                    worker.get(
                        "kyc_result",
                        {},
                    ),
                "human_review_required":
                    worker.get(
                        "human_review_required",
                        False,
                    ),
                "rag_sources":
                    worker.get(
                        "rag",
                        {},
                    ).get(
                        "sources",
                        [],
                    ),
            }

    return details


def process_finsecure_request(
    user_message: str,
    transaction_id: str | None = None,
    application_id: str | None = None,
    include_debug: bool = False,
) -> dict[str, Any]:
    """
    Menjalankan workflow FinSecure dan
    membentuk respons aman untuk frontend.
    """

    clean_message = str(
        user_message
    ).strip()

    if not clean_message:
        return {
            "status": "invalid_input",
            "response": (
                "Masukkan pertanyaan atau "
                "keluhan terlebih dahulu."
            ),
            "human_review_required": False,
        }

    try:
        workflow_result = (
            run_finsecure_workflow(
                user_message=clean_message,
                transaction_id=transaction_id,
                application_id=application_id,
            )
        )

    except Exception as error:
        error_traceback = traceback.format_exc()

        # Ditampilkan pada Streamlit Cloud Logs.
        print(
            "\n===== FINSECURE BACKEND ERROR =====",
            flush=True,
        )
        print(
            error_traceback,
            flush=True,
        )
        print(
            "===== END FINSECURE ERROR =====\n",
            flush=True,
        )

        result = {
            "status": "failure",
            "response": (
                "Permintaan belum dapat diproses. "
                "Silakan periksa kembali data yang "
                "dimasukkan dan coba lagi."
            ),
            "human_review_required": False,
        }

        if include_debug:
            result["debug"] = {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": error_traceback,
            }

        return result

    route = workflow_result.get(
        "route",
        {},
    )

    intent = workflow_result.get(
        "intent",
        {},
    )

    synthesis_llm = workflow_result.get(
        "synthesis_llm",
        {},
    )

    worker_results = workflow_result.get(
        "worker_results",
        [],
    )

    llm_error_category = (
        _classify_llm_error(
            synthesis_llm
        )
    )

    public_result: dict[str, Any] = {
        "status":
            workflow_result.get(
                "status",
                "failure",
            ),
        "response":
            workflow_result.get(
                "final_response"
            )
            or (
                "Permintaan telah diproses, "
                "tetapi jawaban belum tersedia."
            ),
        "intent": {
            "resolved_intent":
                intent.get(
                    "intent"
                ),
            "model_intent":
                intent.get(
                    "model_intent"
                ),
            "confidence":
                intent.get(
                    "confidence"
                ),
            "low_confidence":
                intent.get(
                    "low_confidence",
                    False,
                ),
            "resolution_method":
                intent.get(
                    "resolution_method"
                ),
            "override_applied":
                intent.get(
                    "override_applied",
                    False,
                ),
            "matched_keyword":
                intent.get(
                    "matched_keyword"
                ),
        },
        "routing": {
            "mode":
                route.get(
                    "routing_mode"
                ),
            "primary_agent":
                route.get(
                    "primary_agent"
                ),
            "planned_agents":
                route.get(
                    "planned_agents",
                    [],
                ),
            "executed_agents":
                workflow_result.get(
                    "executed_agents",
                    [],
                ),
            "agent_count":
                route.get(
                    "agent_count",
                    0,
                ),
        },
        "identifiers": {
            "transaction_id":
                workflow_result.get(
                    "transaction_id"
                ),
            "application_id":
                workflow_result.get(
                    "application_id"
                ),
        },
        "human_review_required":
            workflow_result.get(
                "human_review_required",
                False,
            ),
        "worker_details":
            _extract_worker_details(
                worker_results
            ),
        "generation": {
            "provider":
                synthesis_llm.get(
                    "provider"
                ),
            "model":
                synthesis_llm.get(
                    "model"
                ),
            "fallback_used":
                synthesis_llm.get(
                    "fallback_used",
                    False,
                ),
            "error_category":
                llm_error_category,
        },
        "execution_errors":
            workflow_result.get(
                "execution_errors",
                [],
            ),
    }

    if include_debug:
        public_result["debug"] = {
            "route_reasons":
                route.get(
                    "reasons",
                    [],
                ),
            "intent_predictions":
                intent.get(
                    "top_predictions",
                    [],
                ),
            "raw_worker_results":
                worker_results,
            "current_workflow_agent":
                workflow_result.get(
                    "agent"
                ),
            "llm_status":
                synthesis_llm.get(
                    "status"
                ),
            "llm_error_category":
                llm_error_category,
        }

    return public_result
