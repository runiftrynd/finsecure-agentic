from __future__ import annotations

import json
from typing import Any, Literal

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from agents.customer_service_agent import (
    run_customer_service_agent,
)
from agents.fraud_risk_agent import (
    run_fraud_risk_agent,
)
from agents.kyc_compliance_agent import (
    run_kyc_compliance_agent,
)
from agents.supervisor_agent import (
    extract_application_id,
    extract_transaction_id,
    plan_agent_route,
)
from llm_client import generate_text
from schemas import FinSecureWorkflowState
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

SYNTHESIZE_NODE = "synthesize"


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


def _clean_identifier(
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


def _append_worker_result(
    state: FinSecureWorkflowState,
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Menambahkan hasil Worker Agent ke state.
    """

    worker_results = list(
        state.get(
            "worker_results",
            [],
        )
    )

    executed_agents = list(
        state.get(
            "executed_agents",
            [],
        )
    )

    worker_results.append(
        result
    )

    agent_name = str(
        result.get(
            "agent",
            "unknown_agent",
        )
    )

    executed_agents.append(
        agent_name
    )

    return {
        "worker_results":
            worker_results,
        "executed_agents":
            executed_agents,
        "current_node":
            agent_name,
    }


def _append_execution_error(
    state: FinSecureWorkflowState,
    agent_name: str,
    error: Exception,
) -> dict[str, Any]:
    """
    Menambahkan error eksekusi tanpa
    menghentikan seluruh workflow.
    """

    execution_errors = list(
        state.get(
            "execution_errors",
            [],
        )
    )

    execution_errors.append(
        {
            "agent": agent_name,
            "error": str(error),
            "error_type":
                type(error).__name__,
        }
    )

    return {
        "execution_errors":
            execution_errors,
        "current_node":
            agent_name,
    }


def prepare_request_node(
    state: FinSecureWorkflowState,
) -> dict[str, Any]:
    """
    Membersihkan input, mendeteksi identifier,
    menjalankan intent resolver, dan membuat
    rencana routing.
    """

    clean_message = str(
        state.get(
            "user_message",
            "",
        )
    ).strip()

    if not clean_message:
        raise ValueError(
            "Pesan pengguna tidak boleh kosong."
        )

    transaction_id = (
        _clean_identifier(
            state.get(
                "transaction_id"
            )
        )
        or extract_transaction_id(
            clean_message
        )
    )

    application_id = (
        _clean_identifier(
            state.get(
                "application_id"
            )
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
        transaction_id=transaction_id,
        application_id=application_id,
    )

    return {
        "user_message":
            clean_message,
        "transaction_id":
            transaction_id,
        "application_id":
            application_id,
        "intent_result":
            intent_result,
        "route_plan":
            route_plan,
        "worker_results":
            [],
        "executed_agents":
            [],
        "execution_errors":
            [],
        "human_review_required":
            False,
        "status":
            "processing",
        "current_node":
            "prepare_request",
    }


def customer_service_node(
    state: FinSecureWorkflowState,
) -> dict[str, Any]:
    """
    Node Customer Service Agent.
    """

    try:
        result = run_customer_service_agent(
            user_message=state[
                "user_message"
            ],
            intent_result=state[
                "intent_result"
            ],
            top_k=2,
            use_llm=False,
        )

        return _append_worker_result(
            state,
            result,
        )

    except Exception as error:
        return _append_execution_error(
            state,
            CUSTOMER_SERVICE_AGENT,
            error,
        )


def fraud_risk_node(
    state: FinSecureWorkflowState,
) -> dict[str, Any]:
    """
    Node Fraud & Risk Agent.
    """

    try:
        result = run_fraud_risk_agent(
            user_message=state[
                "user_message"
            ],
            transaction_id=state.get(
                "transaction_id"
            ),
            intent_result=state[
                "intent_result"
            ],
            top_k=2,
            use_llm=False,
        )

        return _append_worker_result(
            state,
            result,
        )

    except Exception as error:
        return _append_execution_error(
            state,
            FRAUD_RISK_AGENT,
            error,
        )


def kyc_compliance_node(
    state: FinSecureWorkflowState,
) -> dict[str, Any]:
    """
    Node KYC & Compliance Agent.
    """

    try:
        result = run_kyc_compliance_agent(
            user_message=state[
                "user_message"
            ],
            application_id=state.get(
                "application_id"
            ),
            intent_result=state[
                "intent_result"
            ],
            top_k=2,
            use_llm=False,
        )

        return _append_worker_result(
            state,
            result,
        )

    except Exception as error:
        return _append_execution_error(
            state,
            KYC_COMPLIANCE_AGENT,
            error,
        )


def _compact_worker_result(
    worker_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Mengambil informasi penting Worker Agent
    untuk sintesis akhir.
    """

    agent_name = worker_result.get(
        "agent"
    )

    compact_result: dict[str, Any] = {
        "agent":
            agent_name,
        "status":
            worker_result.get(
                "status"
            ),
        "response":
            worker_result.get(
                "response"
            ),
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

        transaction = (
            transaction_result.get(
                "transaction",
                {},
            )
        )

        compact_result.update(
            {
                "transaction_id":
                    worker_result.get(
                        "transaction_id"
                    ),
                "transaction_lookup_status":
                    transaction_result.get(
                        "status"
                    ),
                "transaction_status":
                    transaction.get(
                        "transaction_status"
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
    worker_results: list[
        dict[str, Any]
    ],
) -> str:
    """
    Membuat respons akhir ketika Gemini
    tidak tersedia.
    """

    valid_responses = [
        str(
            result.get(
                "response"
            )
        ).strip()
        for result in worker_results
        if result.get(
            "response"
        )
    ]

    if not valid_responses:
        return (
            "Permintaan telah diproses, tetapi "
            "belum ada jawaban yang dapat ditampilkan."
        )

    return "\n\n".join(
        valid_responses
    )


def synthesize_node(
    state: FinSecureWorkflowState,
) -> dict[str, Any]:
    """
    Menggabungkan hasil Worker Agent menjadi
    satu jawaban akhir.
    """

    worker_results = list(
        state.get(
            "worker_results",
            [],
        )
    )

    execution_errors = list(
        state.get(
            "execution_errors",
            [],
        )
    )

    human_review_required = any(
        bool(
            result.get(
                "human_review_required",
                False,
            )
        )
        for result in worker_results
    )

    deterministic_response = (
        _create_deterministic_response(
            worker_results
        )
    )

    if not worker_results:
        return {
            "status":
                "failure",
            "human_review_required":
                False,
            "final_response": (
                "Permintaan belum dapat diproses "
                "karena seluruh Worker Agent gagal."
            ),
            "synthesis_llm": {
                "status": "skipped",
                "provider": "local",
                "model": None,
                "fallback_used": True,
                "error": (
                    "Tidak ada Worker Agent "
                    "yang berhasil dijalankan."
                ),
            },
            "current_node":
                SYNTHESIZE_NODE,
        }

    compact_results = [
        _compact_worker_result(
            result
        )
        for result in worker_results
    ]

    synthesis_prompt = f"""
Anda adalah sistem layanan FinSecure.

Gabungkan hasil pemeriksaan berikut menjadi satu jawaban
yang jelas, formal, dan mudah dipahami oleh nasabah.

PESAN NASABAH:
{state["user_message"]}

HASIL ROUTING:
{_to_json(state.get("route_plan", {}))}

HASIL PEMERIKSAAN:
{_to_json(compact_results)}

ATURAN:
1. Jangan mengubah hasil pemeriksaan Worker Agent.
2. Jangan mengubah status transaksi, status KYC,
   probabilitas, risk level, atau rekomendasi.
3. Status operasional transaksi dan hasil fraud model
   merupakan dua informasi yang berbeda.
4. Jangan menyatakan bahwa fraud score menyebabkan
   transaksi pending, gagal, selesai, atau dibatalkan.
5. Gunakan istilah indikasi risiko, bukan tuduhan fraud.
6. Jelaskan kebutuhan Transaction ID atau Application ID
   apabila identifier belum tersedia.
7. Jelaskan apabila pemeriksaan manual diperlukan.
8. Jangan meminta PIN, OTP, CVV, password,
   atau kode keamanan.
9. Jangan menyebut nama agent, prompt, embedding,
   vector database, atau arsitektur internal.
10. Jangan menampilkan label internal dengan underscore.
11. Maksimal delapan kalimat.
""".strip()

    synthesis_result = generate_text(
        prompt=synthesis_prompt,
        max_retries=1,
    )

    if synthesis_result.get(
        "fallback_used",
        False,
    ):
        final_response = (
            deterministic_response
        )
    else:
        final_response = str(
            synthesis_result.get(
                "response",
                deterministic_response,
            )
        ).strip()

    if execution_errors:
        workflow_status = (
            "partial_failure"
        )
    else:
        workflow_status = "success"

    return {
        "status":
            workflow_status,
        "human_review_required":
            human_review_required,
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
        "current_node":
            SYNTHESIZE_NODE,
    }


def route_after_prepare(
    state: FinSecureWorkflowState,
) -> Literal[
    "customer_service",
    "fraud_risk",
    "kyc_compliance",
    "synthesize",
]:
    """
    Menentukan Worker Agent pertama.
    """

    planned_agents = state.get(
        "route_plan",
        {},
    ).get(
        "planned_agents",
        [],
    )

    if CUSTOMER_SERVICE_AGENT in planned_agents:
        return "customer_service"

    if FRAUD_RISK_AGENT in planned_agents:
        return "fraud_risk"

    if KYC_COMPLIANCE_AGENT in planned_agents:
        return "kyc_compliance"

    return "synthesize"


def route_after_customer_service(
    state: FinSecureWorkflowState,
) -> Literal[
    "fraud_risk",
    "kyc_compliance",
    "synthesize",
]:
    """
    Menentukan node setelah Customer Service.
    """

    planned_agents = state.get(
        "route_plan",
        {},
    ).get(
        "planned_agents",
        [],
    )

    if FRAUD_RISK_AGENT in planned_agents:
        return "fraud_risk"

    if KYC_COMPLIANCE_AGENT in planned_agents:
        return "kyc_compliance"

    return "synthesize"


def route_after_fraud(
    state: FinSecureWorkflowState,
) -> Literal[
    "kyc_compliance",
    "synthesize",
]:
    """
    Menentukan node setelah Fraud & Risk.
    """

    planned_agents = state.get(
        "route_plan",
        {},
    ).get(
        "planned_agents",
        [],
    )

    if KYC_COMPLIANCE_AGENT in planned_agents:
        return "kyc_compliance"

    return "synthesize"


def build_finsecure_workflow():
    """
    Membuat dan mengompilasi LangGraph
    FinSecure Agentic.
    """

    graph_builder = StateGraph(
        FinSecureWorkflowState
    )

    graph_builder.add_node(
        "prepare_request",
        prepare_request_node,
    )

    graph_builder.add_node(
        "customer_service",
        customer_service_node,
    )

    graph_builder.add_node(
        "fraud_risk",
        fraud_risk_node,
    )

    graph_builder.add_node(
        "kyc_compliance",
        kyc_compliance_node,
    )

    graph_builder.add_node(
        "synthesize",
        synthesize_node,
    )

    graph_builder.add_edge(
        START,
        "prepare_request",
    )

    graph_builder.add_conditional_edges(
        "prepare_request",
        route_after_prepare,
        {
            "customer_service":
                "customer_service",
            "fraud_risk":
                "fraud_risk",
            "kyc_compliance":
                "kyc_compliance",
            "synthesize":
                "synthesize",
        },
    )

    graph_builder.add_conditional_edges(
        "customer_service",
        route_after_customer_service,
        {
            "fraud_risk":
                "fraud_risk",
            "kyc_compliance":
                "kyc_compliance",
            "synthesize":
                "synthesize",
        },
    )

    graph_builder.add_conditional_edges(
        "fraud_risk",
        route_after_fraud,
        {
            "kyc_compliance":
                "kyc_compliance",
            "synthesize":
                "synthesize",
        },
    )

    graph_builder.add_edge(
        "kyc_compliance",
        "synthesize",
    )

    graph_builder.add_edge(
        "synthesize",
        END,
    )

    return graph_builder.compile()


FINSECURE_WORKFLOW = (
    build_finsecure_workflow()
)


def run_finsecure_workflow(
    user_message: str,
    transaction_id: str | None = None,
    application_id: str | None = None,
) -> dict[str, Any]:
    """
    Menjalankan workflow LangGraph dan
    mengembalikan hasil terstruktur.
    """

    initial_state: FinSecureWorkflowState = {
        "user_message":
            str(user_message).strip(),
        "transaction_id":
            transaction_id,
        "application_id":
            application_id,
    }

    final_state = (
        FINSECURE_WORKFLOW.invoke(
            initial_state
        )
    )

    return {
        "status":
            final_state.get(
                "status",
                "failure",
            ),
        "agent":
            "finsecure_langgraph_workflow",
        "user_message":
            final_state.get(
                "user_message"
            ),
        "transaction_id":
            final_state.get(
                "transaction_id"
            ),
        "application_id":
            final_state.get(
                "application_id"
            ),
        "intent":
            final_state.get(
                "intent_result",
                {},
            ),
        "route":
            final_state.get(
                "route_plan",
                {},
            ),
        "executed_agents":
            final_state.get(
                "executed_agents",
                [],
            ),
        "worker_results":
            final_state.get(
                "worker_results",
                [],
            ),
        "execution_errors":
            final_state.get(
                "execution_errors",
                [],
            ),
        "final_response":
            final_state.get(
                "final_response"
            ),
        "synthesis_llm":
            final_state.get(
                "synthesis_llm",
                {},
            ),
        "human_review_required":
            final_state.get(
                "human_review_required",
                False,
            ),
    }