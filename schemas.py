from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class FinSecureWorkflowState(
    TypedDict,
    total=False,
):
    """
    Shared state untuk workflow LangGraph
    FinSecure Agentic.
    """

    user_message: str

    transaction_id: str | None
    application_id: str | None

    intent_result: dict[str, Any]
    route_plan: dict[str, Any]

    worker_results: list[
        dict[str, Any]
    ]

    executed_agents: list[str]

    execution_errors: list[
        dict[str, str]
    ]

    human_review_required: bool

    final_response: str

    synthesis_llm: dict[str, Any]

    status: str

    current_node: str