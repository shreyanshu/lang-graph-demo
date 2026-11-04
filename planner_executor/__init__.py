"""A plan-and-execute agent built with LangGraph + Google Gemini.

Run it with::

    python -m planner_executor.main "Compare the population of India and China"

See ``planner_executor/README.md`` for the architecture walkthrough.
"""

from .config import SETTINGS, get_llm, has_api_key
from .graph import build_graph, initial_state
from .state import PlanExecuteState

__all__ = [
    "SETTINGS",
    "PlanExecuteState",
    "build_graph",
    "get_llm",
    "has_api_key",
    "initial_state",
]
