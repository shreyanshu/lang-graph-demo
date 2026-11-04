"""Assemble the planner/executor graph.

        START
          |
       planner <--------------+
          |                   |
     review_plan --reject-----+
          | approve
       executor <----+
          |          |
          +----------+  (more steps)
          |
      replanner --done--> END
          |
     (budget spent)
          |
      finalizer --> END

   executor --(plan empty, revision budget spent)--> finalizer
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    executor_node,
    finalize_node,
    planner_node,
    replanner_node,
    review_plan_node,
    route_after_executor,
    route_after_replanner,
    route_after_review,
)
from .state import PlanExecuteState


def build_graph(*, checkpointer=None, store=None, debug: bool = False):
    """Create and compile the graph.

    ``checkpointer`` enables memory, human-in-the-loop (``interrupt``) and time
    travel. ``store`` enables long-term memory that outlives a thread.
    """
    builder = StateGraph(PlanExecuteState)

    builder.add_node("planner", planner_node)
    builder.add_node("review_plan", review_plan_node)
    builder.add_node("executor", executor_node)
    builder.add_node("replanner", replanner_node)
    builder.add_node("finalizer", finalize_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "review_plan")

    builder.add_conditional_edges(
        "review_plan",
        route_after_review,
        {"planner": "planner", "executor": "executor"},
    )
    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {"executor": "executor", "replanner": "replanner", "finalize": "finalizer"},
    )
    builder.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {"executor": "executor", "finalize": "finalizer", "end": END},
    )
    builder.add_edge("finalizer", END)

    return builder.compile(checkpointer=checkpointer, store=store, debug=debug)


def initial_state(task: str, *, auto_approve: bool = True) -> dict:
    """The input state for a fresh run."""
    return {
        "task": task,
        "plan": [],
        "plan_feedback": "",
        "auto_approve": auto_approve,
        "past_steps": [],
        "response": "",
        "revision_count": 0,
    }
