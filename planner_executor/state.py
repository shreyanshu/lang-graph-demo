"""The shared state of the planner/executor graph.

Rules of thumb used here:

* Keep values JSON-serialisable (dicts/lists/strings/ints) so any checkpointer
  can persist them.
* Use ``Annotated[..., operator.add]`` for channels that must *accumulate*
  (``past_steps``) instead of being overwritten by each node.
* Everything else replaces the previous value by default.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class StepRecord(TypedDict):
    """One completed step plus the executor's result."""

    step: str
    result: str


class PlanExecuteState(TypedDict):
    # --- input ---------------------------------------------------------------
    task: str
    """The original user request. Never overwritten."""

    # --- planning ------------------------------------------------------------
    plan: list[dict]
    """Remaining steps. The executor pops from the front, so an empty list
    means 'the plan is finished, ask the replanner what to do next'."""

    plan_feedback: str
    """Human feedback when a plan was rejected ('' when approved)."""

    auto_approve: bool
    """When True the human-review node is a no-op (used by --yes runs)."""

    # --- execution -----------------------------------------------------------
    past_steps: Annotated[list[StepRecord], operator.add]
    """Completed work. The reducer appends instead of replacing."""

    # --- outcome -------------------------------------------------------------
    response: str
    """Final answer for the user. Non-empty means the graph is done."""

    revision_count: int
    """How many times the replanner has run. Guards against endless loops."""
