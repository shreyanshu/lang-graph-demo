"""Pydantic schemas that force Gemini to answer with valid JSON.

LangChain's ``with_structured_output`` turns these models into a response
schema for the model, and validates/parses the reply back into Python objects.
That is what makes the planner reliable enough to drive a graph.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Step(BaseModel):
    """One atomic unit of work in a plan."""

    id: int = Field(description="1-based position of this step in the plan.")
    description: str = Field(
        description=(
            "A concrete, self-contained instruction for the executor. "
            "It must be possible to complete this step in one go."
        )
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Names of the tools likely needed for this step (may be empty).",
    )


class Plan(BaseModel):
    """The initial plan produced by the planner node."""

    goal: str = Field(description="One-sentence restatement of what the user wants.")
    steps: list[Step] = Field(description="Ordered list of steps, 1 to 8 items.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made because the request was ambiguous.",
    )


class ReplanDecision(BaseModel):
    """The replanner either finishes the task or adds follow-up steps."""

    reasoning: str = Field(description="Why the task is or is not complete yet.")
    is_complete: bool = Field(
        description="True only when the results fully answer the original request."
    )
    final_answer: str | None = Field(
        default=None,
        description="The complete answer for the user. Required when is_complete is True.",
    )
    next_steps: list[Step] = Field(
        default_factory=list,
        description="Additional steps required when is_complete is False.",
    )


class ReviewDecision(BaseModel):
    """What the human reviewer sends back to the graph."""

    approved: bool
    feedback: str = ""
