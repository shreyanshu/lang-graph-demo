"""Graph nodes. Each node is a plain function: state in -> partial update out."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from .config import SETTINGS, get_llm
from .prompts import (
    EXECUTOR_STEP,
    EXECUTOR_SYSTEM,
    FINALIZER_SYSTEM,
    PLANNER_WITH_FEEDBACK,
    REPLANNER_SYSTEM,
    planner_system,
)
from .schemas import Plan, ReplanDecision
from .state import PlanExecuteState
from .tools import TOOLS, TOOLS_BY_NAME

MAX_TOOL_ROUNDS = 5
MAX_RESULT_CHARS = 2_500


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _emit(payload: dict[str, Any]) -> None:
    """Emit a custom stream event so the CLI can show progress live."""
    try:
        get_stream_writer()(payload)
    except Exception:  # noqa: BLE001 - custom streaming is best-effort
        pass


def _text(content: Any) -> str:
    """Normalise LangChain message content (str or list of content blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts).strip()
    return str(content).strip()


def _history_text(past_steps: list[dict], limit: int = 6) -> str:
    if not past_steps:
        return "(none yet)"
    start = max(0, len(past_steps) - limit)
    lines = []
    for i, record in enumerate(past_steps[start:], start=start + 1):
        result = record.get("result", "")
        if len(result) > 700:
            result = result[:700] + " ..."
        lines.append(f"{i}. {record.get('step', '')}\n   -> {result}")
    return "\n".join(lines)


def _format_plan(plan: list[dict]) -> str:
    if not plan:
        return "(empty)"
    return "\n".join(f"- [{s.get('id')}] {s.get('description')}" for s in plan)


def _fallback_answer(state: PlanExecuteState) -> str:
    """Deterministic answer when the model cannot produce a final response."""
    if not state.get("past_steps"):
        return "I could not complete the task. No steps were executed."
    lines = [f"Task: {state.get('task', '')}", "", "What was done:"]
    for record in state["past_steps"]:
        lines.append(f"- {record.get('step', '')}: {record.get('result', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. planner
# ---------------------------------------------------------------------------
def planner_node(state: PlanExecuteState) -> dict:
    """Turn the user request into an ordered list of steps."""
    _emit({"node": "planner", "status": "building a plan"})

    messages = [SystemMessage(content=planner_system(SETTINGS.max_steps))]
    human = f"Request:\n{state['task']}"
    feedback = state.get("plan_feedback", "").strip()
    if feedback:
        human += "\n" + PLANNER_WITH_FEEDBACK.format(feedback=feedback)
    messages.append(HumanMessage(content=human))

    plan: Plan = get_llm().with_structured_output(Plan).invoke(messages)

    steps = []
    for new_id, step in enumerate(plan.steps[: SETTINGS.max_steps], start=1):
        steps.append(
            {
                "id": new_id,
                "description": step.description.strip(),
                # keep only tools that actually exist
                "tools": [t for t in step.tools if t in TOOLS_BY_NAME],
            }
        )

    if not steps:  # extremely defensive: structured output should prevent this
        steps = [{"id": 1, "description": state["task"], "tools": []}]

    _emit(
        {
            "node": "planner",
            "status": f"planned {len(steps)} step(s)",
            "plan": steps,
            "assumptions": plan.assumptions,
        }
    )
    return {"plan": steps, "plan_feedback": "", "response": ""}


# ---------------------------------------------------------------------------
# 2. optional human review (LangGraph `interrupt`)
# ---------------------------------------------------------------------------
def review_plan_node(state: PlanExecuteState) -> dict:
    """Pause and ask a human to approve the plan.

    ``interrupt()`` stores its payload in the checkpoint and returns control to
    the caller. On resume the node is re-executed from the start and
    ``interrupt()`` returns the value passed to ``Command(resume=...)``.

    This node has no side effects before the interrupt, so re-execution is safe.
    """
    if state.get("auto_approve", True):
        return {}

    decision = interrupt(
        {
            "type": "plan_review",
            "task": state["task"],
            "plan": state["plan"],
            "question": "Approve this plan? (yes / no + feedback)",
        }
    )

    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        feedback = str(decision.get("feedback", "")).strip()
    else:  # allow a plain string too
        text = str(decision).strip()
        approved = text.lower().startswith(("y", "ok", "approve"))
        feedback = "" if approved else text

    if approved:
        _emit({"node": "review", "status": "plan approved by human"})
        return {"plan_feedback": ""}

    _emit({"node": "review", "status": "plan rejected", "feedback": feedback})
    # Clearing the plan forces the router to send us back to the planner.
    return {"plan_feedback": feedback or "Please produce a better plan.", "plan": []}


# ---------------------------------------------------------------------------
# 3. executor (a mini ReAct loop inside one node)
# ---------------------------------------------------------------------------
def executor_node(state: PlanExecuteState) -> dict:
    """Execute the first remaining step, using tools when needed."""
    plan = list(state.get("plan") or [])
    if not plan:
        return {"plan": []}

    step = plan[0]
    remaining = plan[1:]
    index = len(state.get("past_steps") or []) + 1

    _emit(
        {
            "node": "executor",
            "status": f"step {index}: {step['description'][:90]}",
        }
    )

    llm_with_tools = get_llm(temperature=0).bind_tools(TOOLS)
    tool_hint = (
        f"Suggested tools: {', '.join(step['tools'])}" if step.get("tools") else ""
    )

    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM),
        HumanMessage(
            content=EXECUTOR_STEP.format(
                task=state["task"],
                history=_history_text(state.get("past_steps") or []),
                index=index,
                total=index + len(plan) - 1,
                step=step["description"],
                tool_hint=tool_hint,
            )
        ),
    ]

    answer = ""
    for _ in range(MAX_TOOL_ROUNDS):
        ai = llm_with_tools.invoke(messages)
        messages.append(ai)
        answer = _text(ai.content)

        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            tool = TOOLS_BY_NAME.get(name)
            if tool is None:
                observation = f"ERROR: unknown tool {name!r}"
            else:
                try:
                    observation = str(tool.invoke(args))
                except Exception as exc:  # noqa: BLE001
                    observation = f"ERROR while running {name}: {exc}"
            _emit({"node": "tool", "status": f"{name}({args})"})
            messages.append(
                ToolMessage(content=observation, tool_call_id=call.get("id", name))
            )

    if not answer:
        # Tool loop exhausted with no prose (the last turn was a tool call).
        # Ask once more without tools so the step still yields a result.
        try:
            follow_up = get_llm(temperature=0).invoke(messages)
            answer = _text(follow_up.content)
        except Exception:  # noqa: BLE001
            answer = ""

    if not answer:
        answer = "The step produced no textual result."

    result = answer[:MAX_RESULT_CHARS]
    return {
        "plan": remaining,
        "past_steps": [{"step": step["description"], "result": result}],
    }


# ---------------------------------------------------------------------------
# 4. replanner
# ---------------------------------------------------------------------------
def replanner_node(state: PlanExecuteState) -> dict:
    """Decide: are we done, or do we need more steps?"""
    _emit({"node": "replanner", "status": "checking whether the goal is met"})

    prompt = (
        f"Original request:\n{state['task']}\n\n"
        f"Completed steps:\n{_history_text(state.get('past_steps') or [], limit=20)}\n\n"
        f"Steps that were planned but not executed:\n"
        f"{_format_plan(state.get('plan') or [])}\n\n"
        f"Replanning round {state.get('revision_count', 0) + 1} of "
        f"{SETTINGS.max_revisions}."
    )

    decision: ReplanDecision = (
        get_llm().with_structured_output(ReplanDecision).invoke(
            [SystemMessage(content=REPLANNER_SYSTEM), HumanMessage(content=prompt)]
        )
    )

    revisions = state.get("revision_count", 0) + 1

    if decision.is_complete:
        response = (decision.final_answer or "").strip() or _fallback_answer(state)
        _emit({"node": "replanner", "status": "task complete"})
        return {"response": response, "plan": [], "revision_count": revisions}

    next_steps = []
    for new_id, step in enumerate(decision.next_steps[: SETTINGS.max_steps], start=1):
        next_steps.append(
            {
                "id": new_id,
                "description": step.description.strip(),
                "tools": [t for t in step.tools if t in TOOLS_BY_NAME],
            }
        )

    if not next_steps:
        # Said "not complete" but proposed no work: finish instead of looping.
        return {
            "response": _fallback_answer(state),
            "plan": [],
            "revision_count": revisions,
        }

    _emit(
        {
            "node": "replanner",
            "status": f"added {len(next_steps)} follow-up step(s)",
            "reasoning": decision.reasoning,
            "plan": next_steps,
        }
    )
    return {"plan": next_steps, "revision_count": revisions}


# ---------------------------------------------------------------------------
# 5. finalizer (budget exhausted)
# ---------------------------------------------------------------------------
def finalize_node(state: PlanExecuteState) -> dict:
    """Write the best possible answer when the revision budget is exhausted."""
    _emit({"node": "finalizer", "status": "writing best-effort answer"})

    prompt = (
        f"Original request:\n{state['task']}\n\n"
        f"Completed steps:\n{_history_text(state.get('past_steps') or [], limit=20)}"
    )
    try:
        ai = get_llm().invoke(
            [SystemMessage(content=FINALIZER_SYSTEM), HumanMessage(content=prompt)]
        )
        response = _text(ai.content) or _fallback_answer(state)
    except Exception:  # noqa: BLE001
        response = _fallback_answer(state)

    return {"response": response}


# ---------------------------------------------------------------------------
# routers
# ---------------------------------------------------------------------------
def route_after_review(state: PlanExecuteState) -> str:
    """Rejected plan -> planner. Approved -> executor."""
    if state.get("plan_feedback"):
        return "planner"
    return "executor"


def route_after_executor(state: PlanExecuteState) -> str:
    """Keep executing while steps remain, then ask the replanner.

    The replanner gets at most ``MAX_REVISIONS`` calls; after that the finalizer
    writes the best answer from what was already collected.
    """
    if state.get("plan"):
        return "executor"
    if state.get("revision_count", 0) >= SETTINGS.max_revisions:
        return "finalize"
    return "replanner"


def route_after_replanner(state: PlanExecuteState) -> str:
    """Done -> END. More work -> executor. Nothing left -> finalizer."""
    if state.get("response"):
        return "end"
    if state.get("plan"):
        return "executor"
    return "finalize"
