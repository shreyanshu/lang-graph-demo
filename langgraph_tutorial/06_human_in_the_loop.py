"""06 - Human in the loop: interrupt(), Command(resume=...) and breakpoints.

Agents that send emails, spend money or delete data should not run unattended.
LangGraph gives you two ways to pause:

  A. `interrupt(value)` inside a node  - dynamic, can pause anywhere and return
     whatever value you want. Resuming re-executes the node from the start, and
     `interrupt()` returns the value you pass to `Command(resume=...)`.

  B. `compile(interrupt_before=[...])` / `interrupt_after=[...]` - static
     breakpoints. The graph stops before/after those nodes; resume with
     `invoke(None, config)`.

Both require a checkpointer: the paused state lives in a checkpoint.

Run: python langgraph_tutorial/06_human_in_the_loop.py
Needs: nothing.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from _shared import banner, note, show, show_mermaid


class EmailState(TypedDict):
    request: str
    draft: str
    feedback: str
    approved: bool
    sent: bool
    auto_approve: bool
    history: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
def draft(state: EmailState) -> dict:
    base = f"Subject: {state['request'][:40]}\n\nHi team,\n\n"
    body = "Here is the update you asked for."
    if state.get("feedback"):
        body += f"\n\n(Revised to address: {state['feedback']})"
    return {"draft": base + body + "\n\nBest,\nShreyansh", "history": ["drafted"]}


def review(state: EmailState) -> dict:
    """Pause and let a human approve or edit the draft."""
    if state.get("auto_approve"):
        # Used by the static-breakpoint demo, so control reaches `send`.
        return {"approved": True, "history": ["auto-approved"]}

    decision = interrupt(
        {
            "action": "review_email",
            "draft": state["draft"],
            "question": "Approve, edit, or reject?",
        }
    )

    if isinstance(decision, str):  # allow a bare "yes"/"no" reply
        approved = decision.strip().lower() in {"y", "yes", "ok", "approve"}
        return {
            "approved": approved,
            "feedback": "" if approved else decision,
            "history": [f"reviewed: {'approved' if approved else 'rejected'}"],
        }

    return {
        "approved": bool(decision.get("approved")),
        "feedback": str(decision.get("feedback", "")),
        "draft": str(decision.get("draft") or state["draft"]),
        "history": [f"reviewed: {'approved' if decision.get('approved') else 'rejected'}"],
    }


def send(state: EmailState) -> dict:
    return {"sent": True, "history": [f"sent -> {state['draft'].splitlines()[0]}"]}


def after_review(state: EmailState) -> str:
    return "send" if state["approved"] else "draft"


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------
def make_builder():
    g = StateGraph(EmailState)
    g.add_node("draft", draft)
    g.add_node("review", review)
    g.add_node("send", send)
    g.add_edge(START, "draft")
    g.add_edge("draft", "review")
    g.add_conditional_edges("review", after_review, {"send": "send", "draft": "draft"})
    g.add_edge("send", END)
    return g


def fresh_input(auto_approve: bool = False) -> dict:
    return {
        "request": "Announce the v2 release",
        "draft": "",
        "feedback": "",
        "approved": False,
        "sent": False,
        "auto_approve": auto_approve,
        "history": [],
    }


def main() -> None:
    banner("06 - Human in the loop")

    saver = InMemorySaver()
    app = make_builder().compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "email-1"}}

    show_mermaid(app)

    note(
        """
        `invoke` runs until `interrupt()` is hit, then RETURNS instead of
        finishing. The returned state contains the pending interrupt; it is also
        available as `app.get_state(config).interrupts`.
        """
    )

    app.invoke(fresh_input(), config)
    paused = app.get_state(config)

    print(f"\n  next node : {paused.next}")
    print(f"  interrupts: {len(paused.interrupts)}")
    print(f"\n  draft:\n{paused.values['draft']}")
    payload = paused.interrupts[0].value
    print(f"\n  interrupt value: {payload}")

    note(
        """
        Resume by passing `Command(resume=<value>)` with the SAME config. The
        `review` node re-runs from the top; `interrupt()` immediately returns
        the resume value instead of pausing again.

        Careful: code before `interrupt()` runs twice. Keep it side-effect free
        (or make it idempotent).
        """
    )

    app.invoke(
        Command(resume={"approved": False, "feedback": "Mention the date"}),
        config,
    )
    rejected = app.get_state(config)
    print(f"\n  after rejection -> next={rejected.next}, feedback={rejected.values['feedback']!r}")
    print("  draft now contains the feedback:", "Mention the date" in rejected.values["draft"])

    note(
        """
        A node can also EDIT the draft before approving, because the resume value
        is arbitrary Python data. Approve with a new draft:
        """
    )

    edited = rejected.values["draft"].replace("Best,", "Cheers,")
    app.invoke(Command(resume={"approved": True, "draft": edited}), config)
    done = app.get_state(config)

    show("final state", {k: done.values[k] for k in ("approved", "sent")})
    print(f"\n  history: {done.values['history']}")
    print(f"  next   : {done.next}  (empty means the run finished)")

    # -----------------------------------------------------------------------
    # static breakpoints
    # -----------------------------------------------------------------------
    note(
        """
        `interrupt_before=["send"]` is the declarative alternative. The graph
        stops AFTER the previous node and BEFORE `send`. Nothing pauses inside a
        node, so `invoke` simply returns and you resume with `invoke(None, ...)`.

        This is also where `update_state` shines: inspect, edit, then continue.
        """
    )

    app2 = make_builder().compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["send"],
    )
    config2 = {"configurable": {"thread_id": "email-2"}}
    app2.invoke(fresh_input(auto_approve=True), config2)

    paused2 = app2.get_state(config2)
    print(f"\n  paused before: {paused2.next}")

    app2.update_state(config2, {"draft": paused2.values["draft"] + "\n\nP.S. Edited by a human."})
    app2.invoke(None, config2)

    final2 = app2.get_state(config2)
    print(f"  sent={final2.values['sent']}  next={final2.next}")
    print(f"  human edit survived: {'Edited by a human' in final2.values['draft']}")

    note(
        """
        Tips:
        * Always show the user the payload BEFORE resuming; never auto-approve.
        * Validate the resume value (it comes from the outside world).
        * Use `interrupt_after` for audit points, `interrupt_before` for gates.
        * In a web app, one HTTP request runs until the interrupt; a later
          request sends `Command(resume=...)` with the stored thread id.
        """
    )


if __name__ == "__main__":
    main()
