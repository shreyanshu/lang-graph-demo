"""13 - Command: update + routing in one return value.

So far a node returned a state update and a conditional edge decided where to
go. `Command` merges both:

    return Command(update={"route": "billing"}, goto="billing")

Why it matters:
  * agent HANDOFFS read naturally: the agent decides its successor
  * one node can own its routing logic without a separate router function
  * `goto` can be a list - including a list of `Send` objects for map-reduce
  * `graph=Command.PARENT` lets a node inside a subgraph jump back out to the
    parent graph

Declare `destinations=(...)` on `add_node` so the diagram and validation know
where a `Command` may jump.

Run: python langgraph_tutorial/13_command_and_handoffs.py
Needs: nothing.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from _shared import banner, note, show, show_mermaid


# ---------------------------------------------------------------------------
# Part A - a triage agent that hands off with Command
# ---------------------------------------------------------------------------
class TicketState(TypedDict):
    request: str
    answer: str
    log: Annotated[list[str], operator.add]


def triage(state: TicketState) -> Command:
    text = state["request"].lower()
    if "refund" in text or "invoice" in text:
        return Command(
            update={"log": ["triage -> billing"]},
            goto="billing",
        )
    if "bug" in text or "error" in text:
        return Command(
            update={"log": ["triage -> engineering"]},
            goto="engineering",
        )
    return Command(update={"log": ["triage -> general"]}, goto="general")


def billing(state: TicketState) -> dict:
    return {"answer": "Billing will process your refund within 3 days.", "log": ["billing replied"]}


def engineering(state: TicketState) -> dict:
    return {"answer": "Engineering reproduced the bug and opened issue #42.", "log": ["engineering replied"]}


def general(state: TicketState) -> dict:
    return {"answer": "Here is a general help article.", "log": ["general replied"]}


def build_triage():
    g = StateGraph(TicketState)
    # destinations=... documents the possible Command targets.
    g.add_node("triage", triage, destinations=("billing", "engineering", "general"))
    g.add_node("billing", billing)
    g.add_node("engineering", engineering)
    g.add_node("general", general)
    g.add_edge(START, "triage")
    for node in ("billing", "engineering", "general"):
        g.add_edge(node, END)
    return g.compile()


# ---------------------------------------------------------------------------
# Part B - Command(goto=[Send, Send, ...]) for map-reduce
# ---------------------------------------------------------------------------
class BatchState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]


def dispatch(state: BatchState) -> Command:
    # Equivalent to returning a list of Send from a conditional edge, but the
    # node decides the fan-out itself.
    return Command(
        goto=[Send("worker", {"item": item}) for item in state["items"]]
    )


def worker(state: dict) -> dict:
    item = state["item"]
    return {"results": [f"{item.upper()} ({len(item)} chars)"]}


def collect(state: BatchState) -> dict:
    return {"results": [f"collected {len(state['results'])} results"]}


def build_batch():
    g = StateGraph(BatchState)
    g.add_node("dispatch", dispatch, destinations=("worker",))
    g.add_node("worker", worker)
    g.add_node("collect", collect)
    g.add_edge(START, "dispatch")
    g.add_edge("worker", "collect")
    g.add_edge("collect", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Part C - jump out of a subgraph with Command.PARENT
# ---------------------------------------------------------------------------
class ParentState(TypedDict):
    log: Annotated[list[str], operator.add]


def inner_node(state: ParentState) -> Command:
    return Command(
        update={"log": ["subgraph decided to escalate"]},
        graph=Command.PARENT,  # route in the PARENT graph, not this one
        goto="escalate",
    )


def build_subgraph():
    g = StateGraph(ParentState)
    g.add_node("inner_node", inner_node)
    g.add_edge(START, "inner_node")
    g.add_edge("inner_node", END)
    return g.compile()


def escalate(state: ParentState) -> dict:
    return {"log": ["parent handled the escalation"]}


def build_parent():
    g = StateGraph(ParentState)
    g.add_node("sub", build_subgraph())
    g.add_node("escalate", escalate)
    g.add_edge(START, "sub")
    g.add_edge("escalate", END)
    return g.compile()


def main() -> None:
    banner("13 - Command: update + goto in one return")

    note(
        """
        There is no conditional edge in this graph. `triage` returns
        `Command(update=..., goto=...)`; LangGraph applies the update and
        schedules the target in the same step. This is the standard way to model
        agent handoffs (supervisor -> specialist).
        """
    )

    triage_app = build_triage()
    show_mermaid(triage_app)

    for request in (
        "I need a refund for invoice 88",
        "The app throws an error on import",
        "How do I rename a project?",
    ):
        result = triage_app.invoke({"request": request, "answer": "", "log": []})
        print(f"\n  {request!r}\n    log: {result['log']}\n    answer: {result['answer']}")

    note(
        """
        `Command(goto=[Send(...), Send(...)])` combines dynamic fan-out with
        state updates. Handy when the fan-out decision belongs to the node
        rather than to a router function.

        One catch: the node does NOT return a state update for `results`, so
        only the workers write it. The reducer merges their writes.
        """
    )

    batch = build_batch()
    out = batch.invoke({"items": ["alpha", "beta", "gamma"], "results": []})
    print(f"\n  results: {out['results']}")

    note(
        """
        `graph=Command.PARENT` from inside a subgraph jumps to a node in the
        parent graph. Use it for escalation/handoff flows: a small, focused
        subgraph can decide "this needs the parent's approval step" without the
        parent knowing the details.
        """
    )

    parent = build_parent()
    show("parent result", parent.invoke({"log": []}))

    note(
        """
        Rules and gotchas:
        * `Command` and a normal dict update cannot be mixed in one return.
        * `Command` can only be returned from a node (not from a router).
        * Declare `destinations` so the mermaid diagram is accurate.
        * `Command(resume=...)` is the *input* for resuming an interrupt; it is
          the same class, different role (see 06_human_in_the_loop.py).
        """
    )


if __name__ == "__main__":
    main()
