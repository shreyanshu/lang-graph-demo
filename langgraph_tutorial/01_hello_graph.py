"""01 - Hello graph: nodes, edges, state and compilation.

A LangGraph app is three things:

  1. a STATE schema   -> the data that flows between steps
  2. NODES            -> plain functions: state -> partial state update
  3. EDGES            -> which node runs next (static or conditional)

``START`` is a virtual node where input enters; ``END`` is a virtual node that
terminates the run. ``.compile()`` validates the wiring and returns a runnable
graph you can ``invoke``, ``stream`` or ``ainvoke``.

Run: python langgraph_tutorial/01_hello_graph.py
Needs: nothing (no API key).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from _shared import banner, note, show, show_mermaid


# ---------------------------------------------------------------------------
# 1. State: every node receives this and returns a *partial* update
# ---------------------------------------------------------------------------
class GreetingState(TypedDict):
    raw_name: str
    name: str
    greeting: str
    log: Annotated[list[str], operator.add]  # reducer: append, never overwrite


# ---------------------------------------------------------------------------
# 2. Nodes: ordinary Python functions
# ---------------------------------------------------------------------------
def normalize(state: GreetingState) -> dict:
    """Strip spaces, title-case the name, and record what happened."""
    clean = state["raw_name"].strip().title()
    return {"name": clean, "log": ["normalize"]}


def greet(state: GreetingState) -> dict:
    """Build the greeting and add personalised wording for empty names."""
    name = state.get("name") or "stranger"
    return {"greeting": f"Hello, {name}!", "log": ["greet"]}


def sign_off(state: GreetingState) -> dict:
    return {"greeting": state["greeting"] + " Have a great day.", "log": ["sign_off"]}


# ---------------------------------------------------------------------------
# 3. Wire it up
# ---------------------------------------------------------------------------
def build() -> object:
    builder = StateGraph(GreetingState)          # <- state schema goes here

    builder.add_node("normalize", normalize)     # name -> function
    builder.add_node("greet", greet)
    builder.add_node("sign_off", sign_off)

    builder.add_edge(START, "normalize")         # entry point
    builder.add_edge("normalize", "greet")       # always after normalize
    builder.add_edge("greet", "sign_off")
    builder.add_edge("sign_off", END)            # exit

    return builder.compile()


def main() -> None:
    banner("01 - Hello graph")

    note(
        """
        Nodes only return the keys they change. Keys you omit keep their old
        value. Because `log` has an `operator.add` reducer, each node's list is
        APPENDED to the existing list instead of replacing it.

        Try removing `Annotated[..., operator.add]` and watch `log` keep only
        the last node's entry.
        """
    )

    app = build()
    result = app.invoke({"raw_name": "   ada lovelace  ", "log": []})

    show("final state", result)
    print("\n  greeting ->", result["greeting"])
    print("  log      ->", result["log"])

    note(
        """
        The returned dict is the FULL state, and node order is exactly the edge
        order. LangGraph runs nodes in "super-steps": all ready nodes run, then
        the state updates are merged, then the next nodes are scheduled.

        `invoke` is synchronous. `ainvoke`, `stream` and `astream` are the same
        graph with different execution/observation modes (see 09_streaming.py).
        """
    )

    show_mermaid(app)


if __name__ == "__main__":
    main()
