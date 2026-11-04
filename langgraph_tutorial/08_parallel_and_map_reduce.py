"""08 - Parallel branches and the Send API (map-reduce).

Two kinds of parallelism:

  STATIC  - several edges leave the same node (or START). All target nodes run
            in the same super-step. Their updates must be mergeable, which is
            exactly what reducers are for.

  DYNAMIC - a router returns `Send(node, payload)` objects. The number of
            branches is only known at runtime (e.g. one per search query).
            This is the "map" half of map-reduce; a normal edge into an
            aggregator gives you the "reduce" half.

Run: python langgraph_tutorial/08_parallel_and_map_reduce.py
Needs: nothing.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from _shared import banner, note, show, show_mermaid


# ---------------------------------------------------------------------------
# Part A - static fan-out / fan-in
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    topic: str
    facts: Annotated[list[str], operator.add]  # <- required for parallel writes
    summary: str


def wiki(state: ResearchState) -> dict:
    return {"facts": [f"[wiki] {state['topic']} has an encyclopaedia entry"]}


def news(state: ResearchState) -> dict:
    return {"facts": [f"[news] {state['topic']} was in the news this week"]}


def papers(state: ResearchState) -> dict:
    return {"facts": [f"[papers] a 2025 paper benchmarks {state['topic']}"]}


def combine(state: ResearchState) -> dict:
    return {"summary": f"{len(state['facts'])} sources found for {state['topic']}."}


def build_fanout():
    g = StateGraph(ResearchState)
    g.add_node("wiki", wiki)
    g.add_node("news", news)
    g.add_node("papers", papers)
    g.add_node("combine", combine)

    # Three edges out of START -> three parallel branches in one super-step.
    g.add_edge(START, "wiki")
    g.add_edge(START, "news")
    g.add_edge(START, "papers")

    # They all point at `combine`, which therefore runs only after all three.
    g.add_edge("wiki", "combine")
    g.add_edge("news", "combine")
    g.add_edge("papers", "combine")
    g.add_edge("combine", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Part B - dynamic map-reduce with Send
# ---------------------------------------------------------------------------
class BatchState(TypedDict):
    queries: list[str]
    results: Annotated[list[dict], operator.add]
    report: str


def research_one(state: dict) -> dict:
    """Runs once per Send. The state of this branch is the Send payload."""
    query = state["query"]
    return {"results": [{"query": query, "answer": f"finding for {query!r} (#{len(query)})"}]}


def reduce_results(state: BatchState) -> dict:
    ordered = sorted(state["results"], key=lambda item: item["query"])
    report = "\n".join(f"- {item['query']}: {item['answer']}" for item in ordered)
    return {"report": f"{len(ordered)} branches finished\n{report}"}


def fan_out(state: BatchState) -> list[Send]:
    """Router that returns one Send per item - the map step."""
    return [Send("research_one", {"query": q}) for q in state["queries"]]


def build_map_reduce():
    g = StateGraph(BatchState)
    g.add_node("research_one", research_one)
    g.add_node("reduce", reduce_results)

    # No "dispatch" node: START itself can have a conditional edge.
    g.add_conditional_edges(START, fan_out, ["research_one"])
    g.add_edge("research_one", "reduce")  # implicit fan-in
    g.add_edge("reduce", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Part C - what happens without a reducer
# ---------------------------------------------------------------------------
class NoReducerState(TypedDict):
    topic: str
    facts: list[str]


def write_a(state: NoReducerState) -> dict:
    return {"facts": ["a"]}


def write_b(state: NoReducerState) -> dict:
    return {"facts": ["b"]}


def build_broken():
    g = StateGraph(NoReducerState)
    g.add_node("a", write_a)
    g.add_node("b", write_b)
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", END)
    g.add_edge("b", END)
    return g.compile()


def main() -> None:
    banner("08 - Parallel branches and Send")

    note(
        """
        Static fan-out: `wiki`, `news` and `papers` all start in the same
        super-step. The `facts` key has an `operator.add` reducer, so the three
        writes are concatenated. Without it LangGraph raises InvalidUpdateError
        (see the end of this file).
        """
    )

    app = build_fanout()
    show_mermaid(app)

    print("\n  tasks per super-step (stream_mode='debug'):")
    for chunk in app.stream(
        {"topic": "LangGraph", "facts": [], "summary": ""}, stream_mode="debug"
    ):
        if chunk.get("type") == "task":
            print(f"    step {chunk['step']}: {chunk['payload'].get('name')}")

    show("fan-in result", app.invoke({"topic": "LangGraph", "facts": [], "summary": ""}))

    note(
        """
        Dynamic fan-out with `Send`. The router cannot return a single node name
        here because the work is a list. Instead it returns one Send per query.
        LangGraph creates N tasks in one super-step; `reduce` waits for all of
        them, then runs once.

        Order of `results` is completion order, not input order - sort in the
        reducer if you need determinism.
        """
    )

    batch = build_map_reduce()
    show_mermaid(batch)
    result = batch.invoke(
        {"queries": ["langgraph interrupts", "langgraph checkpoints", "langgraph send api"], "results": [], "report": ""}
    )
    print("\n  " + result["report"].replace("\n", "\n  "))

    note(
        """
        Sanity check: two parallel nodes writing the same key with the default
        (replace) reducer is ambiguous, so LangGraph rejects it at runtime.
        """
    )
    try:
        build_broken().invoke({"topic": "x", "facts": []})
    except InvalidUpdateError as exc:
        print(f"\n  InvalidUpdateError: {str(exc).splitlines()[0]}")

    note(
        """
        When to use which:
        * static parallel - fixed, known branches (different sources, different
          checks). Simple and visible in the diagram.
        * Send - batch work over a runtime-sized list: one item per document,
          per search query, per file, per row.
        * A `Send` payload *replaces* the branch state, so include everything
          that branch needs. Use reducers to merge results back.
        """
    )


if __name__ == "__main__":
    main()
