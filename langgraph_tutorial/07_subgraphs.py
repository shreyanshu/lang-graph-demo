"""07 - Subgraphs: graphs as nodes.

A subgraph is simply a compiled graph used as a node:

    parent.add_node("research", research_graph)

Why bother?
  * isolation   - the inner graph has its OWN state schema; the parent only sees
                  the keys they share
  * reuse       - one tested research graph, many parent agents
  * readability - the parent diagram stays small

Data flow: the parent's state dict is passed to the subgraph, the subgraph
ignores keys it does not know, and the keys it returns are merged back into the
parent state (through the parent's reducers).

Run: python langgraph_tutorial/07_subgraphs.py
Needs: nothing.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from _shared import banner, note, show, show_mermaid


# ---------------------------------------------------------------------------
# inner graph: research -> summarise
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]
    summary: str


def search(state: ResearchState) -> dict:
    topic = state["topic"]
    return {
        "notes": [
            f"{topic} runs graphs as super-steps.",
            f"{topic} supports cycles, persistence and interrupts.",
            f"{topic} is provider agnostic.",
        ]
    }


def summarise(state: ResearchState) -> dict:
    first = state["notes"][0] if state["notes"] else "no notes"
    return {"summary": f"{len(state['notes'])} findings about {state['topic']}. Key point: {first}"}


def build_research():
    g = StateGraph(ResearchState)
    g.add_node("search", search)
    g.add_node("summarise", summarise)
    g.add_edge(START, "search")
    g.add_edge("search", "summarise")
    g.add_edge("summarise", END)
    return g.compile()


# ---------------------------------------------------------------------------
# parent graph: prepare -> (subgraph) -> publish
# ---------------------------------------------------------------------------
class BlogState(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]
    summary: str
    post: str


def prepare(state: BlogState) -> dict:
    return {"topic": state.get("topic") or "LangGraph"}


def publish(state: BlogState) -> dict:
    post = f"# {state['topic']}\n\n{state['summary']}\n\nSources: {len(state['notes'])} notes."
    return {"post": post}


def build_parent(research_graph):
    g = StateGraph(BlogState)
    g.add_node("prepare", prepare)
    g.add_node("research", research_graph)  # <- a whole graph as a node
    g.add_node("publish", publish)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "research")
    g.add_edge("research", "publish")
    g.add_edge("publish", END)
    return g.compile()


def main() -> None:
    banner("07 - Subgraphs")

    research = build_research()
    parent = build_parent(research)

    print("\n--- inner graph (ResearchState) " + "-" * 30)
    print(research.get_graph().draw_mermaid())

    note(
        """
        The parent passes its ENTIRE state to the subgraph. ResearchState only
        declares topic/notes/summary, so `post` is ignored on the way in and
        never returned. `summary` is written by the subgraph and lands in the
        parent state.

        The parent does not need to know that `research` contains two nodes.
        """
    )

    result = parent.invoke({"topic": "", "notes": [], "summary": "", "post": ""})
    show("parent result", result)

    note(
        """
        Streaming with `subgraphs=True` reveals the nested namespaces. Each
        chunk becomes (namespace, mode, data); the namespace is a tuple like
        ('research:<task-id>',) for nodes inside the subgraph.
        """
    )

    print("\n  stream with subgraphs=True:")
    for item in parent.stream(
        {"topic": "Subgraphs", "notes": [], "summary": "", "post": ""},
        stream_mode="updates",
        subgraphs=True,
    ):
        if len(item) == 3:  # (namespace, mode, chunk) when several modes are used
            namespace, _mode, chunk = item
        else:  # (namespace, chunk)
            namespace, chunk = item
        nodes = list(chunk.keys()) if isinstance(chunk, dict) else chunk
        depth = "  " * len(namespace)
        print(f"    {depth}{namespace or 'parent'} -> {nodes}")

    note(
        """
        Advanced options:
        * Give the subgraph its own checkpointer to persist it independently;
          otherwise it inherits the parent's checkpointer (recommended default).
        * `Command(goto=..., graph=Command.PARENT)` lets a node inside the
          subgraph jump back to a node in the parent graph.
        * Prefer plain function composition for trivial helpers - a subgraph is
          worth it when the inner flow has its own state, loops or reuse.
        """
    )

    show_mermaid(parent)


if __name__ == "__main__":
    main()
