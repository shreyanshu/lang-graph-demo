"""09 - Streaming: values, updates, messages, custom, debug (and async).

`invoke` gives you the final state. `stream` gives you the run as it happens -
essential for chat UIs, progress bars and logging.

    stream_mode="values"    full state after every super-step
    stream_mode="updates"   only the delta each node returned  (default-ish)
    stream_mode="messages"  (token_chunk, metadata) from chat models
    stream_mode="custom"    whatever you send with get_stream_writer()
    stream_mode="debug"     task/checkpoint events (advanced)

Pass a LIST to get several modes at once; each item becomes
`(mode, payload)` (or `(namespace, mode, payload)` with subgraphs=True).

Run: python langgraph_tutorial/09_streaming.py
Needs: nothing.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from _shared import ScriptedChatModel, banner, note


class DraftState(TypedDict):
    topic: str
    outline: Annotated[list[str], operator.add]
    messages: Annotated[list, add_messages]
    draft: str


def make_model() -> ScriptedChatModel:
    """A fresh model per node call, so every stream run gets the full script."""
    return ScriptedChatModel(
        script=[
            AIMessage(
                content=(
                    "LangGraph models an agent as a graph of typed state, nodes and "
                    "edges, which makes loops, memory and human review explicit."
                )
            )
        ]
    )


def make_outline(state: DraftState) -> dict:
    writer = get_stream_writer()  # only valid inside a node
    writer({"stage": "outline", "progress": 20, "message": "outlining"})
    outline = [
        "What a graph is",
        "Why cycles matter",
        "Persistence and interrupts",
    ]
    return {"outline": outline}


def write_draft(state: DraftState) -> dict:
    writer = get_stream_writer()
    writer({"stage": "write", "progress": 60, "message": "calling the model"})
    ai = make_model().invoke(
        [HumanMessage(content=f"Write a one-paragraph intro about {state['topic']}")]
    )
    return {"messages": [ai]}


def polish(state: DraftState) -> dict:
    writer = get_stream_writer()
    writer({"stage": "polish", "progress": 100, "message": "polishing"})
    text = state["messages"][-1].text.strip()
    return {"draft": text + " (polished)"}


def build():
    g = StateGraph(DraftState)
    g.add_node("outline", make_outline)
    g.add_node("write", write_draft)
    g.add_node("polish", polish)
    g.add_edge(START, "outline")
    g.add_edge("outline", "write")
    g.add_edge("write", "polish")
    g.add_edge("polish", END)
    return g.compile()


def seed() -> dict:
    return {"topic": "LangGraph", "outline": [], "messages": [], "draft": ""}


def main() -> None:
    banner("09 - Streaming modes")

    app = build()

    note('stream_mode="values" - the complete state after each super-step.')
    for state in app.stream(seed(), stream_mode="values"):
        keys = {k: (len(v) if isinstance(v, list) else v) for k, v in state.items()}
        print(f"  {keys}")

    note('stream_mode="updates" - only what each node changed.')
    for chunk in app.stream(seed(), stream_mode="updates"):
        for node, update in chunk.items():
            print(f"  {node:<8} -> {update}")

    note(
        '''
        stream_mode="messages" - token chunks from chat models, plus metadata
        (which node produced them). This is what you pipe into a chat UI.
        '''
    )
    buffer = ""
    for chunk, metadata in app.stream(seed(), stream_mode="messages"):
        buffer += chunk.text
        print(f"  [{metadata.get('langgraph_node')}] {chunk.text!r}")
    print(f"\n  assembled: {buffer.strip()!r}")

    note(
        """
        stream_mode="custom" - your own progress events via `get_stream_writer()`.
        Useful for "step 2/5", tool names, partial documents, etc. They do not
        touch graph state.
        """
    )
    for event in app.stream(seed(), stream_mode="custom"):
        print(f"  {event}")

    note(
        """
        Multiple modes at once. Each item is a (mode, payload) tuple, so one
        connection can drive a progress bar AND update the final UI.
        """
    )
    for mode, payload in app.stream(seed(), stream_mode=["custom", "updates"]):
        if mode == "custom":
            print(f"  custom  {payload['stage']:<8} {payload['progress']:>3}%")
        else:
            print(f"  update  {list(payload.keys())}")

    note(
        """
        Async works the same way: `astream` / `ainvoke`. Use it in FastAPI and
        other async servers so a slow model call does not block the event loop.
        """
    )
    import asyncio

    async def run_async() -> None:
        async for mode, payload in app.astream(
            seed(), stream_mode=["custom", "updates"]
        ):
            if mode == "custom":
                print(f"  async custom -> {payload['stage']}")

    asyncio.run(run_async())

    note(
        """
        Practical notes:
        * `values` is easiest for debugging; `updates` is cheapest for UIs.
        * `messages` streams from ANY chat model in the graph, including inside
          subgraphs - use `subgraphs=True` to see their namespace.
        * `get_stream_writer()` is a no-op outside a streaming run, so it is safe
          to leave progress events in production nodes.
        * For token streaming in the demo agent, use `stream_mode="messages"`;
          the planner itself uses structured output, which streams at the end.
        """
    )


if __name__ == "__main__":
    main()
