"""03 - Conditional edges: routing and branching.

A conditional edge is a function that reads the state and returns the name of
the next node (or ``END``). It never changes state itself - it only decides.

    builder.add_conditional_edges(source_node, router, path_map)

``router`` can be a plain function, an async function, or a Runnable. It may
return a single label, ``END``, or a *list* of labels (parallel fan-out, see
08_parallel_and_map_reduce.py).

``path_map`` translates the router's return values into node names. It is
optional but makes the graph self-documenting and catches typos at compile time.

Run: python langgraph_tutorial/03_conditional_edges.py
Needs: nothing.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from _shared import banner, note, show, show_mermaid


class RouterState(TypedDict):
    request: str
    category: str
    answer: str


# ---------------------------------------------------------------------------
# conversation "classifier" node
# ---------------------------------------------------------------------------
def classify(state: RouterState) -> dict:
    text = state["request"].lower()
    if any(token in text for token in ("+", "-", "*", "/", "=", "sum", "times")):
        category = "math"
    elif any(token in text for token in ("translate", "in french", "in german")):
        category = "translate"
    else:
        category = "chat"
    return {"category": category}


# ---------------------------------------------------------------------------
# the router: pure decision function
# ---------------------------------------------------------------------------
Route = Literal["math", "translate", "chat"]


def route(state: RouterState) -> Route:
    """Return the *key* that ``path_map`` maps to a node name."""
    return state["category"]  # type: ignore[return-value]


# a second style: no classify node, decide straight from the request
def route_from_start(state: RouterState) -> str:
    return "math" if "=" in state["request"] else "chat"


# ---------------------------------------------------------------------------
# specialised nodes
# ---------------------------------------------------------------------------
def math_node(state: RouterState) -> dict:
    import re

    text = state["request"]
    candidates = [
        chunk.strip()
        for chunk in re.findall(r"[0-9().+\-*/% ]{3,}", text)
        if any(ch.isdigit() for ch in chunk)
    ]
    expression = max(candidates, key=len) if candidates else text
    try:
        value = eval(  # noqa: S307 - tutorial only; see tools.py for a safe version
            expression.replace("x", "*"),
            {"__builtins__": {}},
            {},
        )
        answer = f"{expression} = {value}"
    except Exception:  # noqa: BLE001
        answer = "I could not parse that arithmetic."
    return {"answer": answer}


def translate_node(state: RouterState) -> dict:
    # A real node would call a model here. Kept deterministic for the demo.
    return {"answer": f"[translated to French] {state['request']}"}


def chat_node(state: RouterState) -> dict:
    return {"answer": f"Let's talk about: {state['request']}"}


def build():
    builder = StateGraph(RouterState)

    builder.add_node("classify", classify)
    builder.add_node("math", math_node)
    builder.add_node("translate", translate_node)
    builder.add_node("chat", chat_node)

    builder.add_edge(START, "classify")

    # The three branches converge on END. A path_map (3rd argument) is optional
    # when the router already returns the node names, but it documents intent.
    builder.add_conditional_edges(
        "classify",
        route,
        {"math": "math", "translate": "translate", "chat": "chat"},
    )

    # Every branch must terminate somewhere.
    builder.add_edge("math", END)
    builder.add_edge("translate", END)
    builder.add_edge("chat", END)

    return builder.compile()


def build_with_conditional_entry():
    """A conditional edge can also be attached to START."""
    builder = StateGraph(RouterState)
    builder.add_node("math", math_node)
    builder.add_node("chat", chat_node)
    builder.add_conditional_edges(START, route_from_start, {"math": "math", "chat": "chat"})
    builder.add_edge("math", END)
    builder.add_edge("chat", END)
    return builder.compile()


def main() -> None:
    banner("03 - Conditional edges")

    app = build()
    show_mermaid(app)

    note(
        """
        The router is pure: it looks at `category` and returns a label. Only one
        branch runs. This is a graph-shaped `if/else`, but the routing logic is
        inspectable, testable and visible in the diagram.
        """
    )

    for request in (
        "What is 12 * (3 + 4)?",
        "Translate good morning to French",
        "Tell me about LangGraph",
    ):
        result = app.invoke({"request": request, "category": "", "answer": ""})
        print(f"\n  {request!r}\n    -> [{result['category']}] {result['answer']}")

    note(
        """
        Routers can also be attached to START. And a router may return END
        directly to stop the graph early - useful for "already answered" or
        "nothing to do" shortcuts.

        Design tip: keep routers dumb. They should read state, not call models
        or tools. Do the expensive work in a node, then route on the result.
        """
    )

    entry_app = build_with_conditional_entry()
    print("\n  conditional entry sample:")
    for request in ("2 + 2 = ?", "hello"):
        print(f"    {request!r} -> {entry_app.invoke({'request': request, 'category': '', 'answer': ''})['answer']!r}")

    show(
        "nodes reached (stream_mode='updates')",
        [
            list(chunk.keys())
            for chunk in app.stream(
                {"request": "5 + 5 = ?", "category": "", "answer": ""},
                stream_mode="updates",
            )
        ],
    )


if __name__ == "__main__":
    main()
