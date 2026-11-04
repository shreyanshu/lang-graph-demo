"""02 - State schemas and reducers (the most important concept).

Every key in the state has a *reducer*, i.e. a rule for merging a node's update
into the current value:

    key: str                          -> new value REPLACES the old one (default)
    key: Annotated[list, operator.add] -> lists are CONCATENATED
    key: Annotated[list, add_messages] -> chat messages are appended, and a
                                          message with an existing id REPLACES it

Reducers are how LangGraph supports parallel branches: two nodes may return the
same key in the same super-step, and the reducer defines how they combine.
Without a reducer, concurrent writes to the same key are an error.

Also shown: `MessagesState` (a ready-made chat state) and partial updates.

Run: python langgraph_tutorial/02_state_and_reducers.py
Needs: nothing.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

from _shared import banner, note, show


# ---------------------------------------------------------------------------
# Part A - default reducer = replace
# ---------------------------------------------------------------------------
class ReplaceState(TypedDict):
    counter: int
    items: list[str]


def add_a(state: ReplaceState) -> dict:
    return {"counter": state["counter"] + 1, "items": ["from_a"]}


def add_b(state: ReplaceState) -> dict:
    return {"counter": state["counter"] + 1, "items": ["from_b"]}


def build_replace():
    g = StateGraph(ReplaceState)
    g.add_node("a", add_a)
    g.add_node("b", add_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Part B - explicit reducers = accumulate
# ---------------------------------------------------------------------------
class AccumulateState(TypedDict):
    counter: Annotated[int, operator.add]      # ints are summed
    items: Annotated[list[str], operator.add]  # lists are concatenated


def acc_a(state: AccumulateState) -> dict:
    return {"counter": 1, "items": ["from_a"]}


def acc_b(state: AccumulateState) -> dict:
    return {"counter": 1, "items": ["from_b"]}


def build_accumulate():
    g = StateGraph(AccumulateState)
    g.add_node("a", acc_a)
    g.add_node("b", acc_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Part C - chat state with add_messages
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def human_turn(state: ChatState) -> dict:
    return {"messages": [HumanMessage(content="Write a slogan for a coffee shop")]}


def draft(state: ChatState) -> dict:
    # Same id on purpose: add_messages treats it as an EDIT, not an append.
    return {
        "messages": [
            AIMessage(content="Coffee. It's fine.", id="slogan", name="draft")
        ]
    }


def improve(state: ChatState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="Wake up. Smell the impossible.",
                id="slogan",
                name="improved",
            )
        ]
    }


def build_chat():
    g = StateGraph(ChatState)
    g.add_node("human_turn", human_turn)
    g.add_node("draft", draft)
    g.add_node("improve", improve)
    g.add_edge(START, "human_turn")
    g.add_edge("human_turn", "draft")
    g.add_edge("draft", "improve")
    g.add_edge("improve", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Part D - MessagesState, the built-in chat state
# ---------------------------------------------------------------------------
def echo(state: MessagesState) -> dict:
    return {"messages": [AIMessage(content=f"You said: {state['messages'][-1].text}")]}


def build_messages_state():
    g = StateGraph(MessagesState)
    g.add_node("echo", echo)
    g.add_edge(START, "echo")
    g.add_edge("echo", END)
    return g.compile()


def main() -> None:
    banner("02 - State and reducers")

    note(
        """
        With NO annotation the key is replaced. `counter` still ends at 2 only
        because each node reads the latest value before returning.
        """
    )
    show("default (replace) reducers", build_replace().invoke({"counter": 0, "items": []}))

    note(
        """
        With `Annotated[int, operator.add]` both nodes' writes are summed.
        With `Annotated[list[str], operator.add]` both lists survive. The reducers
        run when LangGraph merges updates into the state - you never call them.
        """
    )
    show(
        "explicit (accumulate) reducers",
        build_accumulate().invoke({"counter": 0, "items": []}),
    )

    note(
        """
        `add_messages` is a special reducer for conversations:

          * appends new messages (id auto-assigned if missing)
          * if an id already exists, REPLACES that message in place
          * accepts message objects, dicts, and (role, content) tuples

        That is how the "improved" node edits the draft without a special API.
        """
    )
    chat = build_chat().invoke({"messages": []})
    for message in chat["messages"]:
        print(f"\n  {type(message).__name__:<12} id={message.id!r:<10} {message.text!r}")

    note(
        """
        `MessagesState` is just a TypedDict with `messages: Annotated[list,
        add_messages]`. Subclass it to add your own keys (e.g. `user_id`) instead
        of copying message plumbing everywhere.
        """
    )
    result = build_messages_state().invoke(
        {"messages": [HumanMessage(content="hello graph")]}
    )
    print("\n  " + "\n  ".join(m.pretty_repr() for m in result["messages"]))

    note(
        """
        Advanced:
        * `langgraph.types.Overwrite(value)` bypasses a reducer for one update.
        * Input values go through the reducer too, so start with [] for
          accumulating keys (or pass a non-empty list to seed history).
        * Keep state JSON-serialisable if you plan to checkpoint it.
        """
    )


if __name__ == "__main__":
    main()
