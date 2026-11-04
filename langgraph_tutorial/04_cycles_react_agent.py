"""04 - Cycles: building a ReAct agent by hand.

A cycle is just an edge that points backwards. That is how tool-calling agents
work:

    assistant --(tool_calls?)--> tools --+
        ^                               |
        +-------------------------------+
        |
        +--(no tool_calls)--> END

Three things make it terminate safely:

  1. a CONDITIONAL EDGE that checks whether the model actually asked for a tool
  2. the model itself eventually answering without tool calls
  3. `recursion_limit` - LangGraph's hard cap on super-steps. Exceeding it
     raises `GraphRecursionError` instead of hanging forever.

Run: python langgraph_tutorial/04_cycles_react_agent.py
Needs: nothing (the "model" is scripted so the demo is deterministic).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from _shared import ScriptedChatModel, banner, note, show_mermaid, tool_call


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 31 C, sunny (mock data)"


@tool
def add_numbers(a: float, b: float) -> str:
    """Add two numbers exactly."""
    return str(a + b)


TOOLS = [get_weather, add_numbers]


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# the scripted "LLM" - one tool call, then a final answer
# ---------------------------------------------------------------------------
def make_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("get_weather", {"city": "Pune"}, call_id="c1"),
                    tool_call("add_numbers", {"a": 31, "b": 4}, call_id="c2"),
                ],
            ),
            AIMessage(
                content=(
                    "Pune is 31 C and sunny. In Fahrenheit that is about 87.8 F."
                )
            ),
        ]
    )


def build_agent(model: ScriptedChatModel):
    system = SystemMessage(content="You are a helpful assistant. Use tools when useful.")

    def assistant(state: AgentState) -> dict:
        # Calling the model with the full history is the "reason" step.
        reply = model.invoke([system, *state["messages"]])
        return {"messages": [reply]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    # ToolNode runs every requested tool and appends ToolMessages for you.
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "assistant")
    # tools_condition returns "tools" when the last AIMessage has tool_calls,
    # otherwise it returns END.
    builder.add_conditional_edges(
        "assistant", tools_condition, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "assistant")  # <- the cycle
    return builder.compile()


# ---------------------------------------------------------------------------
# an intentionally infinite graph, to demonstrate the recursion limit
# ---------------------------------------------------------------------------
class CounterState(TypedDict):
    count: int


def forever(state: CounterState) -> dict:
    return {"count": state["count"] + 1}


def build_infinite():
    builder = StateGraph(CounterState)
    builder.add_node("forever", forever)
    builder.add_edge(START, "forever")
    builder.add_edge("forever", "forever")
    return builder.compile()


def main() -> None:
    banner("04 - Cycles and a hand-built ReAct agent")

    model = make_model().bind_tools(TOOLS)
    app = build_agent(model)
    show_mermaid(app)

    note(
        """
        One super-step = assistant, next = tools, next = assistant again.
        Watch the message list grow: AI(tool_calls), Tool(result), Tool(result),
        AI(final answer). `add_messages` is what makes this accumulate cleanly.

        `tools_condition` is not magic: it checks
        `state["messages"][-1].tool_calls` and returns "tools" or END.
        """
    )

    result = app.invoke(
        {"messages": [HumanMessage(content="What's the weather in Pune, in F?")]},
        {"recursion_limit": 10},
    )

    print()
    for message in result["messages"]:
        label = type(message).__name__
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = ", ".join(f"{c['name']}({c['args']})" for c in message.tool_calls)
            print(f"  {label:<12} -> requests {calls}")
        else:
            text = (message.text or "").replace("\n", " ")
            print(f"  {label:<12} {text[:90]}")

    print(f"\n  tools bound to the model: {model.seen_tools}")

    note(
        """
        Now the failure mode. This graph has a cycle with no exit. LangGraph
        does NOT hang - it counts super-steps and raises GraphRecursionError
        once `recursion_limit` is hit (default 25).
        """
    )
    loop = build_infinite()
    try:
        loop.invoke({"count": 0}, {"recursion_limit": 5})
    except GraphRecursionError as exc:
        print(f"\n  GraphRecursionError: {str(exc).splitlines()[0]}")

    note(
        """
        Practical rules:
        * Every cycle needs a data-driven exit condition (no tool calls, a
          "done" flag, a step budget...), not just hope.
        * Set `recursion_limit` per run in the config. Each loop iteration is
          usually 2-3 super-steps, so a limit of 25 allows ~8 tool rounds.
        * Prebuilt alternative: `langgraph.prebuilt.create_react_agent`, or
          `langchain.agents.create_agent` in LangChain 1.x. See
          10_tools_and_prebuilt_agents.py.
        """
    )


if __name__ == "__main__":
    main()
