"""10 - Real Gemini tools and prebuilt agents.

Everything before this file ran offline with a scripted model. This file talks
to Google's Generative AI API and shows the three layers you will actually use:

  1. `@tool` + `bind_tools`        - the raw tool-calling protocol
  2. `with_structured_output`      - force a Pydantic-valid JSON answer
  3. prebuilt agents               - create_agent / create_react_agent

Requires GOOGLE_API_KEY in `.env` (see .env.example). If it is missing, the
script prints instructions and exits without errors.

Run: python langgraph_tutorial/10_tools_and_prebuilt_agents.py
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from _shared import banner, note, show, show_mermaid

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
@tool
def add(a: float, b: float) -> float:
    """Add two numbers exactly. Use this instead of mental arithmetic."""
    return a + b


@tool
def now(timezone: str = "UTC") -> str:
    """Return the current date and time in an IANA timezone like 'Asia/Kolkata'."""
    if timezone == "UTC":
        return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    from zoneinfo import ZoneInfo

    return dt.datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")


TOOLS = [add, now]


class Release(BaseModel):
    """Structured answer schema - the model must match it."""

    version: str = Field(description="Version number, e.g. 0.6.0")
    highlights: list[str] = Field(description="Two or three short highlights")
    stable: bool = Field(description="Whether this is a stable release")


def get_model():
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=MODEL,
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )


def main() -> None:
    banner("10 - Tools, structured output and prebuilt agents")

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("paste-"):
        print(
            "\n  No API key found. Add GOOGLE_API_KEY to .env and run again.\n"
            "  Get one at https://aistudio.google.com/apikey\n"
        )
        return

    try:
        llm = get_model()

        # ------------------------------------------------------------------
        # 1. the tool-calling protocol
        # ------------------------------------------------------------------
        note(
            """
            `bind_tools` gives the model JSON schemas for your functions. The
            model does NOT execute anything - it returns `tool_calls`, and your
            code (or ToolNode / an agent) runs them. This separation is why tool
            calls are safe and auditable.
            """
        )
        llm_with_tools = llm.bind_tools(TOOLS)
        ai = llm_with_tools.invoke(
            "What is 1234 * 5678? Use the tool, then tell me the exact answer."
        )
        print(f"\n  tool_calls requested: {ai.tool_calls}")

        for call in ai.tool_calls:
            fn = next(t for t in TOOLS if t.name == call["name"])
            result = fn.invoke(call["args"])
            print(f"  executed {call['name']}({call['args']}) -> {result}")

        # ------------------------------------------------------------------
        # 2. structured output
        # ------------------------------------------------------------------
        note(
            """
            `with_structured_output` makes the model answer with JSON that is
            validated into `Release`. No regex, no "please answer in JSON"
            prompt fragility. This is what the planner uses for its Plan.
            """
        )
        release = llm.with_structured_output(Release).invoke(
            "Give a realistic example patch release for a hypothetical Python "
            "graph library, version 1.4.2."
        )
        show("validated object", release)
        print(f"\n  type: {type(release).__name__}")

        # ------------------------------------------------------------------
        # 3. prebuilt agent (LangChain 1.x)
        # ------------------------------------------------------------------
        note(
            """
            `langchain.agents.create_agent` builds the whole
            model -> tools -> model loop for you, including a response schema
            and middleware (summarisation, retries, HITL).
            """
        )
        from langchain.agents import create_agent

        agent = create_agent(
            model=llm,
            tools=TOOLS,
            system_prompt=(
                "You are a precise assistant. Use the tools for arithmetic and "
                "for the current time. Answer in one short sentence."
            ),
        )
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="What is 987654 + 123456, and what time is it in UTC?")
                ]
            }
        )
        print("\n  agent answer:", result["messages"][-1].text)

        # ------------------------------------------------------------------
        # 4. token streaming through the agent
        # ------------------------------------------------------------------
        note(
            """
            `stream_mode="messages"` streams tokens from the agent's model calls,
            including the intermediate tool-calling turns. Metadata tells you
            which node produced each chunk.
            """
        )
        print("\n  tokens:")
        printed = ""
        for chunk, metadata in agent.stream(
            {"messages": [HumanMessage(content="Explain in one sentence what LangGraph is.")]},
            stream_mode="messages",
        ):
            printed += chunk.text
            print(chunk.text, end="", flush=True)
        print()

        # ------------------------------------------------------------------
        # 5. the LangGraph-native equivalent
        # ------------------------------------------------------------------
        note(
            """
            `langgraph.prebuilt.create_react_agent` is the LangGraph-only
            version of the same idea. It is deprecated in favour of
            `create_agent`, but you will still see it in most LangGraph
            examples, and it is handy when you want the prebuilt loop as a
            subgraph inside a larger graph.
            """
        )
        import warnings

        from langgraph.prebuilt import create_react_agent

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            react = create_react_agent(model=llm, tools=TOOLS)
        print(f"\n  create_react_agent nodes: {list(react.get_graph().nodes)}")
        show_mermaid(react)

        note(
            """
            Choosing:
            * raw `bind_tools`       - you control the loop (see nodes.py in the
                                       planner/executor demo)
            * `create_agent`         - fastest path to a good general agent
            * `create_react_agent`   - same, as an embeddable LangGraph subgraph
            * custom StateGraph      - when the agent is one part of a bigger
                                       workflow (planner/executor, RAG, HITL)
            """
        )

    except Exception as exc:  # noqa: BLE001 - make API failures readable
        print(f"\n  API call failed: {type(exc).__name__}: {exc}")
        print(
            "  Check that GOOGLE_API_KEY is valid, the model name exists "
            f"({MODEL!r}), and that your project has quota."
        )


if __name__ == "__main__":
    main()
