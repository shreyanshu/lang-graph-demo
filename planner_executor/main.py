"""Command-line entry point for the planner/executor demo.

Examples
--------
    python -m planner_executor.main "How many seconds are there in 3.5 years?"
    python -m planner_executor.main -i "Find the latest LangGraph release"
    python -m planner_executor.main --show-graph
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

from .config import MissingAPIKeyError, SETTINGS, has_api_key
from .graph import build_graph, initial_state

LINE = "=" * 72
THIN = "-" * 72


# ---------------------------------------------------------------------------
# pretty output
# ---------------------------------------------------------------------------
def _wrap(text: str, width: int = 96) -> str:
    import textwrap

    return "\n".join(
        textwrap.fill(line, width=width) if line.strip() else ""
        for line in text.splitlines()
    )


def print_banner(task: str, thread: str) -> None:
    print(LINE)
    print("  LangGraph planner / executor demo  -  Google Gemini")
    print(LINE)
    print(f"  model  : {SETTINGS.model}")
    print(f"  thread : {thread}")
    print(f"  task   : {task}")
    print(LINE)


def print_plan(plan: list[dict]) -> None:
    print("\n  PLAN")
    for step in plan:
        tools = f"   [tools: {', '.join(step['tools'])}]" if step.get("tools") else ""
        print(f"   {step['id']}. {step['description']}{tools}")
    print()


def print_result(response: str) -> None:
    print("\n" + LINE)
    print("  FINAL ANSWER")
    print(LINE)
    print(_wrap(response))
    print(LINE)


# ---------------------------------------------------------------------------
# streaming renderer
# ---------------------------------------------------------------------------
def render_custom(data: dict) -> None:
    status = data.get("status", "")
    node = data.get("node", "?")
    print(f"   ... {node:<10} {status}")
    if data.get("assumptions"):
        for assumption in data["assumptions"]:
            print(f"       assumption: {assumption}")
    if data.get("reasoning"):
        print(f"       reasoning : {data['reasoning']}")


def render_update(data: dict) -> None:
    for node, update in data.items():
        if node == "__interrupt__" or not isinstance(update, dict):
            continue
        if node == "planner" and update.get("plan"):
            print_plan(update["plan"])
        elif node == "executor" and update.get("past_steps"):
            record = update["past_steps"][0]
            result = record.get("result", "")
            if len(result) > 700:
                result = result[:700] + " ..."
            print(f"\n  -> result of: {record.get('step', '')}")
            print("     " + _wrap(result, width=88).replace("\n", "\n     "))


def ask_for_review(payload: dict) -> dict:
    print("\n" + THIN)
    print("  HUMAN REVIEW REQUESTED")
    print(THIN)
    print_plan(payload.get("plan", []))
    answer = input("  Approve this plan? [Y/n] ").strip().lower()
    if answer in {"", "y", "yes", "ok", "approve"}:
        return {"approved": True, "feedback": ""}
    feedback = input("  What should change? ").strip()
    return {"approved": False, "feedback": feedback or "Please produce a better plan."}


# ---------------------------------------------------------------------------
# run loop
# ---------------------------------------------------------------------------
def _make_checkpointer(db_path: str | None):
    """In-memory by default; durable SQLite when --db is given."""
    if not db_path:
        return InMemorySaver(), None

    import contextlib

    from langgraph.checkpoint.sqlite import SqliteSaver

    stack = contextlib.ExitStack()
    saver = stack.enter_context(SqliteSaver.from_conn_string(db_path))
    return saver, stack


def run(
    task: str,
    *,
    thread: str,
    interactive: bool,
    recursion_limit: int,
    db: str | None = None,
) -> int:
    checkpointer, stack = _make_checkpointer(db)
    try:
        return _run(task, checkpointer, thread, interactive, recursion_limit)
    finally:
        if stack is not None:
            stack.close()


def _run(
    task: str,
    checkpointer,
    thread: str,
    interactive: bool,
    recursion_limit: int,
) -> int:
    graph = build_graph(checkpointer=checkpointer)

    config = {
        "configurable": {"thread_id": thread},
        "recursion_limit": recursion_limit,
    }

    payload = initial_state(task, auto_approve=not interactive)
    started = time.perf_counter()

    # A run may pause several times (one per interrupt), so loop until the graph
    # finishes without asking for input.
    for _ in range(20):
        pending: Interrupt | None = None

        for mode, chunk in graph.stream(
            payload, config, stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                render_custom(chunk)
            else:
                for key in chunk:
                    if key == "__interrupt__":
                        pending = chunk[key][0]
                render_update(chunk)

        if pending is None:
            break

        payload = Command(resume=ask_for_review(pending.value))

    final = graph.get_state(config).values
    elapsed = time.perf_counter() - started

    if final.get("past_steps"):
        print(f"\n  completed {len(final['past_steps'])} step(s) in {elapsed:.1f}s")

    print_result(final.get("response") or "(the graph finished without a response)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan-and-execute agent (LangGraph + Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="What the agent should do. Omit it to be prompted.",
    )
    parser.add_argument(
        "-t",
        "--thread",
        default=None,
        help="Checkpoint thread id (reuse it to continue a previous run).",
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help=(
            "Persist checkpoints to a SQLite file instead of memory. "
            "Reuse --thread across runs to resume or inspect a conversation."
        ),
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Pause after planning so you can approve or reject the plan.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=SETTINGS.recursion_limit,
        help=f"Maximum graph super-steps (default {SETTINGS.recursion_limit}).",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Print the graph as Mermaid text and exit.",
    )
    args = parser.parse_args(argv)

    if args.show_graph:
        print(build_graph().get_graph().draw_mermaid())
        return 0

    if not has_api_key():
        print(
            "No API key configured.\n\n"
            "  copy .env.example .env      (Windows)\n"
            "  cp .env.example .env        (macOS/Linux)\n\n"
            "Then put your key from https://aistudio.google.com/apikey into\n"
            "GOOGLE_API_KEY=... inside .env\n",
            file=sys.stderr,
        )
        return 2

    task = " ".join(args.task).strip()
    if not task:
        try:
            task = input("What should the agent do? ").strip()
        except EOFError:
            task = ""
    if not task:
        parser.error("a task is required")

    thread = args.thread or f"run-{uuid.uuid4().hex[:8]}"
    print_banner(task, thread)

    try:
        return run(
            task,
            thread=thread,
            interactive=args.interactive,
            recursion_limit=args.recursion_limit,
            db=args.db,
        )
    except MissingAPIKeyError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
