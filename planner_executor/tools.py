"""Tools available to the executor node.

A LangChain tool is just a Python function with a docstring and a type
signature. That docstring is what the model sees when it decides which tool to
call, so write tools as if the docstring is the UI.
"""

from __future__ import annotations

import ast
import datetime as dt
import operator as op
from typing import Callable

from langchain_core.tools import tool

MAX_TOOL_OUTPUT = 4_000


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------
_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a tiny, safe subset of Python expressions."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("only numbers are allowed")
    if isinstance(node, ast.BinOp):
        handler = _BIN_OPS.get(type(node.op))
        if handler is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        return handler(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPS.get(type(node.op))
        if handler is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        return handler(_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly.

    Use this for any maths instead of doing mental arithmetic.
    Supports + - * / // % ** and parentheses, e.g. "(1200 * 1.18) / 3".
    """
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree)
    except Exception as exc:  # noqa: BLE001 - report the error to the model
        return f"ERROR: could not evaluate {expression!r}: {exc}"

    if value.is_integer():
        return str(int(value))
    return f"{value:.10g}"


# ---------------------------------------------------------------------------
# current_datetime
# ---------------------------------------------------------------------------
@tool
def current_datetime(timezone: str = "local") -> str:
    """Return the current date and time.

    ``timezone`` is an IANA name such as "Asia/Kolkata" or "UTC".
    Use "local" for the machine's local time.
    """
    try:
        now = dt.datetime.now(dt.timezone.utc) if timezone == "UTC" else dt.datetime.now()
        if timezone not in {"local", "UTC"}:
            from zoneinfo import ZoneInfo

            now = dt.datetime.now(ZoneInfo(timezone))
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: unknown timezone {timezone!r}: {exc}"
    return now.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web and return titles, URLs and snippets.

    Use it for facts that may have changed or that you are unsure about.
    Prefer specific queries ("LangGraph interrupt resume example") over vague ones.
    """
    max_results = max(1, min(int(max_results), 8))
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001 - network issues must not crash the graph
        return f"ERROR: web search failed for {query!r}: {exc}"

    if not results:
        return f"No results found for {query!r}."

    lines: list[str] = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "(no title)")
        url = item.get("href") or item.get("url") or ""
        body = (item.get("body") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {title}\n   {url}\n   {body[:400]}")
    return "\n".join(lines)[:MAX_TOOL_OUTPUT]


TOOLS = [calculator, current_datetime, web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def tools_manifest() -> str:
    """Human-readable tool list used inside the planner prompt."""
    return "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in TOOLS)
