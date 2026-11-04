"""12 - Production concerns: retries, timeouts, caching, visualisation.

Agents fail for boring reasons: a 502 from the model API, a rate limit, a slow
tool. LangGraph lets you attach policies PER NODE instead of wrapping every
function in try/except.

  RetryPolicy(initial_interval, backoff_factor, max_attempts, jitter, retry_on)
  CachePolicy(ttl)                  + compile(cache=...)
  timeout=...                       + TimeoutPolicy (1.2+)
  set_node_defaults(...)            apply policies to every node at once

Retries only apply to *transient* errors by default: network failures and
HTTP 5xx. ValueError/TypeError/KeyError are considered bugs and are NOT retried.

Run: python langgraph_tutorial/12_retries_caching_visualisation.py
Needs: nothing.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, RetryPolicy

from _shared import banner, note, show, show_mermaid


CALLS = {"unstable_api": 0, "expensive_lookup": 0}


# ---------------------------------------------------------------------------
# Part A - retries
# ---------------------------------------------------------------------------
class RetryState(TypedDict):
    query: str
    answer: str


def unstable_api(state: RetryState) -> dict:
    CALLS["unstable_api"] += 1
    if CALLS["unstable_api"] <= 2:
        # ConnectionError IS retried by default; ValueError would not be.
        raise ConnectionError("simulated 503 from the model API")
    return {"answer": f"answer for {state['query']!r} (attempt {CALLS['unstable_api']})"}


def build_retry_app():
    g = StateGraph(RetryState)
    g.add_node(
        "unstable_api",
        unstable_api,
        retry_policy=RetryPolicy(
            max_attempts=4,
            initial_interval=0.05,  # seconds; doubles each attempt by default
            backoff_factor=2.0,
            jitter=False,
        ),
        # TimeoutPolicy example (1.2+): cancel a hung call instead of retrying.
        # timeout=TimeoutPolicy(run_timeout=30, idle_timeout=10),
    )
    g.add_edge(START, "unstable_api")
    g.add_edge("unstable_api", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Part B - node caching
# ---------------------------------------------------------------------------
class CacheState(TypedDict):
    query: str
    normalized: str
    enriched: str


def normalize(state: CacheState) -> dict:
    # Cheap and deterministic: no need to cache, it always runs.
    return {"normalized": " ".join(state["query"].lower().split())}


def expensive_lookup(state: CacheState) -> dict:
    CALLS["expensive_lookup"] += 1
    return {"enriched": f"[enriched] {state['normalized'].upper()}"}


def build_cache_app():
    g = StateGraph(CacheState)
    g.add_node("normalize", normalize)
    g.add_node(
        "expensive_lookup",
        expensive_lookup,
        cache_policy=CachePolicy(ttl=300),  # seconds
    )
    g.add_edge(START, "normalize")
    g.add_edge("normalize", "expensive_lookup")
    g.add_edge("expensive_lookup", END)
    return g.compile(cache=InMemoryCache())


def main() -> None:
    banner("12 - Retries, caching and visualisation")

    note(
        """
        `unstable_api` fails twice, then succeeds. LangGraph retries it with
        exponential backoff WITHOUT re-running any other node. The final state
        does not contain the failures at all.

        The default `retry_on` retries ConnectionError and HTTP 5xx, but treats
        ValueError/TypeError/KeyError as bugs. Pass
        `retry_on=ValueError` (or a predicate) to override.
        """
    )

    retry_app = build_retry_app()
    show("result", retry_app.invoke({"query": "what is langgraph", "answer": ""}))
    print(f"  unstable_api was called {CALLS['unstable_api']} time(s)")

    note(
        """
        Caching is keyed by (node name + the node's input state). The second
        identical invocation reuses the cached output, so `expensive_lookup`
        runs only once across BOTH calls.

        Any change in the input - including an upstream node's output - is a
        cache miss. This is why you should only cache nodes whose output depends
        deterministically on their input.
        """
    )

    cache_app = build_cache_app()
    payload = {"query": "  What is LangGraph?  ", "normalized": "", "enriched": ""}

    cache_app.invoke(dict(payload))
    print(f"  after run #1              : expensive_lookup calls = {CALLS['expensive_lookup']}")

    cache_app.invoke(dict(payload))
    print(f"  after identical run #2    : expensive_lookup calls = {CALLS['expensive_lookup']}")

    cache_app.invoke({**payload, "query": "what is a checkpoint"})
    print(f"  after a different query    : expensive_lookup calls = {CALLS['expensive_lookup']}")

    note(
        """
        Visualisation: `get_graph().draw_mermaid()` returns text (paste it into
        any Mermaid viewer). `draw_mermaid_png()` renders an image but needs
        network access; `xray=True` expands subgraphs.

        Keep the diagram in code review: it is the fastest way to spot a missing
        edge or an accidental cycle.
        """
    )
    print(cache_app.get_graph().draw_mermaid())

    note(
        """
        Production checklist:
        * retry_policy on every node that calls a network
        * timeout=... on slow tools so a hung call cannot block a worker
        * cache_policy only on deterministic nodes (never "write to DB" or
          "send email")
        * durability="async" for throughput, "sync" when you cannot lose a step
        * recursion_limit per run; log when GraphRecursionError fires
        * error_handler=<node> (1.2+) to turn a node failure into a normal state
          update the graph can recover from
        * LangSmith tracing: set LANGSMITH_TRACING=true to see every prompt,
          tool call and checkpoint

        `set_node_defaults(retry_policy=..., timeout=...)` applies a policy to
        every node, so you are safe by default and override per node.
        """
    )

    show_mermaid(retry_app)


if __name__ == "__main__":
    main()
