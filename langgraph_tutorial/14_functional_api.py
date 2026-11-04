"""14 - The Functional API: @entrypoint and @task.

LangGraph has two authoring styles for the same runtime:

  GRAPH API      StateGraph + nodes + edges. Explicit, visual, best when the
                 control flow is the interesting part (branches, cycles, HITL).

  FUNCTIONAL API plain-looking Python functions decorated with @entrypoint and
                 @task. Best when the flow is simple but you still want
                 checkpointing, retries, streaming and parallelism.

`@task` marks a unit of work. Calling a task does not run it immediately - it
returns a future. Tasks with no dependency between them run in parallel when you
call `.result()`. Every completed task is checkpointed, so on resume LangGraph
skips work that already finished.

The entrypoint function's return value is the run's output. You invoke it like a
graph: `.invoke`, `.stream`, `.get_state`, `Command(resume=...)`, etc.

Run: python langgraph_tutorial/14_functional_api.py
Needs: nothing.
"""

from __future__ import annotations

import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, RetryPolicy, interrupt

from _shared import banner, note, show


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
@task
def fetch_price(sku: str) -> float:
    time.sleep(0.2)  # pretend this is a network call
    return {"a-1": 10.0, "b-2": 25.5}.get(sku, 99.0)


@task
def fetch_tax_rate(region: str) -> float:
    time.sleep(0.2)
    return {"IN": 0.18, "DE": 0.19}.get(region, 0.0)


@task(retry_policy=RetryPolicy(max_attempts=3))
def flaky_step(value: int) -> int:
    """Tasks accept the same policies as graph nodes."""
    return value + 1


# ---------------------------------------------------------------------------
# entrypoint 1 - sequential + parallel tasks
# ---------------------------------------------------------------------------
def _quote_body(order: dict) -> dict:
    # An entrypoint receives ONE input value - use a dict for several fields.
    sku = order["sku"]
    region = order.get("region", "IN")

    # Schedule both tasks FIRST, then collect: they run in parallel.
    price_future = fetch_price(sku)
    tax_future = fetch_tax_rate(region)

    price = price_future.result()
    tax = tax_future.result()

    total = flaky_step(int(price * (1 + tax))).result()
    return {"sku": sku, "region": region, "price": price, "tax": tax, "total": total}


quote = entrypoint()(_quote_body)
checkpointed_quote = entrypoint(checkpointer=InMemorySaver())(_quote_body)


# ---------------------------------------------------------------------------
# entrypoint 2 - human in the loop, exactly like a graph
# ---------------------------------------------------------------------------
@entrypoint(checkpointer=InMemorySaver())
def wire_transfer(amount: int) -> dict:
    decision = interrupt(
        {"action": "approve_transfer", "amount": amount, "question": "Approve?"}
    )
    if not decision.get("approved"):
        return {"amount": amount, "status": "rejected", "reason": decision.get("reason", "")}
    return {"amount": amount, "status": "sent"}


def main() -> None:
    banner("14 - The Functional API (@entrypoint / @task)")

    note(
        """
        Compare with the graph files: no StateGraph, no add_edge. The Python
        function IS the control flow. `@task` calls return futures; `.result()`
        blocks until the value is ready.
        """
    )

    started = time.perf_counter()
    result = quote.invoke({"sku": "b-2", "region": "DE"})
    elapsed = time.perf_counter() - started

    show("quote.invoke(...)", result)
    print(
        f"\n  elapsed: {elapsed:.2f}s - two 0.2s tasks ran in parallel "
        "(sequential would be ~0.4s)"
    )

    note(
        """
        Checkpointing: decorate the entrypoint with a checkpointer and pass a
        thread_id. Checkpoints let you inspect state, resume after an interrupt
        and replay a run. Completed `@task` calls inside a run are journaled, so
        a resumed run does not repeat work that already finished.
        """
    )

    with_checkpoint = checkpointed_quote
    out = with_checkpoint.invoke(
        {"sku": "a-1", "region": "IN"},
        {"configurable": {"thread_id": "quote-1"}},
    )
    show("checkpointed run", out)
    print(f"  state next: {with_checkpoint.get_state({'configurable': {'thread_id': 'quote-1'}}).next}")

    note(
        """
        Human in the loop is identical to the graph API: `interrupt()` pauses,
        `Command(resume=...)` continues. The same checkpoint machinery powers it.
        """
    )

    config = {"configurable": {"thread_id": "transfer-1"}}
    wire_transfer.invoke(5000, config)
    paused = wire_transfer.get_state(config)
    print(f"\n  paused at: {paused.next}")
    print(f"  interrupt: {paused.interrupts[0].value}")

    final = wire_transfer.invoke(Command(resume={"approved": True}), config)
    show("after resume", final)

    note(
        """
        Streaming works too - each task completion is an update:
        """
    )
    for chunk in quote.stream({"sku": "a-1", "region": "IN"}, stream_mode="updates"):
        print(f"  update: {chunk}")

    note(
        """
        Choosing an API:
        * Functional - linear pipelines, fan-out/fan-in with futures, quick
          prototypes that still need persistence and HITL.
        * Graph - explicit branching, cycles, subgraphs, parallel edges, and a
          diagram you can review. The planner/executor demo uses this.
        * You can mix them: an `@entrypoint` can call a compiled graph, and a
          graph node can call an entrypoint.
        """
    )


if __name__ == "__main__":
    main()
