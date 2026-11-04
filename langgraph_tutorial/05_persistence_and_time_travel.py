"""05 - Persistence, threads, time travel and durable checkpoints.

A CHECKPOINTER saves the state after every super-step. That single feature gives
you four things:

  1. memory across invocations      -> pass the same `thread_id`
  2. isolated conversations        -> different `thread_id` = different history
  3. inspection                   -> `get_state`, `get_state_history`
  4. time travel / forking         -> resume from an old checkpoint, optionally
                                      after editing the state with `update_state`

For `interrupt()` (file 06) a checkpointer is mandatory - the pause is stored in
the checkpoint.

`InMemorySaver` is great for development. `SqliteSaver` (shown at the end)
survives process restarts; Postgres/Redis savers exist for production.

Run: python langgraph_tutorial/05_persistence_and_time_travel.py
Needs: nothing.
"""

from __future__ import annotations

import operator
import os
import tempfile
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from _shared import banner, note, show


class CounterState(TypedDict):
    value: int
    trail: Annotated[list[str], operator.add]


def tick(state: CounterState) -> dict:
    value = state["value"] + 1
    return {"value": value, "trail": [f"tick {value}"]}


def more(state: CounterState) -> str:
    return "tick" if state["value"] < 3 else "finish"


def finish(state: CounterState) -> dict:
    return {"trail": ["finished"]}


def build(checkpointer):
    g = StateGraph(CounterState)
    g.add_node("tick", tick)
    g.add_node("finish", finish)
    g.add_edge(START, "tick")
    g.add_conditional_edges("tick", more, {"tick": "tick", "finish": "finish"})
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


def main() -> None:
    banner("05 - Persistence, threads and time travel")

    saver = InMemorySaver()
    app = build(saver)

    config_a = {"configurable": {"thread_id": "thread-A"}}
    config_b = {"configurable": {"thread_id": "thread-B"}}

    result = app.invoke({"value": 0, "trail": []}, config_a)
    show("thread-A after run", result)

    note(
        """
        The checkpointer is keyed by `thread_id`. Invoking thread-B starts from a
        clean slate, while invoking thread-A again INHERITS the saved state.
        """
    )
    app.invoke({"trail": ["A ran again"]}, config_a)
    show("thread-A re-invoked", app.get_state(config_a).values)
    show("thread-B first run", app.invoke({"value": 0, "trail": []}, config_b))

    note(
        """
        `get_state(config)` returns a StateSnapshot: current values, which nodes
        run next, the checkpoint id, and metadata (step, source, timestamps).
        """
    )
    snapshot = app.get_state(config_a)
    print(f"\n  values        : {snapshot.values}")
    print(f"  next          : {snapshot.next}")
    print(f"  checkpoint_id : {snapshot.config['configurable']['checkpoint_id']}")
    print(f"  step          : {snapshot.metadata.get('step')}")
    print(f"  source        : {snapshot.metadata.get('source')}")

    note(
        """
        `get_state_history` is the full audit trail, newest first. Every entry is
        a checkpoint you can resume from - this is LangGraph's built-in
        time-travel / replay.
        """
    )
    history = list(app.get_state_history(config_a))
    print(f"\n  {len(history)} checkpoints for thread-A:")
    for snap in history:
        cid = snap.config["configurable"]["checkpoint_id"]
        print(
            f"    step={snap.metadata.get('step'):>2}  next={str(snap.next):<18} "
            f"value={snap.values.get('value')}  id=...{cid[-8:]}"
        )

    # --- fork an old checkpoint and change the timeline ---------------------
    note(
        """
        Time travel in two steps:
          1. take a checkpoint from the history
          2. `update_state(old_config, {"value": 10}, as_node="tick")` writes a
             new branch, then `invoke(None, fork_config)` continues from it

        Passing `None` as the input means "resume; do not add new input".
        `as_node="tick"` says "pretend the tick node just produced this update",
        so the graph follows tick's outgoing edges. Here that jumps straight to
        `finish` instead of running `tick` again.

        The fork becomes the new head of the thread. Older checkpoints are not
        deleted - you can always resume them by config again.
        """
    )

    history_b = list(app.get_state_history(config_b))
    old = next(
        snap
        for snap in history_b
        if snap.values.get("value") == 1 and "tick" in (snap.next or ())
    )

    print(f"\n  forking from checkpoint ...{old.config['configurable']['checkpoint_id'][-8:]}")
    fork_config = app.update_state(old.config, {"value": 10}, as_node="tick")
    forked = app.invoke(None, fork_config)

    show("forked branch", forked)
    print("  tick was NOT re-run: as_node='tick' resumed the graph after it.")

    note(
        """
        The fork is now the head of thread-B, but the original line is still in
        the history and can be resumed explicitly:
        """
    )
    print("\n  head of thread-B now :", app.get_state(config_b).values)
    app.invoke(None, old.config)  # resume the ORIGINAL checkpoint -> re-runs tick
    print("  after resuming old checkpoint:", app.get_state(config_b).values)

    # --- durable checkpointing with SQLite ---------------------------------
    note(
        """
        InMemorySaver dies with the process. SqliteSaver writes checkpoints to a
        file, so a second process (or a crash restart) can continue the thread.
        """
    )
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = os.path.join(tempfile.gettempdir(), "langgraph_tutorial_checkpoints.sqlite")
        if os.path.exists(db_path):
            os.remove(db_path)

        with SqliteSaver.from_conn_string(db_path) as sqlite_saver:
            db_app = build(sqlite_saver)
            db_config = {"configurable": {"thread_id": "durable"}}
            db_app.invoke({"value": 0, "trail": []}, db_config)
            print(f"\n  wrote checkpoints to {db_path}")

        # Re-open the file with a brand-new saver/graph: the state is still there.
        with SqliteSaver.from_conn_string(db_path) as reopened:
            db_app = build(reopened)
            restored = db_app.get_state({"configurable": {"thread_id": "durable"}})
            show("state restored in a new connection", restored.values)

        os.remove(db_path)
    except ImportError:
        print("\n  SqliteSaver not installed - `pip install langgraph-checkpoint-sqlite`")

    note(
        """
        Production notes:
        * `durability="sync"` (default) writes every step; "async" is faster but
          may lose the last steps in a crash; "exit" writes only at the end.
        * Keep thread ids user/session scoped, e.g. f"user:{user_id}:chat:{id}".
        * Checkpoints include pending writes, so long-running tools resume safely.
        """
    )


if __name__ == "__main__":
    main()
