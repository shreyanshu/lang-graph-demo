"""11 - Two kinds of memory, and runtime context.

CHECKPOINTER  = short-term memory. One conversation (thread), full state,
                automatically saved at every step. Dies with the thread.

STORE         = long-term memory. Namespaced key/value (or vector) storage
                shared by all threads and all runs. Use it for user profiles,
                learned facts, document indexes.

RUNTIME       = per-invocation dependency injection. `runtime.store`,
                `runtime.context`, `runtime.stream_writer`, `runtime.previous`.
                A node declares `runtime: Runtime` as a second parameter and
                LangGraph injects it.

This file uses `InMemoryStore`; long-running deployments swap in a Postgres or
Redis store without touching the nodes.

Run: python langgraph_tutorial/11_long_term_memory.py
Needs: nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from _shared import banner, note, show


# ---------------------------------------------------------------------------
# Part A - cross-thread memory with the store
# ---------------------------------------------------------------------------
class MemoryState(TypedDict):
    user_id: str
    action: str  # "save" or "recall"
    fact: str
    reply: str


def save_fact(state: MemoryState, runtime: Runtime) -> dict:
    store = runtime.store
    key = f"fact-{abs(hash(state['fact'])) % 10_000}"
    # namespace is a tuple - think of it as a folder path.
    store.put(("users", state["user_id"], "facts"), key, {"fact": state["fact"]})
    return {"reply": f"Remembered: {state['fact']}"}


def recall_facts(state: MemoryState, runtime: Runtime) -> dict:
    items = runtime.store.search(("users", state["user_id"], "facts"))
    if not items:
        return {"reply": "I don't know anything about you yet."}
    facts = [item.value["fact"] for item in items]
    return {"reply": "I remember:\n- " + "\n- ".join(sorted(facts))}


def route(state: MemoryState) -> str:
    return "save" if state["action"] == "save" else "recall"


def build_memory_graph(store: InMemoryStore, checkpointer):
    g = StateGraph(MemoryState)
    g.add_node("save", save_fact)
    g.add_node("recall", recall_facts)
    g.add_conditional_edges(START, route, {"save": "save", "recall": "recall"})
    g.add_edge("save", END)
    g.add_edge("recall", END)
    return g.compile(checkpointer=checkpointer, store=store)


# ---------------------------------------------------------------------------
# Part B - typed per-request context
# ---------------------------------------------------------------------------
@dataclass
class UserContext:
    user_id: str
    locale: str = "en"


class QaState(TypedDict):
    question: str
    answer: str


def answer(state: QaState, runtime: Runtime[UserContext]) -> dict:
    ctx = runtime.context
    # A real node would pass ctx.locale / ctx.user_id to a model or retriever.
    return {"answer": f"[{ctx.locale}] answer for user {ctx.user_id}: 42"}


def build_context_graph():
    g = StateGraph(QaState, context_schema=UserContext)
    g.add_node("answer", answer)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile()


def main() -> None:
    banner("11 - Long-term memory (store) and runtime context")

    note(
        """
        One store, many threads. Thread 1 writes a fact; thread 2 - a completely
        separate conversation - reads it back. Checkpointers cannot do this
        because they are scoped to a single thread.
        """
    )

    store = InMemoryStore()
    checkpointer = InMemorySaver()
    app = build_memory_graph(store, checkpointer)

    thread_1 = {"configurable": {"thread_id": "chat-1"}}
    thread_2 = {"configurable": {"thread_id": "chat-2"}}

    for fact in ("I prefer dark mode", "I live in Pune", "My timezone is IST"):
        out = app.invoke(
            {"user_id": "u-42", "action": "save", "fact": fact, "reply": ""}, thread_1
        )
        print(f"  [chat-1] {out['reply']}")

    out = app.invoke(
        {"user_id": "u-42", "action": "recall", "fact": "", "reply": ""}, thread_2
    )
    print(f"\n  [chat-2, new thread]\n  {out['reply'].replace(chr(10), chr(10) + '  ')}")

    note(
        """
        Inspect the store directly:

          store.get(namespace, key)                  -> one item
          store.search(namespace_prefix)             -> list[SearchItem]
          store.search(namespace, query="...")       -> semantic search (needs an
                                                        embedding index)
          store.list_namespaces()                    -> all namespaces

        Namespaces are hierarchical tuples, so ("users", "u-42", "facts") is
        easy to scope, filter and delete.
        """
    )

    print("\n  namespaces:")
    for namespace in store.list_namespaces():
        print(f"    {namespace}")

    items = store.search(("users", "u-42", "facts"))
    print(f"\n  {len(items)} stored facts, keys: {[item.key for item in items]}")
    sample_key = items[0].key
    show(
        "store.get(...)",
        dict(store.get(("users", "u-42", "facts"), sample_key).value),
    )

    note(
        """
        Semantic search: create the store with an embedding index, then query
        with `query="..."`. The embedder can be any LangChain Embeddings object,
        including Google's:

            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            store = InMemoryStore(index={"embed": embedder, "dims": 768,
                                         "fields": ["fact"]})

        Without an index, `query=` is ignored; keyword `filter={"fact": ...}`
        still works because filtering is exact-match on the value.
        """
    )

    filtered = store.search(("users", "u-42", "facts"), filter={"fact": "I live in Pune"})
    print(f"\n  exact-match filter found {len(filtered)} item(s)")

    note(
        """
        Runtime context is for things that should NOT be part of the checkpoint:
        the authenticated user, a tenant id, a feature flag, a database handle.
        They are passed per invocation and typed by `context_schema`.
        """
    )

    ctx_app = build_context_graph()
    result_a = ctx_app.invoke(
        {"question": "meaning of life", "answer": ""},
        context=UserContext(user_id="u-42", locale="en"),
    )
    result_b = ctx_app.invoke(
        {"question": "meaning of life", "answer": ""},
        context=UserContext(user_id="u-7", locale="hi"),
    )
    print(f"\n  {result_a['answer']}")
    print(f"  {result_b['answer']}")

    note(
        """
        Rule of thumb:
        * conversation transcript, intermediate results -> checkpointed state
        * durable user knowledge, shared documents       -> store
        * request identity, clients, secrets             -> runtime context

        Keep secrets OUT of state: it is checkpointed, logged and often exposed
        in traces.
        """
    )


if __name__ == "__main__":
    main()
