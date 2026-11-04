# LangGraph — complete concept tour

This folder is a hands-on tour of LangGraph. Each numbered file is standalone,
heavily commented, and prints an explanation while it runs. File names 01–09 and
11–14 run **offline** (no API key), using a scripted fake chat model; file 10
talks to real Gemini.

```bash
# from the project root, with the venv active
python langgraph_tutorial/run_all.py          # every offline lesson
python langgraph_tutorial/10_tools_and_prebuilt_agents.py   # needs GOOGLE_API_KEY
```

| # | File | Concept |
|---|------|---------|
| 01 | `01_hello_graph.py` | `StateGraph`, nodes, edges, `START`/`END`, `compile`, `invoke` |
| 02 | `02_state_and_reducers.py` | state schemas, default replace, `operator.add`, `add_messages`, `MessagesState` |
| 03 | `03_conditional_edges.py` | routing functions, `path_map`, conditional entry, early `END` |
| 04 | `04_cycles_react_agent.py` | cycles, ReAct loop, `ToolNode`, `tools_condition`, `recursion_limit` |
| 05 | `05_persistence_and_time_travel.py` | checkpoints, threads, `get_state`, `get_state_history`, `update_state`, replay, SQLite |
| 06 | `06_human_in_the_loop.py` | `interrupt`, `Command(resume=...)`, `interrupt_before`, editing state |
| 07 | `07_subgraphs.py` | compiled graph as a node, nested state, `subgraphs=True` streaming |
| 08 | `08_parallel_and_map_reduce.py` | parallel edges, fan-out/fan-in, `Send`, map-reduce, `InvalidUpdateError` |
| 09 | `09_streaming.py` | `values`/`updates`/`messages`/`custom`/`debug`, multi-mode, `astream` |
| 10 | `10_tools_and_prebuilt_agents.py` | real Gemini: `@tool`, `bind_tools`, structured output, `create_agent` |
| 11 | `11_long_term_memory.py` | `BaseStore`, namespaces, cross-thread memory, `Runtime` context |
| 12 | `12_retries_caching_visualisation.py` | `RetryPolicy`, `CachePolicy`, timeouts, mermaid diagrams |
| 13 | `13_command_and_handoffs.py` | `Command(update, goto)`, `Send` fan-out, `Command.PARENT` |
| 14 | `14_functional_api.py` | `@entrypoint`, `@task`, parallel futures, HITL, streaming |

---

## 0. The mental model

LangGraph runs a **Pregel-style** loop:

1. All nodes scheduled for this **super-step** run (possibly in parallel).
2. Their returned updates are merged into the state by the **reducers**.
3. The edges (static and conditional) decide which nodes are scheduled next.
4. Repeat until no nodes are left → the run ends at `END`.

That is the whole engine. Everything else is built on it:

- **Persistence** = save the state between super-steps (checkpointer).
- **Human-in-the-loop** = stop the loop and resume later (interrupt).
- **Streaming** = observe the loop while it runs.
- **Time travel** = start the loop from an old saved state.
- **Map-reduce** = schedule many copies of a node in one super-step (`Send`).

Two authoring styles share this runtime:

| | Graph API | Functional API |
|---|---|---|
| Shape | `StateGraph` + nodes + edges | `@entrypoint` + `@task` |
| Best for | branching, cycles, subgraphs, reviewable diagrams | linear pipelines, quick code, parallel futures |
| State | explicit `TypedDict` with reducers | ordinary Python variables / function args |
| File | 01–09, 11–13 | 14 |

You can mix them: a graph node can call an entrypoint, and an entrypoint can
invoke a compiled graph.

---

## 01. Hello graph — `01_hello_graph.py`

**What:** the smallest useful graph. Three functions, three edges, one compiled
app.

```python
builder = StateGraph(GreetingState)
builder.add_node("normalize", normalize)   # a plain function
builder.add_edge(START, "normalize")
builder.add_edge("normalize", "greet")
builder.add_edge("greet", END)
app = builder.compile()
app.invoke({"raw_name": " ada ", "log": []})
```

**Key ideas**

- A **node** is `state -> partial update`. Return only the keys you changed.
- `START` and `END` are virtual nodes; every graph needs a path between them.
- `.compile()` validates the wiring and freezes the topology. Node functions
  can still close over config.
- `invoke` returns the **full final state**, not just the last node's output.

**Gotcha:** returning a key you did not declare in the state schema raises an
error at runtime. TypedDict keys are the contract.

---

## 02. State and reducers — `02_state_and_reducers.py`

**What:** how updates are merged. This is the concept people get wrong first.

| Declaration | Merge behaviour |
|---|---|
| `x: int` | replace |
| `x: Annotated[list[str], operator.add]` | concatenate |
| `x: Annotated[list, add_messages]` | append messages; same `id` replaces in place |
| `x: Annotated[int, operator.add]` | sum |

```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

**Why reducers matter:** in a parallel super-step two nodes may write the same
key. A reducer defines how those writes combine. Without one, LangGraph raises
`InvalidUpdateError` (demonstrated in file 08).

**`add_messages` details**

- accepts `BaseMessage`, dicts, and `(role, content)` tuples;
- deletes a message if you pass `RemoveMessage(id=...)`;
- replacing by id is how chat "editing" works without a custom API.

**Gotchas**

- Input values also go through reducers: seed accumulating keys with `[]`.
- Reducers are not called by you; they run while LangGraph merges updates.
- Keep state JSON/msgpack-friendly if you plan to persist it.

---

## 03. Conditional edges — `03_conditional_edges.py`

**What:** graph-shaped `if/else`.

```python
def route(state) -> Literal["math", "translate", "chat"]:
    return state["category"]

builder.add_conditional_edges(
    "classify", route, {"math": "math", "translate": "translate", "chat": "chat"}
)
```

- The router reads state and returns a label (or `END`, or a **list** of labels).
- `path_map` translates labels to node names and documents the graph; optional
  but strongly recommended.
- Conditional edges can start at `START`.
- A router must not mutate state. Do the work in a node, then route on it.

**Gotcha:** a router that returns a node name never mentioned in an edge or
`path_map` is a compile/runtime error. Declare every destination, including
`END`.

---

## 04. Cycles and ReAct — `04_cycles_react_agent.py`

**What:** tool-calling agents are loops.

```
assistant --tool_calls?--> tools
    ^                        |
    +------------------------+
```

```python
builder.add_conditional_edges("assistant", tools_condition,
                              {"tools": "tools", END: END})
builder.add_edge("tools", "assistant")     # the cycle
```

`tools_condition` simply checks whether the last message has `tool_calls`.
`ToolNode` executes every requested tool and appends `ToolMessage`s for you.

**Stopping safely**

- a data-driven exit (model stops asking for tools, a `done` flag, a step count);
- `{"recursion_limit": 25}` in the config — LangGraph counts super-steps and
  raises `GraphRecursionError` instead of running forever.

**Gotcha:** each tool round costs 2 super-steps (assistant + tools), so a limit
of 25 is roughly 12 rounds. Set the limit per run, not globally.

---

## 05. Persistence and time travel — `05_persistence_and_time_travel.py`

**What:** a checkpointer saves state after every super-step, keyed by
`thread_id`.

```python
app = build(InMemorySaver())
config = {"configurable": {"thread_id": "chat-1"}}
app.invoke(input, config)            # writes checkpoints
app.get_state(config)                # latest snapshot
list(app.get_state_history(config))  # all snapshots, newest first
```

You get four features from that one mechanism:

1. **Memory** — re-invoke the same thread and state carries over.
2. **Isolation** — a different thread starts clean.
3. **Inspection** — `StateSnapshot.values`, `.next`, `.metadata`, `.interrupts`.
4. **Time travel** — resume an old checkpoint, optionally after editing state:

```python
fork = app.update_state(old.config, {"value": 10}, as_node="tick")
app.invoke(None, fork)   # None == resume, do not add new input
```

`as_node` controls where the graph resumes: it pretends that node produced the
edit, so that node's outgoing edges are followed.

**Durable savers:** `SqliteSaver` (shown, survives restart), Postgres, Redis.
`durability="sync" | "async" | "exit"` trades safety for throughput.

**Gotcha:** forking a checkpoint makes it the new head of the thread. Old
checkpoints remain reachable by config, but `get_state` returns the newest head.

---

## 06. Human in the loop — `06_human_in_the_loop.py`

**Two mechanisms**

1. **Dynamic** — `interrupt(payload)` inside a node. The run pauses, the payload
   is stored in the checkpoint, and the call returns to you.
   Resume with `app.invoke(Command(resume=value), config)`.
2. **Static** — `compile(interrupt_before=["send"])` or `interrupt_after=[...]`.
   The graph stops at the node boundary; resume with `app.invoke(None, config)`.

```python
def review(state):
    decision = interrupt({"draft": state["draft"]})   # pauses here
    return {"approved": decision["approved"]}
```

**Critical detail:** on resume the node **re-executes from the top**, and
`interrupt()` returns the resume value instead of pausing. Keep code before an
interrupt side-effect free (or idempotent).

`update_state` lets a human edit the state before resuming — demonstrated with
a breakpoint before `send`.

**Requirement:** a checkpointer. Without one there is nowhere to store the pause.

---

## 07. Subgraphs — `07_subgraphs.py`

**What:** a compiled graph used as a node.

```python
parent.add_node("research", research_graph)
```

- The parent passes its **whole state** in; keys the subgraph does not declare
  are ignored on the way in and never returned.
- Keys the subgraph returns are merged into the parent through the parent's
  reducers.
- `subgraphs=True` in `stream()` exposes nested namespaces such as
  `("research:<task-id>",)`, so you can attribute events to the right level.
- A subgraph with no checkpointer inherits the parent's. Give it its own only
  when it should be a separate durable run.

**Gotcha:** do not add a compiled graph as a node of itself. Also mind key
collisions — overlapping keys are fine, but both schemas must agree on reducers.

---

## 08. Parallelism and map-reduce — `08_parallel_and_map_reduce.py`

**Static fan-out:** several edges leave the same node. All targets run in the
same super-step. Use `stream_mode="debug"` to see they share a `step`.

**Dynamic fan-out:** a router returns `Send` objects.

```python
def fan_out(state) -> list[Send]:
    return [Send("research_one", {"query": q}) for q in state["queries"]]

builder.add_conditional_edges(START, fan_out, ["research_one"])
builder.add_edge("research_one", "reduce")   # fan-in: runs once after all
```

- The `Send` payload **replaces** the branch state — include everything the
  worker needs.
- Results merge through reducers; completion order is not input order.
- Without a reducer, two writes to the same key raise `InvalidUpdateError`.

---

## 09. Streaming — `09_streaming.py`

| Mode | Yields |
|---|---|
| `values` | full state after each super-step |
| `updates` | `{node: update}` delta per node |
| `messages` | `(token_chunk, metadata)` from chat models |
| `custom` | whatever `get_stream_writer()(payload)` sends |
| `debug` | task/checkpoint events, including step numbers |

```python
for mode, payload in app.stream(input, stream_mode=["custom", "updates"]):
    ...
```

- Multiple modes → `(mode, payload)` tuples.
- Async: `astream` / `ainvoke`; use it in FastAPI and other async servers.
- `get_stream_writer()` is safe to call in normal (non-streaming) runs — it is a
  no-op there.

**Gotcha:** `messages` streams from any chat model in the graph, including
inside subgraphs when `subgraphs=True`.

---

## 10. Tools and prebuilt agents — `10_tools_and_prebuilt_agents.py`

Requires `GOOGLE_API_KEY`. Three layers:

1. **Raw protocol** — `@tool` gives the model a JSON schema; `bind_tools`
   returns an `AIMessage` with `tool_calls`. *You* execute them.
2. **Structured output** — `llm.with_structured_output(PydanticModel)` makes the
   model return validated JSON. The planner uses this.
3. **Prebuilt agents** — `langchain.agents.create_agent` (recommended in
   LangChain 1.x) or `langgraph.prebuilt.create_react_agent` (deprecated but
   still the natural choice when you want the agent as a subgraph).

**Why the split matters:** the model never executes code. Tool calls are data,
which makes them loggable, reviewable and safe to gate behind human approval.

---

## 11. Long-term memory and runtime context — `11_long_term_memory.py`

| Concept | Scope | Use for |
|---|---|---|
| checkpointer | one thread | transcript, intermediate results, resume points |
| store | all threads and runs | user profiles, learned facts, shared indexes |
| runtime context | one invocation | user id, tenant, locale, clients, secrets |

```python
def save(state, runtime: Runtime):
    runtime.store.put(("users", "u-42", "facts"), "k1", {"fact": state["fact"]})
```

- `runtime.store` is injected when the node declares a `Runtime` parameter.
- Namespaces are tuple paths, which makes scoping and deletion easy.
- `store.search(prefix, query="...")` is semantic search, but only when the
  store was created with an embedding index; otherwise use `filter={...}`.
- `StateGraph(State, context_schema=UserContext)` + `invoke(..., context=...)`
  gives typed per-request dependencies.

**Rule:** never put secrets in state — it is checkpointed and traced.

---

## 12. Retries, caching, visualisation — `12_retries_caching_visualisation.py`

```python
g.add_node("api", call_api,
           retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.05))
g.add_node("lookup", expensive, cache_policy=CachePolicy(ttl=300))
app = g.compile(cache=InMemoryCache())
```

- Default `retry_on` retries network errors and HTTP 5xx; it deliberately does
  **not** retry `ValueError`, `TypeError`, `KeyError` (those are bugs).
- Caching is keyed by node name + the node's input state. Any upstream change
  is a miss — only cache deterministic nodes.
- `timeout=` / `TimeoutPolicy` (1.2+) cancels hung calls.
- `set_node_defaults(...)` applies a policy to every node at compile time.
- `draw_mermaid()` produces a diagram; `xray=True` expands subgraphs.

---

## 13. `Command` and handoffs — `13_command_and_handoffs.py`

A node can return an update **and** the next destination:

```python
def triage(state) -> Command:
    if "refund" in state["request"]:
        return Command(update={"route": "billing"}, goto="billing")
```

- Declare `destinations=(...)` on `add_node` so the diagram is accurate.
- `goto` accepts a list, including `Send` objects for map-reduce.
- `graph=Command.PARENT` from inside a subgraph routes in the parent graph —
  perfect for escalation flows.
- `Command` cannot be mixed with a plain dict return, and cannot be returned from
  a router.
- `Command(resume=...)` is the same class in its "input" role (file 06).

---

## 14. Functional API — `14_functional_api.py`

```python
@task
def fetch(sku: str) -> float: ...

@entrypoint(checkpointer=InMemorySaver())
def quote(order: dict) -> dict:
    a = fetch(order["sku"])      # returns a future
    b = fetch(order["other"])    # scheduled too
    return {"a": a.result(), "b": b.result()}
```

- Tasks with no dependency between them run in parallel (the file proves it by
  timing two 0.2 s calls).
- Checkpointing, interrupts, retries and streaming work exactly as in the graph
  API.
- An entrypoint takes **one** input value; use a dict for multiple fields.
- Task calls must be resolved with `.result()` (it blocks).

Choose functional for pipelines; choose graph when branching/cycles/subgraphs
and a reviewable diagram matter.

---

## Cross-cutting cheatsheet

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import (Command, Send, interrupt, RetryPolicy,
                             CachePolicy, TimeoutPolicy, Overwrite)
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from langgraph.func import entrypoint, task
```

```python
config = {
    "configurable": {"thread_id": "user-1"},   # checkpointer scope
    "recursion_limit": 40,                     # super-step budget
    "tags": ["prod"],                          # tracing
}
app.invoke(input, config)
app.stream(input, config, stream_mode=["updates", "custom"], subgraphs=True)
app.get_state(config)
app.get_state_history(config)
app.update_state(config, {"k": "v"}, as_node="node_name")
app.invoke(Command(resume=value), config)
```

---

## Common mistakes

1. **Mutating state in place.** Return a new dict; LangGraph merges it.
2. **Forgetting the reducer** on a key written by parallel nodes →
   `InvalidUpdateError`.
3. **Expecting `interrupt()` without a checkpointer.** There is nowhere to store
   the pause.
4. **Side effects before `interrupt()`.** The node re-runs on resume.
5. **No exit condition in a cycle.** Always have a data-driven stop plus a
   `recursion_limit`.
6. **Storing objects that cannot be serialised.** Keep checkpoints as plain data.
7. **Using a stock `StateGraph` schema for a subgraph that needs different
   keys.** Give the subgraph its own `TypedDict`.
8. **Assuming parallel completion order.** It is nondeterministic; sort in the
   reducer or aggregator.
9. **Retrying bugs.** The default retry policy skips `ValueError`/`TypeError` on
   purpose.
10. **Putting secrets in state.** State is checkpointed and traced.

## Debugging

- `compile(debug=True)` prints every task and state update.
- `stream_mode="debug"` gives step-indexed task events.
- `stream_mode="values"` shows the whole state after every step — ideal for
  "where did this field come from?".
- `graph.get_graph().draw_mermaid()` in a PR review catches most wiring bugs.
- Set `LANGSMITH_TRACING=true` to inspect prompts, tool calls and checkpoints.

## Versions tested

- Python 3.14
- `langgraph` 1.2.19
- `langchain` 1.4.7 / `langchain-core` 1.7.3
- `langchain-google-genai` 4.4.2
- `langgraph-checkpoint-sqlite` 3.1.3

LangGraph 1.x renamed nothing important in the APIs used here, but
`create_react_agent` is deprecated in favour of `langchain.agents.create_agent`.
