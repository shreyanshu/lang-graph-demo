# Planner → Executor → Replanner

A plan-and-execute agent built with LangGraph and Google Gemini. Unlike a plain
ReAct loop, it separates **deciding what to do** from **doing it**:

- the **planner** sees the big picture and writes an ordered plan,
- the **executor** does one step at a time with tools,
- the **replanner** checks the results and either finishes or adds follow-ups.

This pattern is easier to debug than a free-running agent: every step is an
explicit state field, and the final answer is grounded in recorded results.

---

## Quick start

```bash
# 1. install (from the project root)
python -m pip install -r requirements.txt

# 2. create .env and paste your key
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux

# 3. run
python -m planner_executor.main "How many seconds are there in 3.5 years?"
```

No key yet? You can still inspect the graph:

```bash
python -m planner_executor.main --show-graph
```

---

## Graph

```
              +----------------------------------+
              |                                  |
           START                                 |
              |                                  |
           planner <--------- reject --------- review_plan
              |                                  ^
              +----------- approve --------------+
              |
              v
        +-> executor --(steps remain)---+
        |                               |
        +-------------------------------+
              |
      (plan empty)
              v
          replanner --(complete)----------------> END
              |
        (more steps)
              |
              +--> executor

          replanner --(budget spent)--> finalizer --> END

      executor --(plan empty, budget spent)--> finalizer --> END
```

Mermaid source: `python -m planner_executor.main --show-graph`.

---

## State

| Key | Type | Written by | Purpose |
|---|---|---|---|
| `task` | `str` | caller | the original request |
| `plan` | `list[dict]` | planner / replanner / executor | remaining steps (executor pops the front) |
| `past_steps` | `Annotated[list, operator.add]` | executor | completed work; the reducer appends |
| `plan_feedback` | `str` | review node | rejection feedback; non-empty routes back to the planner |
| `auto_approve` | `bool` | caller | skips the human review node |
| `response` | `str` | replanner / finalizer | final answer; non-empty ends the graph |
| `revision_count` | `int` | replanner | replanning rounds, used as a budget |

Steps are stored as plain dicts (`id`, `description`, `tools`) rather than
Pydantic objects, so any checkpointer can serialise them.

---

## Nodes

### `planner` — `nodes.py`

Gemini is called with `with_structured_output(Plan)`, so the reply is validated
into a Pydantic `Plan` object. Steps are re-numbered, capped at `MAX_STEPS`, and
any hallucinated tool names are dropped.

Returning a validated plan instead of parsing text is what makes the rest of the
graph reliable.

### `review_plan` — optional human gate

```python
if state.get("auto_approve"):
    return {}
decision = interrupt({"plan": state["plan"], "task": state["task"]})
```

The node pauses a checkpointed run. `main.py` loops: it prints the plan, reads a
yes/no plus feedback, and resumes with `Command(resume={"approved": ..., ...})`.
A rejection sets `plan_feedback` and routes back to the planner.

Because the node has no side effects before `interrupt()`, re-execution on
resume is safe.

### `executor` — a mini ReAct loop

It takes the first remaining step and runs an internal tool loop (up to 5
rounds):

1. call Gemini with `bind_tools`;
2. if the reply has `tool_calls`, execute each one and append a `ToolMessage`;
3. repeat until the model answers without tool calls.

The result is capped at 2 500 characters and appended to `past_steps`; the
executed step is popped from `plan`.

Each step sees the original task plus a truncated history of earlier results,
so the executor has continuity without carrying the full transcript.

### `replanner`

Also structured output (`ReplanDecision`). It receives the task, **all**
completed steps and any unexecuted steps, and must answer one of:

- `is_complete=True` + `final_answer` → `response` is set and the graph ends;
- `is_complete=False` + `next_steps` → the plan is replaced by the follow-ups.

If it claims "not complete" but proposes nothing, the graph finishes with a
deterministic summary instead of looping.

### `finalizer`

Runs only when the revision budget is spent. It writes the best answer possible
from `past_steps` and is explicit about what could not be verified. If even that
API call fails, a deterministic fallback joins the recorded results.

---

## Routers

| Router | Decision |
|---|---|
| `route_after_review` | `plan_feedback` set → `planner`; otherwise → `executor` |
| `route_after_executor` | steps remain → `executor`; revision budget spent → `finalizer`; otherwise → `replanner` |
| `route_after_replanner` | `response` set → `END`; plan present → `executor`; otherwise → `finalizer` |

---

## Budgets and safety

Three independent limits stop runaway loops:

| Setting | Default | Effect |
|---|---|---|
| `MAX_STEPS` | 8 | plan length cap |
| `MAX_REVISIONS` | 2 | maximum number of replanner calls |
| `RECURSION_LIMIT` | 60 | LangGraph super-step cap (`--recursion-limit`) |

`MAX_REVISIONS` bounds the plan/execute cycle; `RECURSION_LIMIT` is the last
line of defence and raises `GraphRecursionError` if a bug creates an unexpected
cycle.

---

## Tools — `tools.py`

| Tool | Purpose |
|---|---|
| `calculator` | safe AST-based arithmetic (`+ - * / // % **`, parentheses) |
| `current_datetime` | current time, optionally in an IANA timezone |
| `web_search` | DuckDuckGo results via `ddgs` (title, URL, snippet) |

Tools are ordinary `@tool` functions. The docstring is the model-facing
description, so it is written as instructions. Tool failures return an
`ERROR: ...` string rather than raising, so one bad call cannot kill a step.

To add a tool, append it to `TOOLS` in `tools.py` — the planner prompt and the
executor's tool binding both read from that list.

---

## CLI

```bash
python -m planner_executor.main "task"
python -m planner_executor.main -i "task"           # approve/reject the plan
python -m planner_executor.main -t my-thread "task" # reuse a thread id
python -m planner_executor.main --db planner.sqlite -t chat-1 "task"
python -m planner_executor.main --show-graph
```

Options: `--thread/-t`, `--db`, `--interactive/-i`, `--recursion-limit`,
`--show-graph`.

`--db` switches the checkpointer from `InMemorySaver` to `SqliteSaver`, so
threads survive process restarts. The CLI streams `updates` + `custom` events so
you see the plan, each step's result, tool calls and replanning as they happen.

---

## What this demo teaches (and where each idea lives)

| LangGraph concept | Where |
|---|---|
| `StateGraph`, nodes, static edges | `graph.py` |
| TypedDict state + `operator.add` reducer | `state.py` |
| Conditional edges / routers | `nodes.py` (bottom) |
| Cycle with a data-driven exit | `executor` ↔ `replanner` |
| Structured output | `planner_node`, `replanner_node` |
| Checkpointing (`InMemorySaver`) | `main.py` |
| `interrupt` + `Command(resume=...)` | `review_plan_node`, `main.py` |
| Custom streaming (`get_stream_writer`) | `_emit` in `nodes.py` |
| Tool calling (`bind_tools`, `ToolMessage`) | `executor_node` |
| Budgets / recursion limit | `config.py`, `route_after_replanner` |

---

## Extending it

**Persist conversations.** Use the built-in flag:

```bash
python -m planner_executor.main --db planner.sqlite -t chat-1 "task"
python -m planner_executor.main --db planner.sqlite -t chat-1 "follow-up"
```

Under the hood that is `SqliteSaver.from_conn_string(path)`. You can also do it
programmatically:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("planner.sqlite") as saver:
    graph = build_graph(checkpointer=saver)
```

Reusing a thread accumulates `past_steps` (the reducer is append-only), which is
what you want for a continuing conversation.

**Swap the model.** Change `GEMINI_MODEL` in `.env`. Planning benefits from a
stronger model; execution is fine with a fast one — point `get_llm()` at
different models per node if you want that split.

**Add long-term memory.** `build_graph(..., store=InMemoryStore())` and make the
planner read user preferences from `Runtime.store` (see
`langgraph_tutorial/11_long_term_memory.py`).

**Parallelise steps.** Independent steps can be fanned out with `Send`; the
`past_steps` reducer is already append-only, so it merges cleanly. See
`langgraph_tutorial/08_parallel_and_map_reduce.py`.

**Add an approval step per action.** Call `interrupt()` inside `executor_node`
before risky tool calls (email, purchases). Keep side effects after the
interrupt.

---

## Cost and latency

A typical run is:

- 1 planner call (structured output),
- 1–5 executor calls per step (each may include tool calls),
- 1 replanner call per round,
- optionally 1 finalizer call.

With `gemini-2.5-flash` this is usually a handful of cheap calls. Streaming
makes the latency visible; `MAX_STEPS` and `MAX_REVISIONS` bound the worst case.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No Google API key found` | create `.env` from `.env.example`, set `GOOGLE_API_KEY` |
| `404 model not found` | the model name is wrong for your key; set `GEMINI_MODEL` |
| `429` / quota errors | rate limit or free-tier quota; wait or enable billing |
| `GraphRecursionError` | raise `RECURSION_LIMIT`, or reduce `MAX_STEPS` / `MAX_REVISIONS` |
| Web search returns `ERROR: ...` | no network, or `ddgs` not installed |
| Planner keeps adding steps | lower `MAX_REVISIONS`; the replanner prompt is tuned to finish, but budgets are the real guard |
| Empty final answer | the replanner returned no text; the finalizer/deterministic fallback handles it |
