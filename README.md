# LangGraph demo: planner / executor + a full concept tour

Two things live in this repo:

1. **`planner_executor/`** — a working plan-and-execute agent that uses Google
   Gemini (real API calls), tool use, checkpointing and an optional human
   approval step.
2. **`langgraph_tutorial/`** — 14 standalone, heavily commented lessons covering
   every major LangGraph concept, plus a written guide
   ([`langgraph_tutorial/README.md`](langgraph_tutorial/README.md)).

---

## Quick start

```bash
# from this folder
python -m venv .venv                 # already exists in this repo
.\.venv\Scripts\activate             # Windows
# source .venv/bin/activate          # macOS/Linux

python -m pip install -r requirements.txt

copy .env.example .env               # Windows
# cp .env.example .env               # macOS/Linux
#   -> open .env and set GOOGLE_API_KEY=...
```

Get a key at <https://aistudio.google.com/apikey>.

Run the agent:

```bash
python -m planner_executor.main "How many seconds are there in 3.5 years?"
```

Run it with a human approving the plan:

```bash
python -m planner_executor.main -i "Find the latest stable LangGraph release and summarise what changed"
```

Look at the graph without spending tokens:

```bash
python -m planner_executor.main --show-graph
```

Run the tutorial (offline, no key needed):

```bash
python langgraph_tutorial/run_all.py
```

---

## Project layout

```
.
├── .env.example              # copy to .env, paste your key
├── requirements.txt
├── planner_executor/
│   ├── config.py             # .env loading, Gemini factory, limits
│   ├── schemas.py            # Pydantic Plan / ReplanDecision
│   ├── state.py              # PlanExecuteState (TypedDict + reducer)
│   ├── prompts.py            # planner / executor / replanner prompts
│   ├── tools.py              # calculator, datetime, web search
│   ├── nodes.py              # the five nodes + routers
│   ├── graph.py              # StateGraph wiring
│   ├── main.py               # CLI: streaming, interrupts, flags
│   └── README.md             # architecture walkthrough
└── langgraph_tutorial/
    ├── README.md             # the written concept guide
    ├── _shared.py            # offline scripted chat model + print helpers
    ├── 01_hello_graph.py
    ├── 02_state_and_reducers.py
    ├── 03_conditional_edges.py
    ├── 04_cycles_react_agent.py
    ├── 05_persistence_and_time_travel.py
    ├── 06_human_in_the_loop.py
    ├── 07_subgraphs.py
    ├── 08_parallel_and_map_reduce.py
    ├── 09_streaming.py
    ├── 10_tools_and_prebuilt_agents.py   # needs GOOGLE_API_KEY
    ├── 11_long_term_memory.py
    ├── 12_retries_caching_visualisation.py
    ├── 13_command_and_handoffs.py
    ├── 14_functional_api.py
    └── run_all.py
```

---

## How the agent works

```
START -> planner -> review_plan -> executor -> ... -> replanner -> END
                         ^                              |
                         +--------- reject -------------+
```

- **planner** asks Gemini for a structured list of steps (`with_structured_output`)
- **review_plan** optionally pauses with `interrupt()` so a human can approve,
  edit or reject the plan
- **executor** does one step at a time in a small tool loop (`bind_tools`)
- **replanner** either writes the final answer or adds follow-up steps
- **finalizer** writes a best-effort answer if the revision budget runs out

Three budgets keep it bounded: `MAX_STEPS`, `MAX_REVISIONS` and the graph
`RECURSION_LIMIT`. All are configurable in `.env`.

Full details, state table and extension ideas: [`planner_executor/README.md`](planner_executor/README.md).

---

## The tutorial

Start with [`langgraph_tutorial/README.md`](langgraph_tutorial/README.md). It
explains the execution model (super-steps, reducers) and then walks through:

| Topic | Lesson |
|---|---|
| graphs, nodes, edges | 01 |
| state schemas and reducers | 02 |
| conditional routing | 03 |
| cycles, ReAct, recursion limits | 04 |
| checkpointing, threads, time travel | 05 |
| `interrupt` and human review | 06 |
| subgraphs | 07 |
| parallel branches and `Send` map-reduce | 08 |
| streaming modes (incl. token streaming) | 09 |
| tools, structured output, prebuilt agents | 10 |
| long-term memory store and runtime context | 11 |
| retries, caching, timeouts, diagrams | 12 |
| `Command` handoffs and escalations | 13 |
| functional API (`@entrypoint` / `@task`) | 14 |

Files 01–09 and 11–14 run offline with a scripted fake model, so the lessons are
fast, deterministic and free. Only lesson 10 (and the real agent) needs the API
key.

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | — | Google AI Studio key (required for real runs) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | model used by planner/executor/replanner |
| `GEMINI_TEMPERATURE` | `0.0` | sampling temperature |
| `MAX_STEPS` | `8` | maximum plan length |
| `MAX_REVISIONS` | `2` | maximum replanner calls before finalizing |
| `RECURSION_LIMIT` | `60` | LangGraph super-step cap |

`.env` is git-ignored. `.env.example` is safe to commit.

---

## Versions

Tested on Windows with Python 3.14:

- `langgraph` 1.2.19, `langchain` 1.4.7, `langchain-core` 1.7.3
- `langchain-google-genai` 4.4.2
- `langgraph-checkpoint-sqlite` 3.1.3
- `ddgs` 9.16.1

`requirements.txt` uses compatible ranges, so a fresh install may resolve to
newer patch/minor versions. The APIs used here (`StateGraph`, `interrupt`,
`Command`, `Send`, `create_agent`, `with_structured_output`) are the current
LangGraph 1.x / LangChain 1.x ones; `create_react_agent` is only shown as the
deprecated alternative.

---

## Disclaimer

The agent can call `web_search`, which sends queries to DuckDuckGo. Everything
else runs locally. Review generated plans before letting an agent act on real
data — that is exactly what the `-i` approval flag is for.
