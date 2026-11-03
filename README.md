# Agentic AI Data Analyst

An agent that answers business questions by writing and running its own SQL and Python. It uses native LLM tool calling: the model inspects the schema, queries the data, reads any errors, fixes its own mistakes, and returns a structured answer that a benchmark can score.

```
$ analyst-agent "Which genre generated the most revenue, and how concentrated is revenue across genres?" --trace
```

## How it works

```mermaid
flowchart LR
    Q[Question] --> L[Agent loop]
    L -->|messages + tool schemas| M[LLM provider<br/>Anthropic or OpenAI-compatible]
    M -->|tool calls| L
    L --> S[list_tables / describe_table]
    L --> R[run_sql]
    L --> P[run_python]
    R --> G1[Parser guard<br/>one SELECT only]
    G1 --> G2[Read-only SQLite<br/>+ engine authorizer]
    G2 -->|DataFrame| P
    P --> SB[Subprocess sandbox<br/>AST check, rlimits, clean env]
    L -->|errors go back to the model| M
    M -->|final_answer| A[Answer + value + assumptions + trace]
```

Design choices:

- **Tool errors are model input.** A failed query comes back as an `is_error` result, so the model corrects itself instead of the run crashing.
- **SQL results flow into Python.** `run_sql` saves a DataFrame under a name, and `run_python` receives it as a global, so full results don't have to pass through the prompt.
- **Answers are structured.** `final_answer` carries a machine-readable `value`, an `answerable` flag and stated assumptions, so scoring doesn't rely on parsing prose.
- **Small models get a way to finish.** Local models often state the answer in prose or write `final_answer` as plain JSON. The agent then offers only `final_answer`, accepts valid JSON from the text, and flags those answers in the results.
- **Every run has budgets.** Steps, tool errors, tokens and wall-clock time are all capped, and the run's status records which limit stopped it.
- **Providers are swappable.** One neutral message format is translated by small adapters, so the loop contains no provider-specific logic.

## Results

### Guardrails (measured, no LLM needed)

`python -m eval.guard_benchmark` runs 37 hand-written attacks against a copy of the database, then compares its hash afterwards.

| Check | Result |
|---|---|
| Attacks that got through both layers | 0 of 37 |
| Stopped by the parser guard | 36 |
| Missed by the parser, stopped by the engine authorizer | 1 (`pragma_table_info` used as a table function) |
| Blocked with the parser bypassed entirely | 34 of 34 |
| Legitimate gold queries wrongly rejected | 0 of 32 |
| Database file changed | No |

The attack list is my own, so this is a regression suite rather than a security audit. The one parser miss is why the second layer exists. Both the attacks and the results are in [`eval/guard_benchmark.py`](eval/guard_benchmark.py) and [`results/guard_benchmark.json`](results/guard_benchmark.json).

### Agent accuracy

**37 of 38 questions correct (97%) in one run with `gemini-3.5-flash-lite`** through its OpenAI-compatible API.

| Category | What it tests | Correct | Accuracy | Mean steps | Mean tokens |
|---|---|---:|---:|---:|---:|
| aggregation | Single-table counts, sums and filters | 8/8 | 100% | 5.25 | 7,608 |
| join | Multi-table joins, anti-joins | 10/10 | 100% | 5.9 | 10,068 |
| multi_step | CTEs, window functions, derived metrics | 8/8 | 100% | 7.12 | 15,658 |
| python | Regression, correlation, Gini, growth rates | 5/6 | 83% | 8.0 | 18,467 |
| refusal | Missing data and write requests that must be declined | 6/6 | 100% | 6.83 | 13,606 |
| **Overall** | | **37/38** | **97%** | 6.5 | 12,612 |

- **Self-correction:** tools returned 20 errors across 12 questions, and 11 of those 12 questions still ended correct.
- **The miss:** `py04` (revenue share of the top 10% of customers) used all 12 steps without submitting an answer, so it counts as wrong.
- **Answer format:** all 37 finished runs used native `final_answer` tool calls, with none recovered from text.

How to read this score:

- **It's one run on a small, hand-written benchmark.** The questions were written for this database and share its schema, so it doesn't generalize the way a public text-to-SQL benchmark such as Spider or BIRD would.
- **Development touched the benchmark's subject matter.** "Which genre generated the most revenue?" was the test question while fixing the agent, and it closely matches `join03`. The other questions weren't used for tuning, and `py04` was left as a failure.
- **Scoring rules:** expected answers are computed from gold SQL at run time, and numbers must match within 1%. Refusing an answerable question counts as wrong, and so does answering one that should be refused.

Full traces are in [`results/benchmark.json`](results/benchmark.json), and the generated summary is in [`results/benchmark.md`](results/benchmark.md).

### Run it yourself

```bash
# Gemini's free tier, with pacing to respect its rate limits
export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export OPENAI_API_KEY=...
python -m eval.run_eval --provider openai --model gemini-3.5-flash-lite --min-seconds-between-calls 5 --max-seconds 600

# Claude
export ANTHROPIC_API_KEY=...
python -m eval.run_eval --provider anthropic --model claude-sonnet-5-5
```

Finished questions are saved as the run goes, so add `--resume` to continue after an interruption. Runs stop after repeated provider errors, and no score is written while any provider errors remain, so outages can't count as wrong answers. Use one model for each score, because `--resume` doesn't check for a model change.

Local models also work through Ollama. A one-question test with `qwen3:4b` on an 8 GB Mac answered correctly but took about 5 minutes, so the full benchmark wasn't run there. The Claude path is covered by adapter tests with a fake server, but it hasn't been run against the live API.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python scripts/download_data.py    # Chinook sample database, pinned and checksummed
pytest -q                          # needs no API key

export ANTHROPIC_API_KEY=...       # or: export OPENAI_API_KEY=... and use --provider openai --model <name>
analyst-agent "What was the revenue trend by year? Save a chart." --trace
```

`--provider openai` works with any Chat Completions server that supports tools. Set `OPENAI_BASE_URL` to point at vLLM, Ollama or another compatible server.

## Security model

| Layer | Protects against |
|---|---|
| sqlglot parser allowlist | Writes, multiple statements, denied functions, system tables |
| `mode=ro` connection + SQLite authorizer | Anything the parser misses, enforced by the engine |
| Row limit and query deadline | Huge results and runaway recursive queries |
| Python AST allowlist | Non-analysis imports, `eval`/`open`, private attributes, pandas file I/O |
| Subprocess with rlimits and a clean environment | Runaway CPU or memory, and leaking API keys to generated code |

**Known limits:** the memory limit uses `RLIMIT_AS`, which Linux enforces and macOS doesn't. On a Mac, generated code is still bound by the CPU limit and timeout, but not by a memory cap, so deploy on Linux. Also, static checks on Python can be bypassed by a determined attacker, so the Python sandbox is defence in depth, not a security boundary. A multi-tenant deployment should also run it in a container or gVisor with no network.

## Project layout

```
src/analyst_agent/
  agent.py       the loop, budgets, trace
  llm.py         Anthropic and OpenAI-compatible adapters, retries
  tools.py       tool schemas and executor
  sql_guard.py   parser guard and read-only database
  sandbox.py     Python static check and subprocess runner
  cli.py         command line
eval/            questions, scoring, benchmark runner, guard benchmark
tests/           loop, adapters, guards, and the benchmark itself
```

## Data

[Chinook](https://github.com/lerocha/chinook-database) (MIT) is a sample digital music store with 11 tables and 412 invoices. It's downloaded at a pinned commit, not vendored.
