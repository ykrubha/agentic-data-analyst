"""Run the benchmark against a real LLM and write results for the README.

    python -m eval.run_eval --provider anthropic --model claude-sonnet-5-5
    python -m eval.run_eval --provider openai --model analyst-model --max-seconds 600

Writes results/benchmark.json and results/benchmark.md. Per-question traces go in
results/benchmark.json so each score can be audited.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyst_agent.agent import AgentConfig, DataAnalystAgent
from analyst_agent.llm import LLMProvider, PacedProvider, make_provider

from .questions import QUESTIONS, Question
from .scoring import gold_answer, score


class BenchmarkInterrupted(RuntimeError):
    """The provider kept failing, so the run stopped before producing a misleading score."""


def load_progress(path: Path | None) -> dict[str, dict]:
    """Finished questions from an earlier run. Provider failures are left out to be retried."""
    if path is None or not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {row["id"]: row for row in rows if row["status"] != "llm_error"}


def evaluate(
    provider: LLMProvider,
    database: str | Path,
    questions: list[Question] | None = None,
    config: AgentConfig | None = None,
    verbose: bool = True,
    progress_path: Path | None = None,
    resume: bool = False,
    max_consecutive_llm_errors: int = 3,
) -> dict:
    questions = questions if questions is not None else QUESTIONS
    agent = DataAnalystAgent(provider, database, config=config)
    done = load_progress(progress_path) if resume else {}
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        # Rewrite without earlier provider failures, so they can't linger in the results.
        progress_path.write_text("".join(json.dumps(r, default=str) + "\n" for r in done.values()))
    rows = []
    consecutive_errors = 0
    for question in questions:
        if question.id in done:
            rows.append(done[question.id])
            if verbose:
                print(f"SKIP {question.id:<6} already finished", flush=True)
            continue
        expected = gold_answer(question, database)
        result = agent.run(question.text)
        correct, reason = score(question, result, expected)
        row = {
            "id": question.id, "category": question.category, "question": question.text,
            "expected": expected, "value": result.value, "correct": correct, "reason": reason,
            "status": result.status, "steps": result.steps, "tool_errors": result.tool_errors,
            "recovered_errors": result.recovered_errors,
            "answer_recovered_from_text": result.answer_recovered_from_text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens, "seconds": result.seconds,
            "answer": result.answer, "sql": result.sql,
            "trace": [vars(event) for event in result.trace],
        }
        if result.status == "llm_error":
            consecutive_errors += 1
            if verbose:
                print(f"ERROR {question.id:<6} {(result.answer or '')[:300]}", flush=True)
            if consecutive_errors >= max_consecutive_llm_errors:
                raise BenchmarkInterrupted(
                    f"The model provider failed on {consecutive_errors} questions in a row. "
                    f"Last error: {(result.answer or '')[:300]}"
                )
            rows.append(row)
            continue
        consecutive_errors = 0
        rows.append(row)
        if progress_path is not None:
            with progress_path.open("a") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        if verbose:
            print(f"{'PASS' if correct else 'FAIL'} {question.id:<6} {reason[:90]}", flush=True)
    return summarize(rows, getattr(provider, "model", "unknown"))


def summarize(rows: list[dict], model: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)

    def stats(items: list[dict]) -> dict:
        n = len(items)
        return {
            "n": n,
            "correct": sum(r["correct"] for r in items),
            "accuracy": round(sum(r["correct"] for r in items) / n, 4),
            "mean_steps": round(sum(r["steps"] for r in items) / n, 2),
            "mean_seconds": round(sum(r["seconds"] for r in items) / n, 2),
            "mean_tokens": round(sum(r["input_tokens"] + r["output_tokens"] for r in items) / n),
        }

    errors = sum(r["tool_errors"] for r in rows)
    return {
        "model": model,
        "statuses": dict(Counter(r["status"] for r in rows)),
        "answers_recovered_from_text": sum(r["answer_recovered_from_text"] for r in rows),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": stats(rows),
        "by_category": {name: stats(items) for name, items in groups.items()},
        "tool_errors": errors,
        "recovered_errors": sum(r["recovered_errors"] for r in rows),
        "questions_with_tool_error": sum(1 for r in rows if r["tool_errors"]),
        "correct_after_tool_error": sum(1 for r in rows if r["tool_errors"] and r["correct"]),
        "rows": rows,
    }


def to_markdown(summary: dict) -> str:
    lines = [
        f"Model: `{summary['model']}`, run {summary['run_at']}.",
        "",
        "| Category | Correct | Accuracy | Mean steps | Mean seconds | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in [*summary["by_category"].items(), ("**Overall**", summary["overall"])]:
        lines.append(f"| {name} | {s['correct']}/{s['n']} | {s['accuracy']:.0%} | "
                     f"{s['mean_steps']} | {s['mean_seconds']} | {s['mean_tokens']} |")
    lines += [
        "",
        f"Tool errors: {summary['tool_errors']} across {summary['questions_with_tool_error']} questions. "
        f"{summary['correct_after_tool_error']} of those questions still ended correct.",
        "",
        f"How runs ended: {summary['statuses']}. "
        f"{summary['answers_recovered_from_text']} answers were written as JSON text and recovered, "
        "not sent as native tool calls.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--database", default="data/chinook.sqlite")
    parser.add_argument("--out", default="results")
    parser.add_argument("--only", default=None, help="Comma-separated question ids.")
    parser.add_argument("--reasoning-effort", default=None,
                        help="openai provider only. Use none to turn off thinking on local models.")
    parser.add_argument("--min-seconds-between-calls", type=float, default=0.0,
                        help="Pace requests to stay under free-tier rate limits.")
    parser.add_argument("--resume", action="store_true",
                        help="Keep finished questions from the last run and retry the rest.")
    parser.add_argument("--max-seconds", type=float, default=180.0,
                        help="Time limit per question. Raise this for slower local models.")
    args = parser.parse_args()

    questions = QUESTIONS
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in QUESTIONS if q.id in wanted]

    started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provider = make_provider(args.provider, args.model, args.reasoning_effort)
    if args.min_seconds_between_calls > 0:
        provider = PacedProvider(provider, args.min_seconds_between_calls)
    try:
        summary = evaluate(provider, args.database, questions,
                           config=AgentConfig(max_seconds=args.max_seconds),
                           progress_path=out / "benchmark_progress.jsonl", resume=args.resume)
    except BenchmarkInterrupted as err:
        print(f"\nStopped: {err}\nProgress is saved, and no score was written. "
              "Wait a while, then run the same command with --resume.")
        return 1
    errors = sum(r["status"] == "llm_error" for r in summary["rows"])
    if errors:
        print(f"\n{errors} questions hit provider errors, so no score was written. "
              "Run the same command with --resume to retry them.")
        return 1
    (out / "benchmark.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "benchmark.md").write_text(to_markdown(summary))
    print("\n" + to_markdown(summary))
    print(f"Finished in {time.monotonic() - started:.0f}s. Wrote {out}/benchmark.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
