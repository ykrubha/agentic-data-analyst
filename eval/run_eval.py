"""Run the benchmark against a real LLM and write results for the README.

    python -m eval.run_eval --provider anthropic --model claude-sonnet-5-5

Writes results/benchmark.json and results/benchmark.md. Per-question traces go in
results/benchmark.json so each score can be audited.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyst_agent.agent import AgentConfig, DataAnalystAgent
from analyst_agent.llm import LLMProvider, make_provider

from .questions import QUESTIONS, Question
from .scoring import gold_answer, score


def evaluate(
    provider: LLMProvider,
    database: str | Path,
    questions: list[Question] | None = None,
    config: AgentConfig | None = None,
    verbose: bool = True,
) -> dict:
    questions = questions if questions is not None else QUESTIONS
    agent = DataAnalystAgent(provider, database, config=config)
    rows = []
    for question in questions:
        expected = gold_answer(question, database)
        result = agent.run(question.text)
        correct, reason = score(question, result, expected)
        rows.append({
            "id": question.id, "category": question.category, "question": question.text,
            "expected": expected, "value": result.value, "correct": correct, "reason": reason,
            "status": result.status, "steps": result.steps, "tool_errors": result.tool_errors,
            "recovered_errors": result.recovered_errors, "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens, "seconds": result.seconds,
            "answer": result.answer, "sql": result.sql,
            "trace": [vars(event) for event in result.trace],
        })
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
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--database", default="data/chinook.sqlite")
    parser.add_argument("--out", default="results")
    parser.add_argument("--only", default=None, help="Comma-separated question ids.")
    args = parser.parse_args()

    questions = QUESTIONS
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in QUESTIONS if q.id in wanted]

    started = time.monotonic()
    summary = evaluate(make_provider(args.provider, args.model), args.database, questions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "benchmark.md").write_text(to_markdown(summary))
    print("\n" + to_markdown(summary))
    print(f"Finished in {time.monotonic() - started:.0f}s. Wrote {out}/benchmark.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
