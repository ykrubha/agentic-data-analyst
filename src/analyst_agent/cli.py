"""Command line: analyst-agent "Which country generates the most revenue?" """

from __future__ import annotations

import argparse
import json
import sys

from .agent import DataAnalystAgent, format_trace
from .llm import LLMError, make_provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a data question in plain English.")
    parser.add_argument("question")
    parser.add_argument("--database", default="data/chinook.sqlite")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--trace", action="store_true", help="Print every tool call.")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args = parser.parse_args(argv)

    try:
        provider = make_provider(args.provider, args.model)
    except LLMError as err:
        print(err, file=sys.stderr)
        return 2

    result = DataAnalystAgent(provider, args.database, output_dir=args.output_dir).run(args.question)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.status == "answered" else 1
    if args.trace:
        print(format_trace(result), "\n")
    print(result.answer or f"No answer (status: {result.status}).")
    for assumption in result.assumptions:
        print(f"  Assumption: {assumption}")
    for chart in result.charts:
        print(f"  Chart: {chart}")
    print(f"\n{result.steps} steps, {result.tool_errors} tool errors, "
          f"{result.input_tokens + result.output_tokens} tokens, {result.seconds:.1f}s")
    return 0 if result.status == "answered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
