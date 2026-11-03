"""Scripted stand-ins for an LLM, used to test the loop and harness without an API key.

These prove the plumbing works. They say nothing about how well a real model
performs, so their scores must never be reported as benchmark results.
"""

from __future__ import annotations

from analyst_agent.llm import LLMResponse, ToolCall


class ScriptedProvider:
    """Returns a fixed sequence of responses and records what it was sent."""

    model = "scripted"

    def __init__(self, turns: list[list[tuple[str, dict]] | str]):
        self.turns = list(turns)
        self.calls: list[list[dict]] = []
        self.tools_offered: list[list[str]] = []

    def complete(self, system, messages, tools) -> LLMResponse:
        self.calls.append(list(messages))
        self.tools_offered.append([t["name"] for t in tools])
        if not self.turns:
            return LLMResponse(text="out of script")
        turn = self.turns.pop(0)
        index = len(self.calls)
        if isinstance(turn, str):
            return LLMResponse(text=turn, input_tokens=10, output_tokens=5)
        calls = [ToolCall(f"call_{index}_{i}", n, a) for i, (n, a) in enumerate(turn)]
        return LLMResponse(tool_calls=calls, input_tokens=10, output_tokens=5)


def final(value, answerable=True, answer="done"):
    return [("final_answer", {"answer": answer, "value": value, "answerable": answerable})]
