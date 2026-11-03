"""Check each adapter's wire format and retries against a fake HTTP server.

These validate the requests we send against the documented formats. They do not
replace running against the live APIs.
"""

import json

import httpx
import pytest

from analyst_agent import llm
from analyst_agent.llm import AnthropicProvider, LLMError, OpenAICompatibleProvider, ToolCall
from analyst_agent.tools import TOOL_SCHEMAS

HISTORY = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "checking", "tool_calls": [
        ToolCall("a", "list_tables", {}),
        ToolCall("b", "describe_table", {"table": "Genre"}),
    ]},
    {"role": "tool", "tool_call_id": "a", "name": "list_tables", "content": "[]", "is_error": False},
    {"role": "tool", "tool_call_id": "b", "name": "describe_table", "content": "nope", "is_error": True},
]


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_request_and_response_format():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        seen["headers"] = request.headers
        return httpx.Response(200, json={
            "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "id": "t1", "name": "run_sql", "input": {"query": "SELECT 1"}},
            ],
            "usage": {"input_tokens": 11, "output_tokens": 7},
        })

    provider = AnthropicProvider(model="m", api_key="k", client=client(handler))
    response = provider.complete("sys", HISTORY, TOOL_SCHEMAS)

    assert seen["system"] == "sys"
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["tools"][0]["input_schema"]["type"] == "object"
    # Both results of one assistant turn must share one user message.
    assert [m["role"] for m in seen["messages"]] == ["user", "assistant", "user"]
    results = seen["messages"][2]["content"]
    assert [b["tool_use_id"] for b in results] == ["a", "b"]
    assert "is_error" not in results[0] and results[1]["is_error"] is True
    assert response.text == "thinking"
    assert response.tool_calls[0].arguments == {"query": "SELECT 1"}
    assert (response.input_tokens, response.output_tokens) == (11, 7)


def test_openai_request_and_response_format():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "run_sql", "arguments": "{\"query\": \"SELECT 1\"}"}},
            ]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

    provider = OpenAICompatibleProvider(model="m", api_key="k", base_url="http://x/v1", client=client(handler))
    response = provider.complete("sys", HISTORY, TOOL_SCHEMAS)

    assert seen["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["tools"][0]["type"] == "function"
    assert json.loads(seen["messages"][2]["tool_calls"][1]["function"]["arguments"]) == {"table": "Genre"}
    assert seen["messages"][3]["role"] == "tool"
    assert seen["messages"][4]["content"].startswith("ERROR:")
    assert response.tool_calls[0].name == "run_sql"
    assert response.input_tokens == 3


def test_openai_reasoning_effort_is_only_sent_when_set():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    OpenAICompatibleProvider(model="m", base_url="http://x/v1", client=client(handler),
                             reasoning_effort="none").complete("s", HISTORY, [])
    OpenAICompatibleProvider(model="m", base_url="http://x/v1", client=client(handler)
                             ).complete("s", HISTORY, [])
    assert bodies[0]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in bodies[1]


def test_openai_malformed_arguments_do_not_crash():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [
            {"id": "c", "function": {"name": "run_sql", "arguments": "{broken"}}]}}]})

    provider = OpenAICompatibleProvider(model="m", api_key="k", base_url="http://x/v1", client=client(handler))
    response = provider.complete("s", HISTORY, [])
    assert "__malformed_arguments__" in response.tool_calls[0].arguments


def test_retries_overload_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(529, text="overloaded")
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "usage": {}})

    provider = AnthropicProvider(model="m", api_key="k", client=client(handler))
    assert provider.complete("s", [{"role": "user", "content": "q"}], []).text == "ok"
    assert len(attempts) == 3


def test_does_not_retry_bad_request(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(400, text="bad")

    provider = AnthropicProvider(model="m", api_key="k", client=client(handler))
    with pytest.raises(LLMError):
        provider.complete("s", [{"role": "user", "content": "q"}], [])
    assert len(attempts) == 1


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()
