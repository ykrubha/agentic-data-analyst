"""Provider adapters for native tool calling.

The agent speaks one neutral message format. Each adapter translates it to a
provider's wire format, so the loop has no provider-specific logic.

Neutral messages:
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def _post_with_retry(
    client: httpx.Client, url: str, headers: dict, body: dict,
    max_retries: int = 4, base_delay_s: float = 1.5,
) -> dict:
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, headers=headers, json=body)
        except httpx.TransportError as err:
            last_error = f"{type(err).__name__}: {err}"
        else:
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:400]}"
            if response.status_code not in RETRYABLE_STATUS:
                raise LLMError(last_error)
        if attempt < max_retries:
            time.sleep(base_delay_s * 2**attempt)
    raise LLMError(f"Gave up after {max_retries + 1} attempts. Last error: {last_error}")


class AnthropicProvider:
    """Claude Messages API with native tool use."""

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 2048,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("Set ANTHROPIC_API_KEY to use the anthropic provider.")
        self.url = base_url.rstrip("/") + "/v1/messages"
        self.max_tokens = max_tokens
        self.client = client or httpx.Client(timeout=120)

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._to_wire(messages),
            "tools": [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        data = _post_with_retry(self.client, self.url, headers, body)
        result = LLMResponse(
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
        )
        for block in data.get("content", []):
            if block["type"] == "text":
                result.text += block["text"]
            elif block["type"] == "tool_use":
                result.tool_calls.append(ToolCall(block["id"], block["name"], block["input"]))
        return result

    @staticmethod
    def _to_wire(messages: list[dict]) -> list[dict]:
        wire: list[dict] = []
        for message in messages:
            if message["role"] == "user":
                wire.append({"role": "user", "content": message["content"]})
            elif message["role"] == "assistant":
                blocks: list[dict] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": call.id, "name": call.name,
                                   "input": call.arguments})
                # The API rejects empty assistant content.
                wire.append({"role": "assistant", "content": blocks or "(no output)"})
            else:
                block = {"type": "tool_result", "tool_use_id": message["tool_call_id"],
                         "content": message["content"]}
                if message.get("is_error"):
                    block["is_error"] = True
                # All results of one assistant turn belong in a single user message.
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
        return wire


class OpenAICompatibleProvider:
    """Any Chat Completions server with tool calling: OpenAI, vLLM, Ollama and others."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.url = base.rstrip("/") + "/chat/completions"
        self.max_tokens = max_tokens
        self.client = client or httpx.Client(timeout=120)

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, *self._to_wire(messages)],
            "tools": [{"type": "function", "function": t} for t in tools],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post_with_retry(self.client, self.url, headers, body)
        message = data["choices"][0]["message"]
        result = LLMResponse(
            text=message.get("content") or "",
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
        )
        for call in message.get("tool_calls") or []:
            raw = call["function"].get("arguments") or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                # Pass malformed JSON to the executor, which reports it back to the model.
                arguments = {"__malformed_arguments__": raw}
            result.tool_calls.append(ToolCall(call["id"], call["function"]["name"], arguments))
        return result

    @staticmethod
    def _to_wire(messages: list[dict]) -> list[dict]:
        wire: list[dict] = []
        for message in messages:
            if message["role"] == "user":
                wire.append({"role": "user", "content": message["content"]})
            elif message["role"] == "assistant":
                entry: dict = {"role": "assistant", "content": message.get("content") or None}
                if message.get("tool_calls"):
                    entry["tool_calls"] = [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in message["tool_calls"]
                    ]
                wire.append(entry)
            else:
                content = message["content"]
                if message.get("is_error"):
                    content = f"ERROR: {content}"
                wire.append({"role": "tool", "tool_call_id": message["tool_call_id"],
                             "content": content})
        return wire


def make_provider(name: str, model: str | None = None) -> LLMProvider:
    if name == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-5-5")
    if name == "openai":
        if not model:
            raise LLMError("--model is required for the openai provider.")
        return OpenAICompatibleProvider(model=model)
    raise LLMError(f"Unknown provider {name!r}. Use anthropic or openai.")
