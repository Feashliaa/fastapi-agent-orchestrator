"""LLM provider abstraction.

The agent loop talks to a `Provider` which exposes two operations:
  - `create_turn(system, messages, tools)` - send a conversation turn, get back
    a normalized `TurnResponse` with text blocks, tool calls, and a stop reason.
  - `format_tool_result(tool_use_id, content, is_error)` - build the
    provider-specific tool-result message that gets appended to history.

Each provider (Anthropic, OpenAI) implements its own wire format internally but
returns the same shape to the agent. Adding Gemini or a local model is a matter
of writing one more subclass.
"""

from __future__ import annotations

import json
import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TurnResponse:
    text_blocks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use" | "other"
    raw_assistant_message: Any = None  # provider-specific; agent appends verbatim
    input_tokens: int = 0
    output_tokens: int = 0


class Provider(ABC):
    name: str

    @abstractmethod
    def create_turn(
        self,
        *,
        system: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> TurnResponse: ...

    @abstractmethod
    def format_tool_result(
        self,
        *,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
    ) -> Any:
        """Return one tool-result entry in provider-native format.

        For Anthropic this is a content block; for OpenAI it's a full message.
        The agent collects these into the shape each provider expects.
        """
        ...

    @abstractmethod
    def wrap_tool_results(self, results: list[Any]) -> list[Any]:
        """Wrap a batch of tool results into message(s) to append.

        - Anthropic: one user message with all results as content blocks.
        - OpenAI: one standalone message per tool result.
        """
        ...
        
    def _with_retry(self, call, max_attempts: int = 4):
        """Retry on rate-limit errors with exponential backoff, respecting server hints."""
        for attempt in range(max_attempts):
            try:
                return call()
            except Exception as e:
                msg = str(e)
                is_rate_limit = "rate_limit" in msg.lower() or "429" in msg or "too_many_requests" in msg.lower()
                if not is_rate_limit or attempt == max_attempts - 1:
                    raise
                # try to parse "try again in Xs" from the error, else exponential backoff
                match = re.search(r"try again in ([\d.]+)s", msg)
                wait = float(match.group(1)) + 1 if match else 2 ** (attempt + 2)
                print(f"  [rate limited; waiting {wait:.1f}s before retry {attempt + 2}/{max_attempts}]")
                time.sleep(wait)
        raise RuntimeError("unreachable")


# ------- Anthropic --------


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int = 8192):
        from anthropic import (
            Anthropic,
        )  # lazy import so openai-only users don't need it

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def create_turn(self, *, system, messages, tools):
        response = self._with_retry(lambda: self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
        ))

        text_blocks: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                if block.text.strip():
                    text_blocks.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        stop = "end_turn"
        if response.stop_reason == "tool_use":
            stop = "tool_use"
        elif response.stop_reason != "end_turn":
            stop = "other"

        return TurnResponse(
            text_blocks=text_blocks,
            tool_calls=tool_calls,
            stop_reason=stop,
            raw_assistant_message={"role": "assistant", "content": response.content},
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def format_tool_result(self, *, tool_use_id, content, is_error=False):
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        }

    def wrap_tool_results(self, results):
        # Anthropic expects a single user message carrying all tool_result blocks.
        return [{"role": "user", "content": results}]


# ------- OpenAI --------


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, api_key: str, model: str, max_tokens: int = 8192):
        from openai import OpenAI  # lazy import

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def _to_openai_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate our Anthropic-flavoured tool schema to OpenAI's function-calling format."""
        converted = []
        for t in tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
            )
        return converted

    def create_turn(self, *, system, messages, tools):
        # OpenAI wants system as the first message, not a separate param.
        full_messages = [{"role": "system", "content": system}] + messages

        response = self._with_retry(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,  # type: ignore[arg-type]
            tools=self._to_openai_tools(tools),  # type: ignore[arg-type]
            max_completion_tokens=self.max_tokens,
        ))
        choice = response.choices[0]
        msg = choice.message

        text_blocks: list[str] = []
        if msg.content:
            text_blocks.append(msg.content)

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                # Newer SDKs include a union type for custom tools; skip anything non-function.
                if tc.type != "function":
                    continue
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        finish = choice.finish_reason
        if finish == "tool_calls":
            stop = "tool_use"
        elif finish == "stop":
            stop = "end_turn"
        else:
            stop = "other"

        # Reconstruct the assistant message for history. We must echo the
        # model's message verbatim including tool_calls, or OpenAI will reject
        # the next turn because tool results won't have a matching call.
        raw_assistant = {
            "role": "assistant",
            "content": msg.content,
        }
        if msg.tool_calls:
            raw_assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
                if tc.type == "function"
            ]

        usage = response.usage
        return TurnResponse(
            text_blocks=text_blocks,
            tool_calls=tool_calls,
            stop_reason=stop,
            raw_assistant_message=raw_assistant,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    def format_tool_result(self, *, tool_use_id, content, is_error=False):
        # OpenAI's tool-result is a standalone message, not a content block.
        # Errors are surfaced in the content string; there's no dedicated flag.
        body = content if not is_error else f"[ERROR] {content}"
        return {
            "role": "tool",
            "tool_call_id": tool_use_id,
            "content": body,
        }

    def wrap_tool_results(self, results):
        # Each tool result is already a complete message.
        return results
    

# ------- Factory --------


def build_provider(kind: str, api_key: str, model: str) -> Provider:
    kind = kind.lower()
    if kind == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    if kind == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    raise ValueError(f"Unknown provider: {kind!r}. Use 'anthropic' or 'openai'.")
