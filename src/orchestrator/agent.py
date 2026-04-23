"""The agent loop - provider-agnostic.

Runs a tool-use conversation via the configured `Provider` until it either
stops with end_turn or hits the iteration cap. Every tool call is dispatched
through `ToolExecutor`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from .config import Config
from .providers import Provider, build_provider
from .tools import TOOL_DEFINITIONS, ChangeGate, ToolExecutor

console = Console()


@dataclass
class AgentResult:
    final_text: str
    iterations: int
    findings: list[dict[str, Any]] = field(default_factory=list)
    applied_changes: list[dict[str, Any]] = field(default_factory=list)
    rejected_changes: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""


class Agent:
    def __init__(self, config: Config, auto_approve: bool = False):
        self.config = config
        self.provider: Provider = build_provider(config.provider, config.api_key, config.model)
        self.gate = ChangeGate(config.auto_apply_threshold, auto_approve=auto_approve)
        self.findings: list[dict[str, Any]] = []
        self.executor = ToolExecutor(config.repo_root, self.gate, self.findings)

    def run(self, system_prompt: str, task: str) -> AgentResult:
        messages: list[Any] = [{"role": "user", "content": task}]
        input_tokens = output_tokens = 0

        for i in range(1, self.config.max_iterations + 1):
            console.print(f"[dim]- iteration {i}/{self.config.max_iterations} ({self.provider.name}:{self.config.model}) -[/dim]")

            turn = self.provider.create_turn(
                system=system_prompt,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
            input_tokens += turn.input_tokens
            output_tokens += turn.output_tokens

            # append the assistant's turn verbatim
            messages.append(turn.raw_assistant_message)

            # echo any text to the user
            for text in turn.text_blocks:
                if text.strip():
                    console.print(f"[green]model:[/green] {text.strip()}")

            if turn.stop_reason == "end_turn":
                final = "\n".join(turn.text_blocks).strip() or "(no final message)"
                return self._finalize(final, i, input_tokens, output_tokens)

            if turn.stop_reason != "tool_use":
                return self._finalize(
                    f"Unexpected stop_reason: {turn.stop_reason}",
                    i, input_tokens, output_tokens,
                )

            # execute each tool call, collect provider-native results
            tool_results = []
            for call in turn.tool_calls:
                console.print(f"[dim]→ {call.name}({_summary(call.arguments)})[/dim]")
                try:
                    output = self.executor.dispatch(call.name, call.arguments)
                    is_error = False
                except Exception as e:  # noqa: BLE001
                    output = f"Tool execution error: {e}"
                    is_error = True
                tool_results.append(
                    self.provider.format_tool_result(
                        tool_use_id=call.id, content=output, is_error=is_error,
                    )
                )
            messages.extend(self.provider.wrap_tool_results(tool_results))

        return self._finalize(
            "Max iterations reached without end_turn.",
            self.config.max_iterations, input_tokens, output_tokens,
        )

    def _finalize(self, final_text: str, iters: int, in_tok: int, out_tok: int) -> AgentResult:
        return AgentResult(
            final_text=final_text,
            iterations=iters,
            findings=list(self.findings),
            applied_changes=list(self.gate.applied),
            rejected_changes=list(self.gate.rejected),
            input_tokens=in_tok,
            output_tokens=out_tok,
            provider=self.provider.name,
            model=self.config.model,
        )


def _summary(d: dict[str, Any], limit: int = 80) -> str:
    parts = []
    for k, v in d.items():
        s = str(v).replace("\n", "\\n")
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s}")
    joined = ", ".join(parts)
    return joined if len(joined) <= limit else joined[: limit - 3] + "..."
