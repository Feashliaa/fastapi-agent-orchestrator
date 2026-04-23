"""Tools the agent can call during its run.

Each tool has:
  - an Anthropic tool schema (shipped to the API)
  - a local executor that actually performs the action

The executors are sandboxed to `repo_root` - no traversing above it.
Destructive actions (write_file, apply_patch) route through `ChangeGate`
which enforces the confidence threshold / human-in-the-loop policy.
"""
from __future__ import annotations

import difflib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.prompt import Confirm
from rich.syntax import Syntax

console = Console()


# ---------- Tool schemas (sent to the Anthropic API) ----------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_directory",
        "description": "List files and subdirectories at a path relative to the repo root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path. Use '.' for root."}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the full contents of a file relative to the repo root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Search the codebase for a regex pattern. Returns matching lines with file:line context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Optional subdirectory to limit the search."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or OVERWRITE a file. The 'content' you provide REPLACES the "
            "entire file — there is no partial edit, no diff, no patch. "
            "To modify an existing file you MUST: "
            "(1) call read_file first to get the full current content, "
            "(2) make your changes to that full content locally, "
            "(3) call write_file with the COMPLETE modified file including all "
            "imports, module-level code, and every function/class that should remain. "
            "Omitting any existing code will DELETE it. "
            "MUST include a 'confidence' score in [0,1] and a brief 'reason'. "
            "Below the auto-apply threshold, the user is asked first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {
                    "type": "string",
                    "description": (
                        "The COMPLETE new file content. Must include every line the "
                        "file should contain after the write, not just the changed parts."
                    ),
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string", "description": "One-sentence justification."},
            },
            "required": ["path", "content", "confidence", "reason"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run pytest (optionally scoped to a path) and return stdout/stderr + exit code.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Optional test path."}},
        },
    },
    {
        "name": "run_linter",
        "description": "Run ruff check on the repo and return diagnostics as JSON.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "ask_user",
        "description": (
            "Pause and ask the human operator a question. Use when confidence is low, when "
            "intent is ambiguous, or before non-trivial behavioural changes. Returns the user's reply."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "report_finding",
        "description": (
            "Record a finding (bug, design smell, coverage gap) for the final HTML report. "
            "Does not modify files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["bug", "smell", "coverage_gap", "note"]},
                "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                "file": {"type": "string"},
                "line": {"type": "integer"},
                "summary": {"type": "string"},
                "detail": {"type": "string"},
            },
            "required": ["kind", "severity", "summary"],
        },
    },
]


# ---------- Change gate ----------

class ChangeGate:
    """Centralises the confidence/approval logic for mutations."""

    def __init__(self, threshold: float, auto_approve: bool = False):
        self.threshold = threshold
        self.auto_approve = auto_approve
        self.applied: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []

    def decide(self, path: str, old: str, new: str, confidence: float, reason: str) -> bool:
        record = {"path": path, "confidence": confidence, "reason": reason}
        if self.auto_approve or confidence >= self.threshold:
            self.applied.append(record)
            return True

        console.rule(f"[yellow]Review required: {path}[/yellow]")
        console.print(f"[bold]Reason:[/bold] {reason}")
        console.print(f"[bold]Confidence:[/bold] {confidence:.2f} (threshold {self.threshold})")
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
            )
        )
        if diff:
            console.print(Syntax(diff, "diff", theme="monokai", line_numbers=False))
        approved = Confirm.ask("Apply this change?", default=False)
        (self.applied if approved else self.rejected).append(record)
        return approved


# ---------- Executors ----------

class ToolExecutor:
    def __init__(self, repo_root: Path, gate: ChangeGate, findings: list[dict[str, Any]]):
        self.repo_root = repo_root
        self.gate = gate
        self.findings = findings

    def _safe_path(self, relative: str) -> Path:
        target = (self.repo_root / relative).resolve()
        if not str(target).startswith(str(self.repo_root)):
            raise ValueError(f"Path {relative} escapes repo root")
        return target

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        fn: Callable[[dict[str, Any]], str] | None = getattr(self, f"_t_{name}", None)
        if fn is None:
            return f"Unknown tool: {name}"
        return fn(args)

    # individual tools - prefix _t_ to map from tool name

    def _t_list_directory(self, args: dict[str, Any]) -> str:
        target = self._safe_path(args["path"])
        if not target.exists():
            return f"Not found: {args['path']}"
        if target.is_file():
            return f"{args['path']} is a file, not a directory"
        entries = sorted(
            [f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir() if not p.name.startswith(".")]
        )
        return "\n".join(entries) or "(empty)"

    def _t_read_file(self, args: dict[str, Any]) -> str:
        target = self._safe_path(args["path"])
        if not target.exists():
            return f"Not found: {args['path']}"
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Binary file, cannot read: {args['path']}"
        # cap to prevent runaway tokens on huge files
        if len(text) > 30_000:
            return text[:30_000] + f"\n\n... [truncated, {len(text) - 30_000} more chars]"
        return text

    def _t_search_code(self, args: dict[str, Any]) -> str:
        pattern = args["pattern"]
        scope = self._safe_path(args.get("path", "."))
        cmd = ["grep", "-rnE", "--include=*.py", pattern, str(scope)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out = result.stdout.strip() or "(no matches)"
        # trim to keep things bounded
        lines = out.splitlines()
        if len(lines) > 200:
            out = "\n".join(lines[:200]) + f"\n... [{len(lines) - 200} more matches]"
        return out

    def _t_write_file(self, args: dict[str, Any]) -> str:
        path = args["path"]
        content = args["content"]
        confidence = float(args["confidence"])
        reason = args["reason"]
        target = self._safe_path(path)
        old = target.read_text(encoding="utf-8") if target.exists() else ""
        if old == content:
            return f"No change: {path} already has this content."
        if not self.gate.decide(path, old, content, confidence, reason):
            return f"REJECTED by policy: {path}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {path} ({len(content)} bytes)"

    def _t_run_tests(self, args: dict[str, Any]) -> str:
        path = args.get("path", "")
        cmd = ["python", "-m", "pytest", "-q", "--tb=short"]
        if path:
            cmd.append(str(self._safe_path(path)))
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root, timeout=120)
        return f"exit={result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"

    def _t_run_linter(self, args: dict[str, Any]) -> str:
        cmd = ["ruff", "check", "--output-format=json", str(self.repo_root)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # ruff returns non-zero when it finds issues; that's fine
        return result.stdout or "(no issues)"

    def _t_ask_user(self, args: dict[str, Any]) -> str:
        console.rule("[cyan]Agent has a question[/cyan]")
        console.print(args["question"])
        reply = console.input("[bold cyan]your answer >[/bold cyan] ")
        return reply or "(no answer)"

    def _t_report_finding(self, args: dict[str, Any]) -> str:
        self.findings.append(args)
        return f"Recorded {args['kind']} ({args['severity']}): {args['summary']}"
