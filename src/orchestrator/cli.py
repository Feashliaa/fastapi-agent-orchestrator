"""Command-line interface."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .agent import Agent, AgentResult
from .config import Config
from .prompts import AUDIT_PROMPT, BUGS_PROMPT, DOCS_PROMPT, TESTS_PROMPT
from .report import write_report
from .scanner import scan_repo

app = typer.Typer(
    add_completion=False,
    help="FastAPI AI-Driven Dev Orchestrator - docs, tests, and trivial bug fixes for FastAPI codebases.",
)
console = Console()


# Shared option definitions
ProviderOpt = typer.Option(
    None, "--provider", "-p",
    help="LLM provider: 'anthropic' or 'openai'. Defaults to $ORCHESTRATOR_PROVIDER or 'anthropic'.",
)
ModelOpt = typer.Option(
    None, "--model", "-m",
    help="Model id. Defaults to a per-provider sensible choice.",
)
RepoOpt = typer.Option(Path.cwd(), "--repo", "-r", help="Path to the repo to analyse.")
ReportOpt = typer.Option(None, "--report", help="Write an HTML report here.")
YesOpt = typer.Option(False, "--yes", "-y", help="Auto-approve all changes.")


def _load(repo: Path | None, provider: str | None, model: str | None) -> Config:
    try:
        cfg = Config.load(repo, provider_override=provider, model_override=model)
        console.print(f"[dim]Using {cfg.provider}:{cfg.model}[/dim]")
        return cfg
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


def _fmt_endpoint(s) -> str:
    method = (s.http_method or "?").upper()
    route = s.route or ""
    return f"{method} {route} ({s.file}:{s.line})"


def _summarise(mode: str, result: AgentResult) -> None:
    t = Table(title=f"{mode} - run summary ({result.provider}:{result.model})", show_header=False)
    t.add_row("Iterations", str(result.iterations))
    t.add_row("Changes applied", str(len(result.applied_changes)))
    t.add_row("Changes rejected", str(len(result.rejected_changes)))
    t.add_row("Findings", str(len(result.findings)))
    t.add_row("Tokens in/out", f"{result.input_tokens} / {result.output_tokens}")
    console.print(t)


@app.command()
def audit(
    repo: Path = RepoOpt,
    provider: Optional[str] = ProviderOpt,
    model: Optional[str] = ModelOpt,
    report: Optional[Path] = ReportOpt,
):
    """Read-only pass: scan the repo and let the agent flag risks."""
    cfg = _load(repo, provider, model)
    console.print(f"[bold]Auditing[/bold] {cfg.repo_root}")
    audit_data = scan_repo(cfg.repo_root)
    console.print(
        f"Found {len(audit_data.files)} Python files, {audit_data.total_symbols} symbols, "
        f"{len(audit_data.endpoints)} FastAPI endpoints, "
        f"{len(audit_data.undocumented)} undocumented, "
        f"{len(audit_data.untested_files)} modules without tests."
    )
    task = (
        "Perform an audit of this FastAPI codebase. Here is a machine-generated inventory "
        "to ground you:\n\n" + json.dumps(
            {
                "files": len(audit_data.files),
                "endpoints": [_fmt_endpoint(s) for s in audit_data.endpoints],
                "undocumented_sample": [
                    f"{s.kind} {s.name} ({s.file}:{s.line})" for s in audit_data.undocumented[:20]
                ],
                "untested_files": [f.path for f in audit_data.untested_files[:20]],
            },
            indent=2,
        )
        + "\n\nUse the tools to dig deeper where it matters, then stop."
    )
    agent = Agent(cfg, auto_approve=False)
    result = agent.run(AUDIT_PROMPT, task)
    _summarise("audit", result)
    if report:
        write_report(path=report, mode="audit", repo=cfg.repo_root, model=cfg.model, result=result, audit=audit_data)
        console.print(f"[green]Report written to {report}[/green]")


@app.command()
def docs(
    repo: Path = RepoOpt,
    provider: Optional[str] = ProviderOpt,
    model: Optional[str] = ModelOpt,
    yes: bool = YesOpt,
    report: Optional[Path] = ReportOpt,
):
    """Generate/update docstrings."""
    cfg = _load(repo, provider, model)
    audit_data = scan_repo(cfg.repo_root)
    task = (
        f"Target repo: {cfg.repo_root}\n"
        f"{len(audit_data.undocumented)} symbols lack docstrings.\n"
        "Undocumented items (first 40):\n"
        + "\n".join(f"- {s.kind} {s.name} - {s.file}:{s.line}" for s in audit_data.undocumented[:40])
        + "\n\nAdd Google-style docstrings. Stop when you've processed the meaningful items."
    )
    agent = Agent(cfg, auto_approve=yes)
    result = agent.run(DOCS_PROMPT, task)
    _summarise("docs", result)
    if report:
        write_report(path=report, mode="docs", repo=cfg.repo_root, model=cfg.model, result=result, audit=audit_data)


@app.command()
def tests(
    repo: Path = RepoOpt,
    provider: Optional[str] = ProviderOpt,
    model: Optional[str] = ModelOpt,
    yes: bool = YesOpt,
    report: Optional[Path] = ReportOpt,
):
    """Generate pytest test cases for endpoints and core logic."""
    cfg = _load(repo, provider, model)
    audit_data = scan_repo(cfg.repo_root)
    endpoints = "\n".join(f"- {_fmt_endpoint(s)}" for s in audit_data.endpoints)
    untested = "\n".join(f"- {f.path}" for f in audit_data.untested_files[:30])
    task = (
        f"Target repo: {cfg.repo_root}\n"
        f"Endpoints in the codebase:\n{endpoints or '(none detected)'}\n\n"
        f"Modules without a matching test file:\n{untested or '(all covered)'}\n\n"
        "Generate pytest tests (with pytest-asyncio + httpx for async endpoints). "
        "Run them before finishing."
    )
    agent = Agent(cfg, auto_approve=yes)
    result = agent.run(TESTS_PROMPT, task)
    _summarise("tests", result)
    if report:
        write_report(path=report, mode="tests", repo=cfg.repo_root, model=cfg.model, result=result, audit=audit_data)


@app.command()
def bugs(
    repo: Path = RepoOpt,
    provider: Optional[str] = ProviderOpt,
    model: Optional[str] = ModelOpt,
    yes: bool = YesOpt,
    report: Optional[Path] = ReportOpt,
):
    """Detect and (optionally) fix trivial logical bugs."""
    cfg = _load(repo, provider, model)
    audit_data = scan_repo(cfg.repo_root)
    task = (
        f"Target repo: {cfg.repo_root}\n"
        f"{len(audit_data.endpoints)} endpoints, {len(audit_data.files)} files.\n"
        "Start by running the linter and existing tests. Then read strategically. "
        "Fix only trivial logical bugs you're highly confident about; report the rest."
    )
    agent = Agent(cfg, auto_approve=yes)
    result = agent.run(BUGS_PROMPT, task)
    _summarise("bugs", result)
    if report:
        write_report(path=report, mode="bugs", repo=cfg.repo_root, model=cfg.model, result=result, audit=audit_data)


@app.command("all")
def all_cmd(
    repo: Path = RepoOpt,
    provider: Optional[str] = ProviderOpt,
    model: Optional[str] = ModelOpt,
    yes: bool = YesOpt,
    report_dir: Optional[Path] = typer.Option(None, "--report-dir"),
):
    """Run docs → tests → bugs in sequence."""
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("docs", "tests", "bugs"):
        console.rule(f"[bold cyan]{mode}[/bold cyan]")
        report_path = report_dir / f"{mode}.html" if report_dir else None
        cmd = {"docs": docs, "tests": tests, "bugs": bugs}[mode]
        cmd(repo=repo, provider=provider, model=model, yes=yes, report=report_path)


@app.command()
def watch(
    repo: Path = RepoOpt,
    provider: Optional[str] = ProviderOpt,
    model: Optional[str] = ModelOpt,
    yes: bool = YesOpt,
):
    """Watch the repo; on any .py change, run the docs agent for that file."""
    from watchfiles import watch as _watch
    cfg = _load(repo, provider, model)
    console.print(f"[bold]Watching[/bold] {cfg.repo_root} (Ctrl-C to stop)")
    for changes in _watch(cfg.repo_root):
        py_changed = [p for _, p in changes if p.endswith(".py") and "/tests/" not in p]
        if not py_changed:
            continue
        console.print(f"[yellow]Changed:[/yellow] {py_changed}")
        task = (
            "These files just changed. Review them and update docstrings where needed:\n"
            + "\n".join(f"- {p}" for p in py_changed)
        )
        agent = Agent(cfg, auto_approve=yes)
        result = agent.run(DOCS_PROMPT, task)
        _summarise("watch:docs", result)


if __name__ == "__main__":
    app()
