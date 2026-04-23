# FastAPI Orchestrator

An AI-driven dev orchestrator for FastAPI codebases. It audits code, writes docstrings, generates pytest coverage, and fixes trivial bugs - and it knows when to ask before acting.

Works with either Anthropic or OpenAI models out of the box.

## The shape of the thing

```
orchestrator audit   # read-only inventory + risk flags
orchestrator docs    # add/update Google-style docstrings
orchestrator bugs    # detect and fix trivial logical bugs
orchestrator tests   # generate pytest + httpx tests
orchestrator all     # docs → bugs → tests in sequence
orchestrator watch   # file-watcher mode: docs update on .py change
```

Every command accepts `--repo PATH`, `--provider {anthropic|openai}`, `--model ID`, `--yes` (auto-approve), and `--report FILE.html`.

## Install

```bash
git clone <this-repo>
cd fastapi-agent-orchestrator
python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

## Running against the SMRT eval repo

On each branch they test:

```bash
# Base branch - initial audit + docs + bug fixes + tests
git checkout base
orchestrator audit --repo . --report reports/base-audit.html
orchestrator all   --repo . --report-dir reports/

# Feature branches - re-run to update docs, catch new bugs, refresh tests
git checkout feature-1
orchestrator all --repo . --report-dir reports/feature-1/

git checkout feature-2
orchestrator all --repo . --report-dir reports/feature-2/
```

Interactive runs (default) will pause for low-confidence changes; `--yes` runs fully autonomously. To switch providers at runtime:

```bash
orchestrator all --repo . --provider openai --model gpt-5.4-nano-2026-03-17 --yes --report-dir reports/
```

## Design decisions, briefly

**CLI, not a daemon.** A file-watcher is available (`orchestrator watch`), but the primary interface is a one-shot CLI. Reviewers need deterministic, inspectable runs against three branches - a persistent background process is the wrong shape for that.

**Real tool-use loop, not prompt-stuffing.** `src/orchestrator/agent.py` runs a genuine tool-use conversation. The model navigates the repo via `list_directory`, `read_file`, `search_code`, `run_tests`, `run_linter`, and mutates only through `write_file`. It can also call `ask_user` and `report_finding`. This is not a chain of one-shot prompts; the model reasons over real tool output across iterations.

**Provider-agnostic.** `providers.py` abstracts the LLM behind a `Provider` interface with two concrete implementations (`AnthropicProvider`, `OpenAIProvider`). The agent loop doesn't care which one's underneath. Switching is a flag; adding Gemini or a local model is a subclass.

**Confidence-gated mutations.** Every `write_file` call must include a confidence score in [0,1] and a one-sentence reason. `ChangeGate` applies changes >= 0.85 automatically and routes the rest through a human review (unified diff in terminal, y/n prompt). This directly addresses the spec's "knows when it needs a human's okay" line and makes the agent's reasoning auditable.

**AST grounding.** Before the LLM sees anything, `scanner.py` walks the repo with `ast` and produces a structured inventory: FastAPI endpoints (including their HTTP method and route), undocumented symbols, modules without a matching test file. That inventory is injected into the agent's first message, so it cannot hallucinate "I documented the "_____" endpoint" when no such endpoint exists.

**Three prompts, one engine.** The docs/bugs/tests modes share the same tool loop - what differs is the system prompt and the auto-injected worklist. The prompts live in `prompts.py` for easy iteration.

**FastAPI-aware test generation.** The tests prompt mandates `pytest-asyncio` + `httpx.AsyncClient` with `ASGITransport` for async endpoints (the common failure mode is using sync `TestClient` on async routes and getting confusing errors). Tautological tests (`assert x is not None`) are explicitly disallowed.

**Pipeline order: docs → bugs → tests.** Running bugs before tests avoids a chicken-and-egg problem: if tests are written against current (possibly buggy) behavior, they codify the bugs and bugs-mode can't fix anything without breaking them. With this ordering, bugs-mode operates on clean code + sparse existing tests, and tests-mode writes tests against corrected behavior.

**Findings, not just fixes.** The `report_finding` tool lets the agent record things it noticed but chose not to act on (design smells, ambiguous bugs, coverage gaps). These land in the HTML report alongside applied/rejected changes and the agent's final summary.

**HTML report.** `orchestrator <mode> --report out.html` produces a standalone report: counts, findings table with severity, diff of applied changes, rejected changes with confidence scores, and the agent's final summary.

## Project layout

```
src/orchestrator/
  cli.py        # Typer commands - audit, docs, tests, bugs, all, watch
  agent.py      # Provider-agnostic tool-use loop
  providers.py  # Anthropic + OpenAI provider adapters
  tools.py      # Tool schemas, ChangeGate, ToolExecutor
  scanner.py    # AST-based repo inventory
  prompts.py    # System prompts per mode
  report.py     # Jinja2 HTML report
  config.py     # Env-driven config
tests/
  test_scanner.py  # Unit tests for the AST scanner
```

## Configuration

All via env (or `.env`):

| Var                                 | Default              | Meaning                                                      |
| ----------------------------------- | -------------------- | ------------------------------------------------------------ |
| `ORCHESTRATOR_PROVIDER`             | `anthropic`          | `anthropic` or `openai`. Override per-run with `--provider`. |
| `ANTHROPIC_API_KEY`                 |                      | Required when provider is `anthropic`.                       |
| `OPENAI_API_KEY`                    |                      | Required when provider is `openai`.                          |
| `ORCHESTRATOR_MODEL`                | per-provider default | Override per-run with `--model`.                             |
| `ORCHESTRATOR_MAX_ITERATIONS`       | `30`                 | Hard cap on tool-use turns per run.                          |
| `ORCHESTRATOR_AUTO_APPLY_THRESHOLD` | `0.85`               | Changes below this confidence require user approval.         |

Per-provider defaults if `ORCHESTRATOR_MODEL` is unset: Anthropic → `claude-sonnet-4-6`, OpenAI → `gpt-4o-mini`.

## Model notes

The agent has been exercised against OpenAI's `gpt-5.4-nano-2026-03-17` with good results - a full `all` run against a small FastAPI codebase typically costs under $0.15 and converges in 30-50 total iterations across the three modes. Weaker models (e.g. `gpt-4o-mini`) will occasionally get stuck in syntax-fixup loops or hit the confidence threshold imprecisely; stronger reasoning models give noticeably cleaner output.

## What's intentionally out of scope

- No VCS automation - the agent does not commit, push, or open PRs. The evaluator runs it against a branch checkout and inspects the diff.
- No cross-repo learning or memory. Every run starts clean so results are reproducible.
- Bugs mode will not refactor. Anything beyond a trivial, local fix becomes a finding, not a mutation.