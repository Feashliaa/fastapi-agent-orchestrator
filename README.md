# FastAPI Orchestrator

An AI-driven dev orchestrator for FastAPI codebases. It audits code, writes docstrings, generates pytest coverage, and fixes trivial bugs - and it knows when to ask before acting.

Works with either Anthropic or OpenAI models out of the box.

## What it does

Given a FastAPI codebase, the orchestrator runs three modes:

- **docs** - adds Google-style docstrings to undocumented functions, classes, and endpoints.
- **bugs** - identifies and fixes trivial logical bugs (off-by-one, missing `None` checks, wrong HTTP status codes, mishandled async, illegal state transitions).
- **tests** - generates pytest coverage using `pytest-asyncio` + `httpx.AsyncClient` for async endpoints.

It does this through a real tool-use loop - the model navigates the repo with `read_file`, `list_directory`, `search_code`, `run_tests`, `run_linter`, and mutates files through a confidence-gated `write_file`. Low-confidence changes prompt you for approval; high-confidence changes auto-apply.

## Install

```bash
git clone <this-repo>
cd fastapi-agent-orchestrator
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
```

Create a `.env` file in the project root with your API key:

```
ORCHESTRATOR_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Or, for OpenAI:

```
ORCHESTRATOR_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

You can set both and switch per-run with `--provider`. Verify the install:

```bash
orchestrator --help
```

## How to run it

Sample reports from a reference run on a toy FastAPI app are in `reports/sample/`.

The orchestrator is a CLI tool that operates on a **separate** FastAPI codebase (your "target"). You run it from the orchestrator's directory and point it at the target with `--repo`.

**Typical flow for evaluating one branch:**

```bash
# 1. Check out the branch of the target repo you want to evaluate.
cd /path/to/your/target-repo
git checkout base                 # or whichever branch

# 2. Back in the orchestrator directory, run the pipeline.
cd /path/to/fastapi-agent-orchestrator
orchestrator all --repo /path/to/your/target-repo --yes --report-dir reports/base/
```

`orchestrator all` runs three modes in sequence: **docs → bugs → tests**. It writes changes directly to files in the target repo and drops three HTML reports into `reports/base/`.

**For each branch you want to evaluate**, repeat: check out the branch in the target, rerun `orchestrator all` with a distinct `--report-dir`.

**Individual modes** are also available if you want to isolate one step:

```bash
orchestrator audit --repo /path/to/target --report reports/audit.html   # read-only; no changes
orchestrator docs  --repo /path/to/target --report reports/docs.html
orchestrator bugs  --repo /path/to/target --report reports/bugs.html
orchestrator tests --repo /path/to/target --report reports/tests.html
```

### Flags

| Flag                             | Meaning                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| `--repo PATH`                    | Target repo to operate on. Required.                                                           |
| `--yes`, `-y`                    | Auto-approve all changes. Without it, you'll get a y/n prompt for every low-confidence change. |
| `--provider {anthropic\|openai}` | Override the provider set in `.env`.                                                           |
| `--model ID`                     | Override the default model for the chosen provider.                                            |
| `--report FILE.html`             | Write an HTML summary (single-mode commands).                                                  |
| `--report-dir DIR/`              | Write `docs.html`, `bugs.html`, `tests.html` here (for `all`).                                 |

### Full example - OpenAI, autonomous

```bash
orchestrator all \
    --repo /path/to/target-repo \
    --provider openai \
    --model gpt-5.4-nano-2026-03-17 \
    --yes \
    --report-dir reports/
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

Default models: Anthropic → `claude-sonnet-4-6`, OpenAI → `gpt-4o-mini`.

## Design decisions

**CLI, not a daemon.** A file-watcher is available (`orchestrator watch`), but the primary interface is a one-shot CLI. Reviewers need deterministic, inspectable runs against multiple branches - a persistent background process is the wrong shape for that.

**Real tool-use loop, not prompt-stuffing.** The model runs a genuine multi-turn tool-use conversation. It navigates the repo, reads real files, runs real tests, and gets real output back. Not a chain of single-shot prompts; the agent reasons across iterations.

**Provider-agnostic.** `providers.py` defines a `Provider` interface with two implementations (`AnthropicProvider`, `OpenAIProvider`). The agent loop doesn't care which is underneath. Switching is a flag; adding Gemini or a local model is a new subclass.

**Confidence-gated mutations.** Every `write_file` call must include a confidence score in `[0,1]` and a one-sentence reason. `ChangeGate` applies changes ≥ 0.85 automatically and routes the rest through a human review (unified diff in terminal, y/n prompt). This directly addresses the spec's "knows when it needs a human's okay" line and makes the agent's reasoning auditable.

**AST grounding.** Before the LLM sees anything, `scanner.py` walks the repo with `ast` and produces a structured inventory: FastAPI endpoints (with HTTP method and route), undocumented symbols, modules without a matching test file. That inventory is injected into the agent's first message, so it cannot hallucinate "I documented the FooBar endpoint" when no such endpoint exists.

**Three prompts, one engine.** The docs, bugs, and tests modes share the same tool loop - what differs is the system prompt and the auto-injected worklist. Prompts live in `prompts.py` for easy iteration.

**Pipeline order: docs → bugs → tests.** Running bugs before tests avoids a chicken-and-egg problem. If tests are written against current (possibly buggy) behavior, they codify the bugs, and bugs-mode can't fix anything without breaking them. With this ordering, bugs-mode operates on cleaner code, and tests-mode writes tests against corrected behavior.

**FastAPI-aware test generation.** The tests prompt mandates `pytest-asyncio` + `httpx.AsyncClient` with `ASGITransport` for async endpoints. A common failure mode is using sync `TestClient` on async routes and getting confusing errors. Tautological tests (`assert x is not None`) are explicitly disallowed.

**Findings, not just fixes.** The `report_finding` tool lets the agent record things it noticed but chose not to act on (design smells, ambiguous bugs, coverage gaps). These land in the HTML report alongside applied/rejected changes and the agent's final summary.

**HTML report.** Every mode produces a standalone report: counts, findings with severity, applied changes, rejected changes with confidence scores, and the agent's final summary.

## Project layout

```
src/orchestrator/
  cli.py        # Typer commands - audit, docs, bugs, tests, all, watch
  agent.py      # Provider-agnostic tool-use loop
  providers.py  # Anthropic + OpenAI adapters
  tools.py      # Tool schemas, ChangeGate, ToolExecutor
  scanner.py    # AST-based repo inventory
  prompts.py    # System prompts per mode
  report.py     # Jinja2 HTML report
  config.py     # Env-driven config
tests/
  test_scanner.py   # Unit tests for the AST scanner
```

## Model notes

Tested end-to-end against OpenAI's `gpt-5.4-nano-2026-03-17`: a full `all` run on a small FastAPI codebase typically costs under $0.15 and converges in 30–50 total iterations across the three modes. Weaker models (e.g. `gpt-4o-mini`) occasionally get stuck in syntax-fixup loops; stronger reasoning models produce noticeably cleaner output. Anthropic's Sonnet tier is the reference target for highest-quality results.

## Out of scope

- **No VCS automation.** The agent does not commit, push, or open PRs. The evaluator runs it against a branch checkout and inspects the diff.
- **No cross-repo memory.** Every run starts clean so results are reproducible.
- **Bugs mode will not refactor.** Anything beyond a trivial, local fix becomes a finding, not a mutation.