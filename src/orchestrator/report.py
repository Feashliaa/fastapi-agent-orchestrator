"""HTML report generator"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Template

from .agent import AgentResult
from .scanner import RepoAudit

_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FastAPI Orchestrator - {{ mode }} run</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 0.3rem; }
  h2 { margin-top: 2rem; color: #444; }
  .meta { background: #f4f4f4; padding: 0.8rem 1rem; border-radius: 6px; font-size: 0.9rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }
  .card { background: #fafafa; border: 1px solid #e0e0e0; padding: 1rem; border-radius: 6px; }
  .card .n { font-size: 1.8rem; font-weight: 600; }
  .card .l { font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
  table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; font-size: 0.92rem; }
  th, td { text-align: left; padding: 0.5rem 0.7rem; border-bottom: 1px solid #eee; }
  th { background: #f8f8f8; font-weight: 600; }
  .sev-high { color: #c0392b; font-weight: 600; }
  .sev-medium { color: #d68910; font-weight: 600; }
  .sev-low { color: #7f8c8d; }
  .kind-bug { color: #c0392b; }
  .kind-smell { color: #8e44ad; }
  .kind-coverage_gap { color: #2980b9; }
  .kind-note { color: #555; }
  pre { background: #2d2d2d; color: #f8f8f2; padding: 0.8rem; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; }
  .final { background: #eef6ff; border-left: 4px solid #2980b9; padding: 0.8rem 1rem; border-radius: 0 6px 6px 0; }
</style>
</head>
<body>
<h1>FastAPI Orchestrator - {{ mode }}</h1>
<div class="meta">
  <strong>Repo:</strong> {{ repo }}<br>
  <strong>Model:</strong> {{ model }}<br>
  <strong>Run:</strong> {{ timestamp }}
</div>

<div class="grid">
  <div class="card"><div class="n">{{ result.iterations }}</div><div class="l">Iterations</div></div>
  <div class="card"><div class="n">{{ result.applied_changes|length }}</div><div class="l">Changes applied</div></div>
  <div class="card"><div class="n">{{ result.rejected_changes|length }}</div><div class="l">Rejected</div></div>
  <div class="card"><div class="n">{{ result.findings|length }}</div><div class="l">Findings</div></div>
  <div class="card"><div class="n">{{ result.input_tokens + result.output_tokens }}</div><div class="l">Tokens</div></div>
</div>

{% if audit %}
<h2>Codebase at a glance</h2>
<div class="grid">
  <div class="card"><div class="n">{{ audit.files|length }}</div><div class="l">Python files</div></div>
  <div class="card"><div class="n">{{ audit.total_symbols }}</div><div class="l">Symbols</div></div>
  <div class="card"><div class="n">{{ audit.endpoints|length }}</div><div class="l">FastAPI endpoints</div></div>
  <div class="card"><div class="n">{{ audit.undocumented|length }}</div><div class="l">Undocumented</div></div>
  <div class="card"><div class="n">{{ audit.untested_files|length }}</div><div class="l">Untested modules</div></div>
</div>
{% endif %}

{% if result.findings %}
<h2>Findings</h2>
<table>
  <thead><tr><th>Kind</th><th>Severity</th><th>Location</th><th>Summary</th></tr></thead>
  <tbody>
  {% for f in result.findings %}
    <tr>
      <td class="kind-{{ f.kind }}">{{ f.kind }}</td>
      <td class="sev-{{ f.severity }}">{{ f.severity }}</td>
      <td>{% if f.file %}<code>{{ f.file }}{% if f.line %}:{{ f.line }}{% endif %}</code>{% endif %}</td>
      <td>{{ f.summary }}{% if f.detail %}<br><small>{{ f.detail }}</small>{% endif %}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

{% if result.applied_changes %}
<h2>Applied changes</h2>
<table>
  <thead><tr><th>File</th><th>Confidence</th><th>Reason</th></tr></thead>
  <tbody>
  {% for c in result.applied_changes %}
    <tr><td><code>{{ c.path }}</code></td><td>{{ "%.2f"|format(c.confidence) }}</td><td>{{ c.reason }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

{% if result.rejected_changes %}
<h2>Rejected changes</h2>
<table>
  <thead><tr><th>File</th><th>Confidence</th><th>Reason</th></tr></thead>
  <tbody>
  {% for c in result.rejected_changes %}
    <tr><td><code>{{ c.path }}</code></td><td>{{ "%.2f"|format(c.confidence) }}</td><td>{{ c.reason }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<h2>Agent's final summary</h2>
<div class="final">{{ result.final_text }}</div>

</body>
</html>
""")


def write_report(
    *,
    path: Path,
    mode: str,
    repo: Path,
    model: str,
    result: AgentResult,
    audit: RepoAudit | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = _TEMPLATE.render(
        mode=mode,
        repo=str(repo),
        model=model,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        result=result,
        audit=audit,
    )
    path.write_text(html, encoding="utf-8")
