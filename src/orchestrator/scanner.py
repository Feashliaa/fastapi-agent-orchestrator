"""AST-based codebase scanner.

Grounds the agent in reality before it starts generating:
- What files exist?
- What FastAPI endpoints/functions/classes are defined?
- What's missing docstrings?
- What has no corresponding test file?
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Symbol:
    name: str
    kind: str  # "function" | "async_function" | "class" | "endpoint"
    file: str
    line: int
    has_docstring: bool
    is_async: bool = False
    http_method: str | None = None
    route: str | None = None


@dataclass
class FileAudit:
    path: str
    symbols: list[Symbol] = field(default_factory=list)
    has_test_file: bool = False
    loc: int = 0


@dataclass
class RepoAudit:
    files: list[FileAudit] = field(default_factory=list)

    @property
    def total_symbols(self) -> int:
        return sum(len(f.symbols) for f in self.files)

    @property
    def undocumented(self) -> list[Symbol]:
        return [s for f in self.files for s in f.symbols if not s.has_docstring]

    @property
    def endpoints(self) -> list[Symbol]:
        return [s for f in self.files for s in f.symbols if s.kind == "endpoint"]

    @property
    def untested_files(self) -> list[FileAudit]:
        return [f for f in self.files if f.symbols and not f.has_test_file]


HTTP_DECORATORS = {"get", "post", "put", "delete", "patch", "options", "head"}


def _extract_route(decorator: ast.expr) -> tuple[str | None, str | None]:
    """Return (http_method, route) if decorator is @router.get('/x'), else (None, None)."""
    if not isinstance(decorator, ast.Call):
        return None, None
    func = decorator.func
    if not isinstance(func, ast.Attribute):
        return None, None
    method = func.attr.lower()
    if method not in HTTP_DECORATORS:
        return None, None
    route: str | None = None
    if decorator.args and isinstance(decorator.args[0], ast.Constant):
        value = decorator.args[0].value
        if isinstance(value, str):
            route = value
    return method, route


def _audit_file(path: Path, repo_root: Path) -> FileAudit:
    rel = path.relative_to(repo_root).as_posix()
    audit = FileAudit(path=rel)
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return audit
    audit.loc = source.count("\n") + 1
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return audit

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_async = isinstance(node, ast.AsyncFunctionDef)
            kind = "async_function" if is_async else "function"
            method = route = None
            for dec in node.decorator_list:
                m, r = _extract_route(dec)
                if m:
                    kind = "endpoint"
                    method = m
                    route = r
                    break
            audit.symbols.append(
                Symbol(
                    name=node.name,
                    kind=kind,
                    file=rel,
                    line=node.lineno,
                    has_docstring=ast.get_docstring(node) is not None,
                    is_async=is_async,
                    http_method=method,
                    route=route,
                )
            )
        elif isinstance(node, ast.ClassDef):
            audit.symbols.append(
                Symbol(
                    name=node.name,
                    kind="class",
                    file=rel,
                    line=node.lineno,
                    has_docstring=ast.get_docstring(node) is not None,
                )
            )
    return audit


def scan_repo(repo_root: Path, ignore: set[str] | None = None) -> RepoAudit:
    """Walk the repo, parse every .py file, return a structured audit."""
    ignore = ignore or {".venv", "venv", "__pycache__", ".git", "node_modules", "build", "dist"}
    files: list[FileAudit] = []
    all_py = [
        p for p in repo_root.rglob("*.py")
        if not any(part in ignore for part in p.parts)
    ]
    test_files = {p.stem for p in all_py if p.stem.startswith("test_") or p.parent.name == "tests"}

    for py in all_py:
        if py.stem.startswith("test_"):
            continue
        audit = _audit_file(py, repo_root)
        audit.has_test_file = f"test_{py.stem}" in test_files
        files.append(audit)

    return RepoAudit(files=files)
