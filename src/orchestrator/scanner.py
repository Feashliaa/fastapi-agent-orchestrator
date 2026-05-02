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
    """Represent a discovered symbol in the repository.

    Attributes:
        name: The function/class name.
        kind: Symbol category.
        file: Repository-relative file path.
        line: 1-based line number where the symbol starts.
        has_docstring: Whether the symbol has a docstring.
        is_async: Whether the symbol is declared as async.
        http_method: HTTP method for endpoints (e.g., "get"), otherwise None.
        route: Route path for endpoints (e.g., "/health"), otherwise None.
    """

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
    """Audit results for a single file.

    Attributes:
        path: Repository-relative file path.
        symbols: All discovered symbols within the file.
        has_test_file: True if the repository has a corresponding test module.
        loc: Approximate lines of code counted as number of newline characters.
    """

    path: str
    symbols: list[Symbol] = field(default_factory=list)
    has_test_file: bool = False
    loc: int = 0


@dataclass
class RepoAudit:
    """Aggregate audit information across the repository."""

    files: list[FileAudit] = field(default_factory=list)

    @property
    def total_symbols(self) -> int:
        """Return the total number of discovered symbols across all files."""
        return sum(len(f.symbols) for f in self.files)

    @property
    def undocumented(self) -> list[Symbol]:
        """Return symbols that are missing docstrings."""
        return [s for f in self.files for s in f.symbols if not s.has_docstring]

    @property
    def endpoints(self) -> list[Symbol]:
        """Return symbols that were detected as FastAPI endpoints."""
        return [s for f in self.files for s in f.symbols if s.kind == "endpoint"]

    @property
    def untested_files(self) -> list[FileAudit]:
        """Return files that contain symbols but have no corresponding test file."""
        return [f for f in self.files if f.symbols and not f.has_test_file]


HTTP_DECORATORS = {"get", "post", "put", "delete", "patch", "options", "head"}


def _extract_route(decorator: ast.expr) -> tuple[str | None, str | None]:
    """Extract HTTP method and route string from a decorator expression.

    This only supports decorator shapes like 
    ``@router.get('/x')`` where the first positional argument is a string.

    Args:
        decorator: AST expression representing a decorator.

    Returns:
        A tuple of (http_method, route). If the decorator is not recognized as an
        HTTP endpoint decorator, returns (None, None).
    """
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
    """Audit a single Python file for symbols and docstrings.

    Args:
        path: Path to the Python file to audit.
        repo_root: Root directory of the repository, used to compute relative paths.

    Returns:
        A :class:`FileAudit` instance.
    """
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
    """Scan the repository and return an audit.

    The scanner walks all ``.py`` files under ``repo_root`` (excluding a default
    ignore list), parses them with :mod:`ast`, and reports symbols and
    whether they have docstrings.

    Args:
        repo_root: Root directory to scan.
        ignore: Optional set of directory-name fragments to exclude.

    Returns:
        A :class:`RepoAudit` containing per-file symbol lists and derived
        properties (e.g., undocumented symbols, endpoints).
    """
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
