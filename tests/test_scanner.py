"""Tests for the scanner module - exercises the AST logic without any network calls."""
from __future__ import annotations

import textwrap
from pathlib import Path

from orchestrator.scanner import scan_repo


def _write(root: Path, relpath: str, content: str) -> None:
    p = root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content), encoding="utf-8")


def test_scanner_finds_endpoints(tmp_path: Path):
    _write(tmp_path, "app/main.py", '''
        from fastapi import FastAPI
        router = FastAPI()

        @router.get("/users/{uid}")
        async def get_user(uid: int):
            """Fetch a user."""
            return {"uid": uid}

        @router.post("/users")
        async def create_user(payload: dict):
            return payload
    ''')
    audit = scan_repo(tmp_path)
    endpoints = audit.endpoints
    assert len(endpoints) == 2
    methods = {(e.http_method, e.route) for e in endpoints}
    assert ("get", "/users/{uid}") in methods
    assert ("post", "/users") in methods


def test_scanner_flags_undocumented(tmp_path: Path):
    _write(tmp_path, "pkg/util.py", '''
        def documented():
            """Has a docstring."""
            return 1

        def undocumented():
            return 2
    ''')
    audit = scan_repo(tmp_path)
    names = {s.name for s in audit.undocumented}
    assert "undocumented" in names
    assert "documented" not in names


def test_scanner_detects_test_coverage(tmp_path: Path):
    _write(tmp_path, "pkg/a.py", "def f(): return 1\n")
    _write(tmp_path, "pkg/b.py", "def g(): return 2\n")
    _write(tmp_path, "tests/test_a.py", "def test_f(): pass\n")
    audit = scan_repo(tmp_path)
    untested = {f.path for f in audit.untested_files}
    assert "pkg/b.py" in untested
    assert "pkg/a.py" not in untested


def test_scanner_ignores_venv(tmp_path: Path):
    _write(tmp_path, ".venv/lib/foo.py", "def nope(): pass\n")
    _write(tmp_path, "src/real.py", "def yep(): pass\n")
    audit = scan_repo(tmp_path)
    paths = {f.path for f in audit.files}
    assert "src/real.py" in paths
    assert not any(".venv" in p for p in paths)


def test_scanner_survives_syntax_errors(tmp_path: Path):
    _write(tmp_path, "broken.py", "def oops(:\n  pass\n")
    _write(tmp_path, "ok.py", "def fine(): pass\n")
    audit = scan_repo(tmp_path)
    paths = {f.path for f in audit.files}
    assert "broken.py" in paths
    assert "ok.py" in paths
    # broken file yields no symbols but doesn't crash the scan
    broken = next(f for f in audit.files if f.path == "broken.py")
    assert broken.symbols == []
