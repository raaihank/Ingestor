from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pytest
from typer.testing import CliRunner

import ingestor
from ingestor.cli import app


def _iter_text_files(root: Path) -> Iterable[Path]:
    skip_dirs = {".git", "dist", "build", "__pycache__", ".venv", "venv"}
    allowed_exts = {".py", ".md", ".toml", ".yaml", ".yml", ".txt"}
    for dirpath, dirnames, filenames in os.walk(root):
        # skip unwanted directories
        parts = set(Path(dirpath).parts)
        if parts.intersection(skip_dirs):
            continue
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in allowed_exts:
                yield p


def test_package_directory_name():
    pkg_dir = Path(ingestor.__file__).resolve().parent
    assert pkg_dir.name == "ingestor"


def test_pyproject_scripts_and_packages():
    try:
        import tomllib  # type: ignore[attr-defined]
    except Exception:
        pytest.skip("tomllib not available on this Python version")

    repo_root = Path(__file__).resolve().parents[1]
    pyproject = repo_root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    scripts = data.get("project", {}).get("scripts", {})
    assert "ingestor" in scripts
    assert scripts["ingestor"] == "ingestor.cli:app"

    wheel = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
    )
    pkgs = wheel.get("packages", [])
    assert "ingestor" in pkgs


def test_cli_commands_registered():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    help_text = result.stdout
    # Ensure core commands stay stable
    for cmd in ("run", "verify", "test", "version"):
        assert cmd in help_text
