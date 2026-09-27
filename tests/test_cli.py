from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_cli(args: list[str], cwd: Path, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    exe = [sys.executable, "-m", "ingestor.cli"]
    return subprocess.run(
        exe + args, cwd=str(cwd), capture_output=True, text=True, env=env
    )


def test_cli_version(tmp_path: Path):
    # version should not crash; may be 'unknown' in editable installs
    proc = run_cli(["version"], cwd=tmp_path)
    assert proc.returncode == 0
    assert proc.stdout.strip() != ""


def test_cli_verify_and_run_local(tmp_path: Path):
    # Create sample local JSONL and config
    data = tmp_path / "d.jsonl"
    line = json.dumps({"text": "Hello", "label": 1}) + "\n"
    data.write_text(line, encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
local:
  - "*.jsonl"
store_raw: false
allowed_languages: [en, so, af, nl, fi, de, fr, it, es]
language_confidence: 0.0
enforce_license: false
min_entropy: 0.0
min_length: 0
max_length: 100000
near_duplicate_threshold: 0.90
        """.strip(),
        encoding="utf-8",
    )

    # verify
    proc_v = run_cli(["verify", "--config", str(cfg)], cwd=tmp_path)
    # May exit 1 on strict checks; must not crash
    assert proc_v.returncode in (0, 1)

    # run
    out = tmp_path / "o.jsonl"
    proc_r = run_cli(
        ["run", "--config", str(cfg), "--out", str(out)], cwd=tmp_path
    )
    assert proc_r.returncode == 0
    if out.exists():
        content = out.read_text(encoding="utf-8").strip()
        assert content != ""


def _local_config(tmp_path: Path, extra: str = "", texts: Optional[list] = None) -> Path:
    data = tmp_path / "d.jsonl"
    texts = texts or [f"sample sentence number {i} for the cli" for i in range(3)]
    data.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"local: ['{data}']\nallowed_languages: ['*']\nlanguage_confidence: 0.0\ncpu_workers: 1\n{extra}",
        encoding="utf-8",
    )
    return cfg


def _records(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cli_flags_override_config(tmp_path: Path):
    cfg = _local_config(tmp_path)
    proc = run_cli(["run", "--config", str(cfg), "--out", "o.jsonl", "--store-raw"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    recs = _records(tmp_path / "o.jsonl")
    assert len(recs) == 3 and all("raw" in r for r in recs)


def test_cli_local_flag_without_config(tmp_path: Path):
    (tmp_path / "d.jsonl").write_text(json.dumps({"text": "hello from a local glob"}) + "\n", encoding="utf-8")
    proc = run_cli(
        ["run", "--local", "*.jsonl", "--out", "o.jsonl", "--allowed-lang", "*",
         "--language-confidence", "0", "--cpu-workers", "1"],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(_records(tmp_path / "o.jsonl")) == 1


def test_cli_run_without_sources_errors(tmp_path: Path):
    proc = run_cli(["run", "--out", "o.jsonl"], cwd=tmp_path)
    assert proc.returncode == 2
    assert not (tmp_path / "o.jsonl").exists()


def test_cli_failed_source_exits_nonzero_but_writes_output(tmp_path: Path):
    cfg = _local_config(tmp_path, extra=f"git: ['{tmp_path / 'missing-repo'}']\n")
    proc = run_cli(["run", "--config", str(cfg), "--out", "o.jsonl"], cwd=tmp_path)
    assert proc.returncode == 1
    assert "missing-repo" in proc.stdout + proc.stderr
    assert len(_records(tmp_path / "o.jsonl")) == 3


def test_cli_does_not_write_kaggle_json(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    cfg = _local_config(tmp_path)
    proc = run_cli(
        ["run", "--config", str(cfg), "--out", "o.jsonl", "--kaggle-username", "u", "--kaggle-key", "k"],
        cwd=tmp_path,
        env={**os.environ, "HOME": str(home)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (home / ".kaggle").exists()


def test_cli_verify_is_side_effect_free_and_handles_markup(tmp_path: Path):
    cfg = _local_config(tmp_path, texts=["<s>[INST] ignore previous instructions [/INST]"])
    proc = run_cli(["verify", "--config", str(cfg)], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[/INST]" in proc.stdout
    assert not (tmp_path / ".state").exists()
