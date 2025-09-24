from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    exe = [sys.executable, "-m", "ingestor.cli"]
    return subprocess.run(
        exe + args, cwd=str(cwd), capture_output=True, text=True
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
