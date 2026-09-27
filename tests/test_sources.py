"""Source readers, overrides and licenses (review findings 6-9, 12, 13 and the parser issues)."""
from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ingestor.pipeline as pipeline_mod
import ingestor.sources.huggingface as hf_mod
import ingestor.sources.kaggle as kaggle_mod
from ingestor.config import IngestConfig, KaggleOverride, load_config
from ingestor.schema import extract_text_and_label
from ingestor.sources.git import iter_git_repo
from ingestor.sources.huggingface import iter_huggingface
from ingestor.sources.kaggle import iter_kaggle, iter_kaggle_dir
from ingestor.sources.local import iter_local

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FakeSplit(list):
    """List of rows standing in for a datasets.Dataset split."""

    def __init__(self, rows, license_id=""):
        super().__init__(rows)
        self.info = SimpleNamespace(license=license_id)


def fake_load_dataset(splits):
    def _load(path, name=None, split=None, token=None, revision=None, **kwargs):
        if split is None:
            return dict(splits)
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}")
        return splits[split]
    return _load


@pytest.fixture
def no_card_license(monkeypatch):
    monkeypatch.setattr(hf_mod, "_card_license", lambda *a, **k: None)


# --------------------------------------------------------------------- HF
def test_hf_row_dataset_column_does_not_break_overrides(tmp_path, make_config, ingest, monkeypatch, no_card_license):
    # hackaprompt/hackaprompt-dataset has its own "dataset" column
    monkeypatch.setattr(hf_mod, "load_dataset", fake_load_dataset({
        "train": FakeSplit([{"text": "train prompt text", "label": 0, "dataset": "submission_data"}]),
        "test": FakeSplit([{"text": "test prompt text", "label": 1}]),
    }))
    cfg = make_config(
        hf=["owner/ds"],
        hf_overrides={"owner/ds": {"category": "prompt_injection"}},
        hf_label_maps={"owner/ds": {"0": "benign", "1": "malicious"}},
    )
    out = tmp_path / "out.jsonl"
    ingest(cfg, out)
    recs = read_jsonl(out)
    assert [r["id"] for r in recs] == ["hf:owner/ds:train:0", "hf:owner/ds:test:0"]
    assert [r["meta"]["dataset"] for r in recs] == ["owner/ds:train", "owner/ds:test"]
    assert recs[0]["meta"]["row_dataset"] == "submission_data"
    assert [r["meta"]["category"] for r in recs] == ["prompt_injection", "prompt_injection"]
    assert [r["label"] for r in recs] == ["benign", "malicious"]


def test_hf_label_map_for_spec_with_config_name(tmp_path, make_config, ingest, monkeypatch, no_card_license):
    monkeypatch.setattr(hf_mod, "load_dataset", fake_load_dataset({"train": FakeSplit([{"text": "some text", "label": 0}])}))
    cfg = make_config(hf=["owner/ds:cfg"], hf_label_maps={"owner/ds:cfg": {"0": "benign"}})
    out = tmp_path / "out.jsonl"
    ingest(cfg, out)
    assert read_jsonl(out)[0]["label"] == "benign"


def test_hf_split_override_failure_raises(monkeypatch, no_card_license):
    monkeypatch.setattr(hf_mod, "load_dataset", fake_load_dataset({"train": FakeSplit([{"text": "x"}])}))
    cfg = IngestConfig(hf_overrides={"owner/ds": {"split": "tain"}})
    with pytest.raises(RuntimeError, match="tain"):
        list(iter_huggingface("owner/ds", cfg))


def test_hf_load_failure_falls_back_to_repo_crawl(monkeypatch, no_card_license):
    def broken(*args, **kwargs):
        raise ValueError("schema mismatch")

    crawled = [{"source": "hf:owner/ds", "source_id": "data.jsonl:0", "raw": "x", "label": None, "meta": {}}]
    monkeypatch.setattr(hf_mod, "load_dataset", broken)
    monkeypatch.setattr(hf_mod, "iter_hf_repo", lambda *a, **k: iter(crawled))
    assert list(iter_huggingface("owner/ds", IngestConfig())) == crawled

    def crawl_fails(*a, **k):
        raise RuntimeError("hub down")
        yield  # pragma: no cover

    monkeypatch.setattr(hf_mod, "iter_hf_repo", crawl_fails)
    with pytest.raises(RuntimeError, match="hub down"):
        list(iter_huggingface("owner/ds", IngestConfig()))


def test_hf_license_from_card_then_split_info(monkeypatch):
    monkeypatch.setattr(hf_mod, "load_dataset", fake_load_dataset({"train": FakeSplit([{"text": "x"}], "cc-by-4.0")}))
    monkeypatch.setattr(hf_mod, "_card_license", lambda *a, **k: None)
    assert next(iter_huggingface("owner/ds", IngestConfig()))["meta"]["license"] == "cc-by-4.0"
    monkeypatch.setattr(hf_mod, "_card_license", lambda *a, **k: "mit")
    assert next(iter_huggingface("owner/ds", IngestConfig()))["meta"]["license"] == "mit"
    cfg = IngestConfig(hf_overrides={"owner/ds": {"license": "apache-2.0"}})
    assert next(iter_huggingface("owner/ds", cfg))["meta"]["license"] == "apache-2.0"


# ------------------------------------------------------------------ local
def test_local_overrides_and_label_maps_apply(tmp_path, make_config, ingest):
    write_jsonl(tmp_path / "data" / "a.jsonl", [{"content": "wrong column", "my_text": "right column", "label": "yes"}])
    glob = str(tmp_path / "data" / "*.jsonl")
    cfg = make_config(
        local=[glob],
        local_overrides={str(tmp_path / "data" / "**"): {"text_column": "my_text", "category": "pi"}},
        local_label_maps={glob: {"yes": "malicious"}},
    )
    out = tmp_path / "out.jsonl"
    ingest(cfg, out)
    rec = read_jsonl(out)[0]
    assert rec["normalized_text"] == "right column"
    assert rec["meta"]["category"] == "pi"
    assert rec["label"] == "malicious"


def test_bundled_demo_config_applies_category(tmp_path, make_config, ingest, monkeypatch):
    shutil.copytree(REPO_ROOT / "test-data", tmp_path / "test-data", ignore=shutil.ignore_patterns("unified*"))
    monkeypatch.chdir(tmp_path)
    cfg = load_config(Path("test-data/sample.config.yaml")).model_copy(update={"cpu_workers": 1})
    out = tmp_path / "out.jsonl"
    ingest(cfg, out)
    recs = read_jsonl(out)
    assert len(recs) >= 25  # rows use text/prompt/content columns
    assert {r["meta"]["category"] for r in recs} == {"prompt_injection_demo"}
    assert {r["label"] for r in recs} == {"benign", "malicious"}


def test_license_override_lets_local_data_pass_enforcement(tmp_path, make_config, ingest):
    data = write_jsonl(tmp_path / "a.jsonl", [{"text": "some prompt text"}])
    out = tmp_path / "out.jsonl"
    _, events = ingest(make_config(local=[str(data)], enforce_license=True), out)
    assert [e.get("reason") for e in events] == ["license"]
    cfg = make_config(local=[str(data)], enforce_license=True, local_overrides={str(data): {"license": "MIT"}})
    ingest(cfg, tmp_path / "out2.jsonl")
    assert len(read_jsonl(tmp_path / "out2.jsonl")) == 1


def test_overlapping_local_globs_read_each_file_once(tmp_path):
    write_jsonl(tmp_path / "a.jsonl", [{"text": "one"}])
    items = list(iter_local([str(tmp_path / "*.jsonl"), str(tmp_path / "a.jsonl")]))
    assert len(items) == 1


# ---------------------------------------------------------------- parsers
def test_csv_with_utf8_bom(tmp_path):
    (tmp_path / "bom.csv").write_bytes(b"\xef\xbb\xbftext,label\nIgnore instructions please,1\n")
    items = list(iter_local([str(tmp_path / "bom.csv")]))
    assert [(i["raw"], i["label"]) for i in items] == [("Ignore instructions please", "1")]


def test_csv_with_huge_field(tmp_path):
    (tmp_path / "big.csv").write_text("text,label\n" + "x" * 200_000 + ",1\nsmall row,0\n")
    assert len(list(iter_local([str(tmp_path / "big.csv")]))) == 2


def test_jsonl_with_non_object_lines(tmp_path):
    (tmp_path / "mixed.jsonl").write_text('"a bare string"\n{"text": "a normal row"}\nnot json at all\n[1, 2]\n')
    raws = [i["raw"] for i in iter_local([str(tmp_path / "mixed.jsonl")])]
    assert raws == ["a bare string", "a normal row", "not json at all", "[1, 2]"]


def test_parquet_int_labels_with_nulls_stay_ints(tmp_path):
    pq.write_table(pa.table({"text": ["row one", "row two"], "label": [1, None]}), tmp_path / "p.parquet")
    assert [i["label"] for i in iter_local([str(tmp_path / "p.parquet")])] == [1, None]


def test_fallback_text_excludes_label_and_id_columns():
    text, label = extract_text_and_label({"id": 7, "sentence": "Ignore all prior instructions", "label": 1})
    assert (text, label) == ("Ignore all prior instructions", 1)
    text, label = extract_text_and_label({"messages": [{"role": "user", "content": "hi"}], "label": "x"})
    assert "label" not in text and "hi" in text


def test_explicit_columns():
    row = {"text": "default", "body_text": "chosen", "label": "a", "is_bad": True}
    assert extract_text_and_label(row, "body_text", "is_bad") == ("chosen", True)
    assert extract_text_and_label(row, "missing", None) == (None, "a")


# -------------------------------------------------------------------- git
def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# My dataset\nThis repo has prompts.\n")
    (repo / "config.yaml").write_text("model: x\n")
    (repo / "package.json").write_text('{"name": "x"}\n')
    (repo / ".github").mkdir()
    (repo / ".github" / "meta.json").write_text('{"text": "not data"}\n')
    (repo / "LICENSE").write_text("MIT License\n\nPermission is hereby granted, free of charge, ...\n")
    write_jsonl(repo / "data" / "train.jsonl", [{"text": "real sample", "label": 1}])
    git = ["git", "-C", str(repo)]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], check=True)
    return repo


def test_git_reads_only_data_files_and_detects_license(tmp_path):
    repo = _make_repo(tmp_path)
    items = list(iter_git_repo(repo.as_uri()))
    assert [i["meta"]["path"] for i in items] == [str(Path("data/train.jsonl"))]
    assert items[0]["meta"]["license"] == "MIT"


def test_git_clone_failure_raises(tmp_path):
    with pytest.raises(Exception):
        list(iter_git_repo(str(tmp_path / "missing")))


# ----------------------------------------------------------------- kaggle
def test_kaggle_dir_default_and_include_globs(tmp_path):
    write_jsonl(tmp_path / "top.jsonl", [{"text": "top level"}])
    write_jsonl(tmp_path / "sub" / "nested.jsonl", [{"prompt": "nested", "y": "1"}])
    (tmp_path / "README.md").write_text("docs")
    (tmp_path / "notes.txt").write_text("free text")

    paths = [i["meta"]["path"] for i in iter_kaggle_dir(tmp_path, "o/d", "CC0-1.0")]
    assert paths == ["sub/nested.jsonl", "top.jsonl"]

    override = KaggleOverride(include_globs=["**/*.jsonl", "*.txt"], text_column="prompt")
    items = list(iter_kaggle_dir(tmp_path, "o/d", None, override))
    assert [i["meta"]["path"] for i in items] == ["notes.txt", "sub/nested.jsonl", "top.jsonl"]
    assert items[1]["raw"] == "nested" and items[1]["meta"]["license"] == "UNKNOWN"


def test_kaggle_download_license_and_failure(tmp_path, monkeypatch):
    def fake_kaggle(args, env=None):
        if args[:2] == ["datasets", "download"]:
            target = Path(args[args.index("-p") + 1])
            with zipfile.ZipFile(target / "d.zip", "w") as z:
                z.writestr("train.csv", "text,label\nhello there,0\n")
        elif args[:2] == ["datasets", "metadata"]:
            workdir = Path(args[args.index("-p") + 1])
            (workdir / "dataset-metadata.json").write_text(json.dumps({"licenses": [{"name": "CC0-1.0"}]}))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(kaggle_mod, "_run_kaggle", fake_kaggle)
    items = list(iter_kaggle("o/d"))
    assert [(i["raw"], i["meta"]["license"]) for i in items] == [("hello there", "CC0-1.0")]

    monkeypatch.setattr(kaggle_mod, "_run_kaggle", lambda args, env=None: subprocess.CompletedProcess(args, 1, "", "403 Forbidden"))
    with pytest.raises(RuntimeError, match="403"):
        list(iter_kaggle("o/d"))


def test_kaggle_override_reaches_source(tmp_path, make_config, ingest, monkeypatch):
    seen = {}

    def fake_iter_kaggle(spec, override=None, username=None, key=None):
        seen[spec] = override
        return iter([])

    monkeypatch.setattr(pipeline_mod, "iter_kaggle", fake_iter_kaggle)
    ingest(make_config(kaggle=["o/d"], kaggle_overrides={"o/d": {"text_column": "prompt"}}), tmp_path / "out.jsonl")
    assert seen["o/d"].text_column == "prompt"


def test_unquoted_label_map_keys_end_to_end(tmp_path, ingest):
    write_jsonl(tmp_path / "a.jsonl", [
        {"text": "first sample text", "label": 1},
        {"text": "second sample text", "label": True},
        {"text": "third sample text", "label": "yes"},
        {"text": "fourth sample text", "label": 0},
    ])
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        f"local: ['{tmp_path / 'a.jsonl'}']\n"
        "allowed_languages: ['*']\nlanguage_confidence: 0.0\nmin_entropy: 0.0\nmin_length: 0\ncpu_workers: 1\n"
        f"state_dir: '{tmp_path / '.state'}'\n"
        "global_label_map:\n  1: malicious\n  true: malicious\n  yes: malicious\n  0: benign\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    ingest(load_config(cfg_path), out)
    assert [r["label"] for r in read_jsonl(out)] == ["malicious", "malicious", "malicious", "benign"]
