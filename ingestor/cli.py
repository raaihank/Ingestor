from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import structlog
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import IngestConfig, load_config
from .logging_utils import log_summary, set_verbosity
from .pipeline import IngestPipeline
from .sources.huggingface import _parse_hf_spec

app = typer.Typer(add_completion=False, no_args_is_help=True)
log = structlog.get_logger()


@app.command("version")
def version() -> None:
    """Show package version."""
    try:
        from importlib.metadata import version as get_version

        typer.echo(get_version("ingestor"))
    except Exception:
        typer.echo("unknown")


@app.command("test")
def test(
    config: Path = typer.Option(Path("test-data/sample.config.yaml"), help="Sample config YAML"),
    out: Path = typer.Option(Path("test-data/unified.sample.jsonl"), help="Output JSONL path"),
    debug: bool = typer.Option(False, "--debug", help="Enable detailed debug logs"),
):
    """Run a demo ingest using bundled test data to showcase the pipeline."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    cfg: IngestConfig = load_config(config)

    pipeline = IngestPipeline(config=cfg)
    set_verbosity(2 if debug else 0)
    is_tty = sys.stderr.isatty() and os.getenv("CI") not in ("1", "true", "True")
    if is_tty:
        with Progress(SpinnerColumn(spinner_name="line", style="grey50"), TextColumn("{task.description}")) as progress:
            task = progress.add_task("Ingesting (demo)")
            for _ in pipeline.run(out_path=out):
                pass
            progress.update(task, description="Ingesting (demo) [green]\u2713[/green]")
    else:
        for _ in pipeline.run(out_path=out):
            pass
    log_summary(approved=pipeline.approved_count, rejected=pipeline.rejected_count)
    log.info("ingest_complete", total=pipeline.approved_count, out=str(out))


@app.command("run")
def run(
    out: Path = typer.Option(..., help="Output JSONL path"),
    config: Optional[Path] = typer.Option(None, help="YAML config file"),
    hf: List[str] = typer.Option([], help="HuggingFace dataset name", metavar="HF"),
    git: List[str] = typer.Option([], help="Git repo URLs", metavar="URL"),
    kaggle: List[str] = typer.Option([], help="Kaggle dataset spec", metavar="DATASET"),
    store_raw: bool = typer.Option(
        False, "--store-raw", is_flag=True, help="Include raw text in output"
    ),
    allowed_lang: List[str] = typer.Option([], help="Allowed languages (repeatable)"),
    language_confidence: float = typer.Option(0.7, help="Language detection confidence"),
    enforce_license: bool = typer.Option(
        False, "--enforce-license", is_flag=True, help="Reject items without approved licenses"
    ),
    hf_token: Optional[str] = typer.Option(None, help="Hugging Face token (or set HF_TOKEN env)"),
    kaggle_username: Optional[str] = typer.Option(None, help="Kaggle username (or KAGGLE_USERNAME env)"),
    kaggle_key: Optional[str] = typer.Option(None, help="Kaggle key (or KAGGLE_KEY env)"),
    debug: bool = typer.Option(False, "--debug", help="Enable detailed debug logs"),
):
    """Run ingestion from selected sources into a unified JSONL file."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])

    if config:
        cfg: IngestConfig = load_config(config)
    else:
        cfg = IngestConfig.from_cli(
            hf=hf,
            git=git,
            kaggle=kaggle,
            local=[],
            store_raw=store_raw,
            allowed_languages=allowed_lang or None,
            language_confidence=language_confidence,
            enforce_license=enforce_license,
            hf_token=hf_token or os.getenv("HF_TOKEN"),
            kaggle_username=kaggle_username or os.getenv("KAGGLE_USERNAME"),
            kaggle_key=kaggle_key or os.getenv("KAGGLE_KEY"),
        )

    # Apply HF token to environment for datasets/hf hub
    token = cfg.hf_token or os.getenv("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

    # Configure Kaggle credentials if provided
    if cfg.kaggle_username and cfg.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = cfg.kaggle_username
        os.environ["KAGGLE_KEY"] = cfg.kaggle_key
        try:
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)
            kaggle_json = kaggle_dir / "kaggle.json"
            import json

            content = {"username": cfg.kaggle_username, "key": cfg.kaggle_key}
            kaggle_json.write_text(json.dumps(content), encoding="utf-8")
            kaggle_json.chmod(0o600)
        except Exception:
            pass

    pipeline = IngestPipeline(config=cfg)
    set_verbosity(2 if debug else 0)
    is_tty = sys.stderr.isatty() and os.getenv("CI") not in ("1", "true", "True")
    dataset_counts: dict[str, dict[str, int]] = {}
    last_update: dict[str, float] = {}
    overall_approved = 0
    overall_rejected = 0

    def get_dataset_id(o: dict) -> str:
        meta = o.get("meta", {}) if isinstance(o, dict) else {}
        return str(meta.get("dataset") or o.get("source") or "unknown")

    if is_tty:
        with Progress(
            SpinnerColumn(spinner_name="line", style="grey50"),
            TextColumn("{task.description}"),
            TextColumn("[green]approved={task.fields[approved]}[/green] [red]rejected={task.fields[rejected]}[/red]"),
            transient=True,
        ) as progress:
            # Keep a mapping Dataset -> TaskID for typed Progress.update
            from rich.progress import TaskID  # local import for typing
            tasks: dict[str, TaskID] = {}
            for outcome in pipeline.run(out_path=out):
                if isinstance(outcome, dict) and outcome.get("event") == "rejected":
                    ds = outcome.get("dataset", "unknown")
                    c = dataset_counts.setdefault(ds, {"approved": 0, "rejected": 0})
                    c["rejected"] += 1
                    overall_rejected += 1
                elif isinstance(outcome, dict):
                    ds = get_dataset_id(outcome)
                    c = dataset_counts.setdefault(ds, {"approved": 0, "rejected": 0})
                    c["approved"] += 1
                    overall_approved += 1
                else:
                    continue
                now = time.time()
                lu = last_update.get(ds, 0)
                if now - lu < 0.1:
                    continue
                last_update[ds] = now
                if ds not in tasks:
                    tasks[ds] = progress.add_task(f"[{ds}]", approved=0, rejected=0)
                progress.update(tasks[ds], description=f"[{ds}]", approved=dataset_counts[ds]["approved"], rejected=dataset_counts[ds]["rejected"])
            for ds, tid in tasks.items():
                symbol = "[green]\u2713[/green]" if dataset_counts.get(ds, {}).get("rejected", 0) == 0 else "[red]\u2717[/red]"
                progress.update(tid, description=f"[{ds}] {symbol}", refresh=True)
    else:
        for outcome in pipeline.run(out_path=out):
            if isinstance(outcome, dict) and outcome.get("event") == "rejected":
                overall_rejected += 1
            elif isinstance(outcome, dict):
                overall_approved += 1
        for ds, c in dataset_counts.items():
            Console().print(f"[{ds}] \u2713 approved: {c['approved']} rejected: {c['rejected']}")

    log_summary(approved=overall_approved, rejected=overall_rejected)
    log.info("ingest_complete", total=overall_approved, out=str(out))


@app.command("verify")
def verify(
    config: Path = typer.Option(..., help="YAML config file"),
    per_dataset: int = typer.Option(50, help="Max samples to inspect per dataset"),
    debug: bool = typer.Option(False, "--debug", help="Enable detailed debug logs"),
):
    """Dry-run: preview columns/category, sample texts, and label distribution. Fails if columns missing (HF)."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    set_verbosity(2 if debug else 0)
    cfg: IngestConfig = load_config(config)

    # Auth envs
    if cfg.hf_token or os.getenv("HF_TOKEN"):
        token = cfg.hf_token or os.getenv("HF_TOKEN")
        os.environ["HF_TOKEN"] = token  # type: ignore
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token  # type: ignore
    if cfg.kaggle_username and cfg.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = cfg.kaggle_username
        os.environ["KAGGLE_KEY"] = cfg.kaggle_key

    # HF column existence checks against overrides
    errors: list[str] = []
    for spec, ov in cfg.hf_overrides.items():
        try:
            path, name, revision = _parse_hf_spec(spec)
            import datasets  # lazy

            ds = datasets.load_dataset(path, name=name, split=ov.split or "train", token=os.getenv("HF_TOKEN"), revision=revision)
            cols = set(getattr(ds, "column_names", []))
            if ov.text_column and ov.text_column not in cols:
                errors.append(f"HF {spec}: text_column '{ov.text_column}' not in columns {sorted(cols)}")
            if ov.label_column and ov.label_column not in cols:
                errors.append(f"HF {spec}: label_column '{ov.label_column}' not in columns {sorted(cols)}")
        except Exception:
            # Skip strict HF HEAD checks on failure; pipeline will still ingest heuristically
            pass

    # Sample items via pipeline sources
    pipeline = IngestPipeline(config=cfg)
    seen_counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}
    samples: dict[str, list[str]] = {}

    def normalize_label(dataset_id: str, raw_label: object) -> str:
        if raw_label is None:
            return "<none>"
        s = str(raw_label)
        mapped = None
        if dataset_id in cfg.hf_label_maps:
            mapped = cfg.hf_label_maps[dataset_id].get(s)
        if mapped is None and dataset_id in cfg.kaggle_label_maps:
            mapped = cfg.kaggle_label_maps[dataset_id].get(s)
        if mapped is None:
            mapped = cfg.global_label_map.get(s)
        return mapped or s

    for item in pipeline._iter_sources():  # type: ignore[attr-defined]
        meta = item.get("meta", {})
        dataset_id = str(meta.get("dataset") or item.get("source", "unknown"))
        count = seen_counts.get(dataset_id, 0)
        if count >= per_dataset:
            continue
        seen_counts[dataset_id] = count + 1

        # normalized preview
        text = str(item.get("raw", ""))
        label = normalize_label(dataset_id, item.get("label"))
        label_counts.setdefault(dataset_id, {})[label] = label_counts.get(dataset_id, {}).get(label, 0) + 1
        if dataset_id not in samples:
            samples[dataset_id] = []
        if len(samples[dataset_id]) < 3:
            samples[dataset_id].append(text[:240].replace("\n", " "))

    # Print summary
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("Verification summary", style="grey50")
    table = Table(title="Datasets and label distribution")
    table.add_column("Dataset")
    table.add_column("Samples")
    table.add_column("Labels (count)")
    for ds, n in sorted(seen_counts.items(), key=lambda x: x[0]):
        dist = label_counts.get(ds, {})
        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items(), key=lambda x: -x[1])) or "<none>"
        table.add_row(ds, str(n), dist_str)
    console.print(table)

    # Show top-k samples per dataset
    for ds, lst in samples.items():
        console.print(f"[grey50]Samples for {ds}[/grey50]")
        for i, s in enumerate(lst):
            console.print(f"  [{i+1}] {s}", style="grey50")

    # Fail if strict errors collected
    if errors:
        for e in errors:
            console.print(e, style="red")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()

