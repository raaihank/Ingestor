from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import IngestConfig, load_config
from .logging_utils import log_error, log_summary, set_quiet, set_verbosity
from .pipeline import IngestPipeline, dataset_label
from .sources.huggingface import _parse_hf_spec

app = typer.Typer(add_completion=False, no_args_is_help=True)
log = structlog.get_logger()


def _describe(error: Exception) -> str:
    """One line per problem, naming the config key."""
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'config'}: {err['msg']}" for err in error.errors()
        )
    return str(error)


def _load_config_or_exit(path: Path) -> IngestConfig:
    try:
        return load_config(path)
    except (ValidationError, yaml.YAMLError, OSError) as e:
        log_error(f"Invalid config {path}: {_describe(e)}")
        raise typer.Exit(code=2) from None


def _apply_credentials(cfg: IngestConfig) -> None:
    token = cfg.hf_token or os.getenv("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    # The kaggle CLI reads these env vars; ~/.kaggle/kaggle.json is left untouched
    if cfg.kaggle_username and cfg.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = cfg.kaggle_username
        os.environ["KAGGLE_KEY"] = cfg.kaggle_key


def _tally(outcome: Dict[str, Any], counts: Dict[str, Dict[str, int]], order: List[str]) -> str:
    """Count a pipeline outcome under its dataset; returns the dataset key."""
    event = outcome.get("event")
    ds = str(outcome.get("dataset", "unknown")) if event else dataset_label(outcome)
    c = counts.setdefault(ds, {"approved": 0, "rejected": 0, "existing": 0})
    c[event or "approved"] += 1
    if ds not in order:
        order.append(ds)
    return ds


def _count_line(c: Dict[str, int]) -> str:
    line = f"approved: {c['approved']} rejected: {c['rejected']}"
    if c["existing"]:
        line += f" existing: {c['existing']}"
    return line


def _finish(pipeline: IngestPipeline, out: Path) -> None:
    log_summary(
        approved=pipeline.approved_count,
        rejected=pipeline.rejected_count,
        existing=pipeline.existing_count,
    )
    log.info("ingest_complete", total=pipeline.total_records, new=pipeline.approved_count, out=str(out))
    if pipeline.failed_sources:
        for spec, error in pipeline.failed_sources:
            log_error(f"Source failed: {spec}: {error}")
        raise typer.Exit(code=1)


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
    cfg: IngestConfig = _load_config_or_exit(config)
    _apply_credentials(cfg)

    pipeline = IngestPipeline(config=cfg)
    set_verbosity(2 if debug else cfg.verbose)
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
    _finish(pipeline, out)


@app.command("run")
def run(
    out: Path = typer.Option(..., help="Output JSONL path"),
    config: Optional[Path] = typer.Option(None, help="YAML config file (other flags override it)"),
    hf: List[str] = typer.Option([], help="HuggingFace dataset name", metavar="HF"),
    git: List[str] = typer.Option([], help="Git repo URLs", metavar="URL"),
    kaggle: List[str] = typer.Option([], help="Kaggle dataset spec", metavar="DATASET"),
    local: List[str] = typer.Option([], help="Local file glob (repeatable)", metavar="GLOB"),
    store_raw: Optional[bool] = typer.Option(
        None, "--store-raw/--no-store-raw", help="Include raw text in output"
    ),
    allowed_lang: List[str] = typer.Option([], help="Allowed languages (repeatable)"),
    language_confidence: Optional[float] = typer.Option(None, help="Language detection confidence"),
    enforce_license: Optional[bool] = typer.Option(
        None, "--enforce-license/--no-enforce-license", help="Reject items without approved licenses"
    ),
    hf_token: Optional[str] = typer.Option(None, help="Hugging Face token (or set HF_TOKEN env)"),
    kaggle_username: Optional[str] = typer.Option(None, help="Kaggle username (or KAGGLE_USERNAME env)"),
    kaggle_key: Optional[str] = typer.Option(None, help="Kaggle key (or KAGGLE_KEY env)"),
    io_workers: Optional[int] = typer.Option(None, help="Thread workers for IO stage (auto if omitted)"),
    cpu_workers: Optional[int] = typer.Option(None, help="Process workers for CPU stage (auto if omitted)"),
    batch_size: Optional[int] = typer.Option(None, help="Items per CPU batch (auto if omitted)"),
    state_dir: Optional[Path] = typer.Option(None, help="Directory for resumable state (default .state)"),
    fresh: bool = typer.Option(False, "--fresh", help="Discard this output's saved state and rebuild it"),
    debug: bool = typer.Option(False, "--debug", help="Enable detailed debug logs"),
):
    """Run ingestion from selected sources into a unified JSONL file."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])

    cfg = _load_config_or_exit(config) if config else IngestConfig()
    # Flags given on the command line override the config file
    updates: Dict[str, Any] = {}
    for key, extra in (("hf", hf), ("git", git), ("kaggle", kaggle), ("local", local)):
        if extra:
            updates[key] = [*getattr(cfg, key), *extra]
    if allowed_lang:
        updates["allowed_languages"] = allowed_lang
    flag_values = {
        "store_raw": store_raw,
        "language_confidence": language_confidence,
        "enforce_license": enforce_license,
        "hf_token": hf_token,
        "kaggle_username": kaggle_username,
        "kaggle_key": kaggle_key,
        "io_workers": io_workers,
        "cpu_workers": cpu_workers,
        "batch_size": batch_size,
        "state_dir": str(state_dir) if state_dir is not None else None,
    }
    updates.update({k: v for k, v in flag_values.items() if v is not None})
    if updates:
        try:
            # Re-validate so flag values get the same checks as the config file
            cfg = IngestConfig.model_validate({**cfg.model_dump(), **updates})
        except ValidationError as e:
            log_error(f"Invalid option: {_describe(e)}")
            raise typer.Exit(code=2) from None
    if not (cfg.hf or cfg.git or cfg.kaggle or cfg.local):
        log_error("No sources configured: pass --config or --hf/--git/--kaggle/--local")
        raise typer.Exit(code=2)

    _apply_credentials(cfg)

    pipeline = IngestPipeline(config=cfg)
    set_verbosity(2 if debug else cfg.verbose)
    is_tty = sys.stderr.isatty() and os.getenv("CI") not in ("1", "true", "True")
    dataset_counts: Dict[str, Dict[str, int]] = {}
    last_update: Dict[str, float] = {}
    dataset_order: List[str] = []
    current_ds: Optional[str] = None

    if is_tty:
        # Suppress dataset log lines during spinner rendering
        set_quiet(True)
        with Progress(
            SpinnerColumn(spinner_name="line", style="grey50", finished_text=""),
            TextColumn("{task.fields[status]}", justify="right"),
            TextColumn("{task.description}"),
            TextColumn(
                "[green]approved: {task.fields[approved]}[/green]  "
                "[red]rejected: {task.fields[rejected]}[/red]  "
                "[grey50]existing: {task.fields[existing]}[/grey50]"
            ),
            transient=True,
        ) as progress:
            # Keep a mapping Dataset -> TaskID for typed Progress.update
            from rich.progress import TaskID  # local import for typing
            tasks: Dict[str, TaskID] = {}
            for outcome in pipeline.run(out_path=out, fresh=fresh):
                ds = _tally(outcome, dataset_counts, dataset_order)
                if current_ds is None:
                    current_ds = ds
                elif ds != current_ds and current_ds in tasks:
                    # Stop animating the previous dataset line
                    progress.stop_task(tasks[current_ds])
                    current_ds = ds
                now = time.time()
                if now - last_update.get(ds, 0) < 0.1:
                    continue
                last_update[ds] = now
                if ds not in tasks:
                    tasks[ds] = progress.add_task(ds, approved=0, rejected=0, existing=0, status="")
                # Stop all other dataset tasks to avoid multiple spinning lines
                for other_ds, tid in list(tasks.items()):
                    if other_ds != ds and not progress.tasks[tid].finished:
                        # finalize other line with tick symbol now (no cross)
                        progress.update(tid, status="[green]\u2713[/green]")
                        progress.stop_task(tid)
                c = dataset_counts[ds]
                progress.update(
                    tasks[ds],
                    description=ds,
                    approved=c["approved"],
                    rejected=c["rejected"],
                    existing=c["existing"],
                    status="",
                )
            for ds, tid in tasks.items():
                progress.update(tid, status="[green]\u2713[/green]", refresh=True)
                progress.stop_task(tid)
        # After progress ends (transient), print final per-dataset summary lines
        set_quiet(False)
        console = Console()
        for ds in dataset_order:
            console.print("")
            console.print(f"{ds} \u2713 {_count_line(dataset_counts[ds])}")
    else:
        for outcome in pipeline.run(out_path=out, fresh=fresh):
            _tally(outcome, dataset_counts, dataset_order)
        console = Console()
        for ds in dataset_order:
            console.print(f"{ds} {_count_line(dataset_counts[ds])}")

    _finish(pipeline, out)


@app.command("verify")
def verify(
    config: Path = typer.Option(..., help="YAML config file"),
    per_dataset: int = typer.Option(50, help="Max samples to inspect per dataset"),
    debug: bool = typer.Option(False, "--debug", help="Enable detailed debug logs"),
):
    """Dry-run: preview columns/category, sample texts, and label distribution. Fails if columns missing (HF)."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    cfg: IngestConfig = _load_config_or_exit(config)
    set_verbosity(2 if debug else cfg.verbose)
    _apply_credentials(cfg)

    # HF column existence checks against overrides
    errors: list[str] = []
    for spec, ov in cfg.hf_overrides.items():
        if not (ov.text_column or ov.label_column):
            continue
        try:
            path, name, revision = _parse_hf_spec(spec)
            import datasets  # lazy

            loaded = datasets.load_dataset(path, name=name, split=ov.split, token=os.getenv("HF_TOKEN"), revision=revision)
            splits = {ov.split: loaded} if ov.split else dict(loaded)
        except Exception as e:
            errors.append(f"HF {spec}: could not load dataset to check columns: {e}")
            continue
        for split_name, ds in splits.items():
            cols = set(getattr(ds, "column_names", []))
            if ov.text_column and ov.text_column not in cols:
                errors.append(f"HF {spec}:{split_name}: text_column '{ov.text_column}' not in columns {sorted(cols)}")
            if ov.label_column and ov.label_column not in cols:
                errors.append(f"HF {spec}:{split_name}: label_column '{ov.label_column}' not in columns {sorted(cols)}")

    # Sample items via pipeline sources
    pipeline = IngestPipeline(config=cfg)
    seen_counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}
    samples: dict[str, list[str]] = {}

    for source in pipeline.iter_source_specs():
        try:
            for item in source.open():
                item.setdefault("spec", source.spec)
                dataset_id = dataset_label(item)
                count = seen_counts.get(dataset_id, 0)
                if count >= per_dataset:
                    continue
                seen_counts[dataset_id] = count + 1

                # normalized preview
                text = str(item.get("raw", ""))
                label = pipeline.map_label(item) or "<none>"
                label_counts.setdefault(dataset_id, {})[label] = label_counts.get(dataset_id, {}).get(label, 0) + 1
                if dataset_id not in samples:
                    samples[dataset_id] = []
                if len(samples[dataset_id]) < 3:
                    samples[dataset_id].append(text[:240].replace("\n", " "))
        except Exception as e:
            errors.append(f"{source.kind} {source.spec}: {e}")

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
            console.print(f"  [{i+1}] {s}", style="grey50", markup=False)

    # Fail if strict errors collected
    if errors:
        for error in errors:
            console.print(error, style="red", markup=False)
        raise typer.Exit(code=1)


@app.command("tune")
def tune(
    sample: Optional[Path] = typer.Option(
        None,
        help="Optional JSONL file to sample for estimating average record size",
    ),
    top_n: int = typer.Option(200, help="Number of lines to sample from JSONL if provided"),
    target_batch_bytes: int = typer.Option(
        2 * 1024 * 1024, help="Target bytes per batch for throughput (default ~2MB)"
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON suggestions"),
):
    """Suggest optimal io-workers, cpu-workers, and batch size for this machine."""
    cores = os.cpu_count() or 1
    io_workers = min(32, max(4, cores * 4))
    cpu_workers = max(1, cores - 1)

    avg_bytes: Optional[float] = None
    if sample and sample.exists():
        try:
            total = 0
            lines = 0
            with sample.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    total += len(line)
                    lines += 1
                    if lines >= top_n:
                        break
            if lines > 0:
                avg_bytes = total / lines
        except Exception:
            avg_bytes = None

    def clamp(n: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, n))

    if avg_bytes and avg_bytes > 0:
        batch_size = int(target_batch_bytes / avg_bytes)
    else:
        batch_size = 256
    batch_size = clamp(batch_size, 64, 1024)

    suggestion = {
        "os": platform.system().lower(),
        "cpu_cores": cores,
        "io_workers": io_workers,
        "cpu_workers": cpu_workers,
        "batch_size": batch_size,
        "target_batch_bytes": target_batch_bytes,
        "avg_record_bytes": (avg_bytes or None),
    }

    if json_out:
        try:
            import json as _json

            typer.echo(_json.dumps(suggestion))
            return
        except Exception:
            pass

    typer.echo(
        f"OS={suggestion['os']} cores={cores} \n"
        f"io-workers={io_workers} cpu-workers={cpu_workers} batch-size={batch_size}"
    )


if __name__ == "__main__":
    app()
