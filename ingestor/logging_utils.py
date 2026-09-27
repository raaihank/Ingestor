from __future__ import annotations

import logging

from rich.console import Console

_VERBOSITY = 0  # 0,1,2
# soft_wrap: never insert line breaks into log lines (CI logs and redirected output)
_console = Console(soft_wrap=True)
_QUIET = False


def _suppress_huggingface_logging() -> None:
    """Suppress verbose HuggingFace datasets logging messages."""
    # Suppress datasets library verbose messages
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("datasets.builder").setLevel(logging.ERROR)
    logging.getLogger("datasets.info").setLevel(logging.ERROR)
    logging.getLogger("datasets.utils").setLevel(logging.ERROR)
    logging.getLogger("datasets.arrow_dataset").setLevel(logging.ERROR)
    logging.getLogger("datasets.dataset_dict").setLevel(logging.ERROR)


def set_verbosity(level: int) -> None:
    global _VERBOSITY
    if level < 0:
        level = 0
    if level > 2:
        level = 2
    _VERBOSITY = level

    # Always suppress HuggingFace verbose logging unless in debug mode
    if level < 2:
        _suppress_huggingface_logging()


def set_quiet(quiet: bool) -> None:
    global _QUIET
    _QUIET = quiet


def log_dataset(message: str) -> None:
    # Suppress during live spinners to avoid line interference
    if _QUIET:
        return
    _console.print(message, style="grey50")


def log_success(message: str) -> None:
    # Show on level >= 1
    if _VERBOSITY >= 1:
        _console.print(message, style="green")


def log_warning(message: str) -> None:
    _console.print(message, style="yellow", markup=False)


def log_error(message: str) -> None:
    _console.print(message, style="red", markup=False)


def log_debug(message: str) -> None:
    # Show on level >= 2
    if _VERBOSITY >= 2:
        _console.print(message, style="grey50")


def log_summary(approved: int, rejected: int, existing: int = 0) -> None:
    _console.print(f"Approved {approved}", style="green")
    _console.print(f"Rejected {rejected}", style="red")
    if existing:
        _console.print(f"Already ingested {existing}", style="grey50")
