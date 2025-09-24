from __future__ import annotations

from rich.console import Console

_VERBOSITY = 0  # 0,1,2
_console = Console()
_QUIET = False


def set_verbosity(level: int) -> None:
    global _VERBOSITY
    if level < 0:
        level = 0
    if level > 2:
        level = 2
    _VERBOSITY = level


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


def log_error(message: str) -> None:
    _console.print(message, style="red")


def log_debug(message: str) -> None:
    # Show on level >= 2
    if _VERBOSITY >= 2:
        _console.print(message, style="grey50")


def log_summary(approved: int, rejected: int) -> None:
    _console.print(f"Approved {approved}", style="green")
    _console.print(f"Rejected {rejected}", style="red")
