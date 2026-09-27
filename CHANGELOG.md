# Changelog

All notable changes to Ingestor are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release can include
breaking changes; they are listed under **Changed (breaking)**.

## [0.1.0] - 2025-10-06

### Added

- `ingestor` command line with `run`, `verify`, `test`, `tune` and `version`.
- Sources: Hugging Face datasets (all splits by default), Kaggle datasets, Git repositories and
  local files (JSONL, JSON, CSV/TSV, Parquet, Arrow and plain text).
- Per-dataset overrides for the text column, label column, category and split.
- Label normalization with a global label map and per-dataset maps; labels are written as
  `lowercase_with_underscores`.
- Quality filters for entropy, text length, language (fastText or langdetect) and license.
- Exact de-duplication tracked in a SQLite state directory (`.state/`).
- Parallel loading and processing, and atomic JSONL output.
- CI workflows for build, tests and security scans.

[0.1.0]: https://github.com/raaihank/Ingestor/releases/tag/v0.1.0
