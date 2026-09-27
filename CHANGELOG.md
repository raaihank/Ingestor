# Changelog

All notable changes to Ingestor are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release can include
breaking changes; they are listed under **Changed (breaking)**.

## [0.2.1] - 2026-09-27

### Added

- A "Config examples" section in the README with ready-to-use configs: local files with your own
  columns, a prompt-injection corpus from Hugging Face, mixed sources, a license-clean corpus,
  keeping almost everything for inspection, and tuning de-duplication and speed.

### Fixed

- The README's "Minimal YAML" example failed to load when copied, because of its
  `hf_token: ***` placeholder.

## [0.2.0] - 2026-09-27

This release makes resuming, de-duplication and the configuration work as documented.

### Changed (breaking)

- Each output file has its own state file under `.state/` that stores every accepted record and every
  rejection decision. The JSONL file is written from it when a run finishes. The old
  `.state/ingest.sqlite` and `.state/near_dup_sigs.sqlite` files are no longer used.
- `normalized_text` is now NFC-normalized, lowercased text with whitespace collapsed; zero-width and
  BiDi characters are kept. It used to be transliterated to ASCII. Use `--store-raw` to keep the
  original text as well.
- Exact duplicates are removed across all sources; before, the same text was only dropped when it
  came from the same row.
- Git sources read only structured data files, and Kaggle does too unless `include_globs` says
  otherwise. README, config and other text files were being ingested as samples.
- Unknown keys in a config file are errors instead of being silently ignored.
- `ingestor run` exits with code 1 if a source failed (after writing the output) and with code 2 if
  no sources are configured.
- Requires `datasketch` 2.x.

### Added

- Resuming: rerun an interrupted or crashed run and the output ends up identical to an
  uninterrupted run. `--fresh` rebuilds an output from scratch and `--state-dir` (or `state_dir`)
  sets where state is kept.
- Near-duplicate detection now actually runs; its signatures are saved and reused across runs.
- Evasion variants of a kept text (zero-width or BiDi characters, look-alike letters, encoded
  payloads) are kept and marked with `meta.evasion_variant_of` and `meta.evasion_type`.
- A `license` setting in overrides. Licenses are read from Hugging Face dataset cards, Kaggle
  metadata and Git `LICENSE` files, and license names are matched in any spelling
  (`apache-2.0`, `Apache 2.0`, `CC0: Public Domain`).
- These settings were accepted before but had no effect; they now work: `local_overrides`,
  `local_label_maps`, Kaggle `text_column` / `label_column` / `include_globs`,
  `near_duplicate_threshold` (one fixed threshold), `preserve_evasion_variants`,
  `enable_duplicate_logging`, `fasttext_lid_path` and `verbose`.
- `ingestor run` gains `--local`, `--state-dir`, `--fresh`, `--no-store-raw` and
  `--no-enforce-license`; options given on the command line now override the config file.
- Hugging Face datasets that the `datasets` library can't load are read file by file instead.
- Language detection and near-duplicate signatures are computed in parallel worker processes.

### Fixed

- Records were lost when a run was interrupted and resumed, when the same config was run with a
  new `--out` path, or when a source was added and the output was overwritten with only new records.
- Stopping a run early (for example with Ctrl-C) could hang.
- Results no longer depend on download timing or on langdetect's randomness, so repeated runs give
  the same output.
- A dataset column named `dataset` (as in HackAPrompt) no longer breaks category overrides and
  label maps for that dataset.
- When no text column was found, the label value was copied into the sample text.
- CSV files with a byte-order mark or fields over 128 KB, JSONL files with a non-object line, and
  integer Parquet labels with missing values were read incorrectly or silently skipped.
- A config with an empty override entry (`owner/dataset:` with nothing below it) failed to load.
- Labels given as booleans or floats (`true`, `1.0`) now match label maps.
- `ingestor verify` crashed on samples containing text like `[/INST]`, and it no longer writes state.
- Metadata values that JSON can't represent (such as decimals) crashed the final write.
- The bundled demo read its own previous output back in as input.
- Failed sources are reported instead of silently producing no rows.
- `ingestor run` no longer overwrites `~/.kaggle/kaggle.json`.

### Security

- `prompt_security.yaml` and `*.config.yaml` files are ignored by git, since they can contain
  access tokens.

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

[0.2.1]: https://github.com/raaihank/Ingestor/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/raaihank/Ingestor/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/raaihank/Ingestor/releases/tag/v0.1.0
