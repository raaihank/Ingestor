<!-- Badges -->
[![Build](https://github.com/raaihank/ingestor/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/raaihank/ingestor/actions/workflows/build.yml)
[![Test](https://github.com/raaihank/ingestor/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/raaihank/ingestor/actions/workflows/test.yml)
[![Security](https://github.com/raaihank/ingestor/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/raaihank/ingestor/actions/workflows/security.yml)

# Ingestor
Fast, reproducible dataset ingestion for LLM security and general NLP. Pull from Hugging Face, Kaggle, Git, or local folders; label normalization, filter, dedupe, and write atomic JSONL.

### Use cases

- Build a unified security/classification corpus from HF/Kaggle/Git/local
- Normalize labels to a canonical set (e.g., malicious/benign)
- Enforce language and license rules
- Remove low‑quality and near‑duplicate samples
- Produce atomic JSONL for downstream training/indexing

### Install

```bash
pip install -e .[dev]
```

### Quick start (demo)

```bash
ingestor test
cat test-data/unified.sample.jsonl | head -n 3
```

### Minimal YAML

```yaml
# Auth
hf_token: *******************
# Models
hf:
  - deepset/prompt-injections

store_raw: false
allowed_languages: [en]
language_confidence: 0.7
enforce_license: true

# Normalize labels to malicious/benign
global_label_map:
  "1": malicious
  "0": benign
  malicious: malicious
  benign: benign
```

### Verify (dry-run)

```bash
ingestor verify --config my.config.yaml
```

Shows sample counts, label distribution, previews; fails if HF override columns are missing.

### Run

```bash
ingestor run --config my.config.yaml --out data/unified.jsonl
```

### Auth

```bash
export HF_TOKEN=hf_...       # Hugging Face (accept gated terms in web UI once)
export KAGGLE_USERNAME=...   # Kaggle
export KAGGLE_KEY=...
```

### Output

Atomic JSONL with: `id`, `source`, `source_id`, `normalized_text`, `prompt_hash`, `label`, `meta` (and optional `raw`).

### Commands

- `ingestor run` — ingest sources to JSONL (flags: `--config`, `--out`, `--store-raw`, `--allowed-lang`, `--language-confidence`, `--enforce-license`, `--hf-token`, `--kaggle-username`, `--kaggle-key`, `--debug`)
- `ingestor verify` — dry‑run preview (flags: `--config`, `--per-dataset`, `--debug`)
- `ingestor test` — demo on bundled `test-data/`
- `ingestor version` — show version

### In‑depth configuration

- Sources
  - `hf`: list of HF datasets (all splits ingested by default)
  - `kaggle`: list of Kaggle dataset refs
  - `git`: list of Git repo URLs (recurse and ingest data files)
  - `local`: list of filesystem globs (supports `**` recursion)

- Overrides (per-source)
  - `*_overrides.<id>.text_column` — pick text field when auto-detect is wrong
  - `*_overrides.<id>.label_column` — pick label field
  - `*_overrides.<id>.category` — annotate category into `meta.category`
  - `kaggle_overrides.<id>.include_globs` — restrict to data files

- Label normalization
  - `global_label_map` maps raw → canonical (e.g., "1" → `malicious`)
  - `hf_label_maps` / `kaggle_label_maps` / `local_label_maps` override per dataset

- Quality thresholds
  - `min_entropy`, `min_length`, `max_length`, `near_duplicate_threshold`

- Language detection
  - `allowed_languages`, `language_confidence`, `fasttext_lid_path` (optional heavier model)

### Logging UX

- Default: one spinner line per active dataset, updated in place with green approved / red rejected counts
- Non‑TTY/CI: plain final lines without spinner
- `--debug`: structured debug logs to stderr as NDJSON, in addition to summaries

### Troubleshooting

- HF gated datasets: accept terms once in web UI, then set `HF_TOKEN`
- Kaggle: set `KAGGLE_USERNAME`/`KAGGLE_KEY` (or `~/.kaggle/kaggle.json` with 600 perms)
- Missing columns: use `*_overrides` to set `text_column`/`label_column`, then re‑run `ingestor verify`

Default logging shows a spinner per dataset with green approved/red rejected counts. Add `--debug` for detailed logs.

### Advanced config

Optional per-dataset overrides:

```yaml
hf_overrides:
  "deepset/prompt-injections":
    text_column: prompt
    label_column: label
    category: prompt_injection
kaggle_overrides:
  "owner/dataset":
    include_globs: ["**/*.jsonl","**/*.csv","**/*.tsv","**/*.parquet","**/*.arrow"]
    text_column: text
    label_column: label
    category: prompt_injection
local_overrides:
  "/data/security/**":
    text_column: text
    label_column: label
    category: prompt_injection

# Dataset-specific label maps (override global)
hf_label_maps: {}
kaggle_label_maps: {}
local_label_maps: {}

# Quality thresholds
min_entropy: 2.5
min_length: 10
max_length: 10000
near_duplicate_threshold: 0.85

# Language detection
fasttext_lid_path: null
```

### Processing flow

```mermaid
flowchart TB
  A["HF Datasets"] --> B
  A2["HF Repo Crawl"] --> B
  K["Kaggle"] --> B
  G["Git"] --> B
  L["Local"] --> B
  B["Source Readers<br/>(stream/recursive)"] --> C["Normalize"]
  C --> D["Quality Filters<br/>entropy/length/language"]
  D --> E{"Near-dup?"}
  E -->|yes| X["Drop"]
  E -->|no| F["Label Map<br/>+ Category"]
  F --> S["State Store<br/>SQLite (batched)"]
  S --> W["Writer<br/>(orjson JSONL, atomic)"]
  W --> O["unified.jsonl"]
```

#### How it works

- Source readers: Stream HF/Kaggle splits or crawl Git/local paths, parse supported file types, and attach `meta.dataset`/`meta.split` (and license when available).
- Normalize: Apply deterministic text normalization (Unicode NFKC, homoglyph mapping, transliteration, whitespace collapse).
- Quality filters: Reject by entropy, length bounds, and language detection with configurable thresholds.
- Near-duplicate check: MinHash LSH flags highly similar prompts; flagged items are dropped.
- Label + category: Map raw labels to a canonical set and annotate `meta.category` using global or per-dataset maps.
- State store: SQLite with WAL + batching ensures idempotency/dedup by `(source, source_id, prompt_hash)`.
- Writer: Batch-serialize with orjson and `writelines` to a temp file, then atomically replace the target JSONL.

