# Use cases

Ingestor turns labeled text from Hugging Face, Kaggle, Git and local files into one clean JSONL file. It maps every source's labels to one set, removes low-quality and duplicate rows, and records every decision it makes. This page shows what you can build with it. Each example is a config to start from: check it with `ingestor verify --config <file>` before a full run. Installation and every option are in the [README](README.md).

**LLM security**
- [Train a prompt-injection or jailbreak classifier](#train-a-prompt-injection-or-jailbreak-classifier)
- [Test your detector against disguised attacks](#test-your-detector-against-disguised-attacks)
- [Look up known attacks by hash](#look-up-known-attacks-by-hash)
- [Build a corpus you can use commercially](#build-a-corpus-you-can-use-commercially)

**General NLP**
- [Merge classification datasets with different label schemes](#merge-classification-datasets-with-different-label-schemes)
- [Clean your own text before labeling or fine-tuning](#clean-your-own-text-before-labeling-or-fine-tuning)
- [Keep test data out of your training set](#keep-test-data-out-of-your-training-set)

**Auditing datasets**
- [See what a dataset contains](#see-what-a-dataset-contains)
- [Find prompts that datasets label differently](#find-prompts-that-datasets-label-differently)
- [Measure overlap between datasets](#measure-overlap-between-datasets)
- [Grow a corpus over time](#grow-a-corpus-over-time)

[Where it doesn't fit](#where-it-doesnt-fit)

---

## LLM security

### Train a prompt-injection or jailbreak classifier

Public prompt-injection datasets each use their own column names (`text`, `prompt`, `user_input`) and label schemes (`0`/`1`, `jailbreak`/`benign`, `is_dangerous`), and many of them repeat each other's rows. Ingestor maps them all to one label set and drops the rows that appear in more than one dataset.

```yaml
hf:
  - deepset/prompt-injections                      # text, label (0/1)
  - qualifire/prompt-injections-benchmark          # text, label (jailbreak/benign)
  - Necent/llm-jailbreak-prompt-injection-dataset  # prompt, is_dangerous (0/1)

store_raw: true          # also keep the original text
allowed_languages: [en]
min_entropy: 1.5         # short attacks have low entropy
min_length: 5

hf_overrides:
  Necent/llm-jailbreak-prompt-injection-dataset:
    label_column: is_dangerous

global_label_map:
  1: malicious
  0: benign
  jailbreak: malicious
  benign: benign
```

```bash
ingestor run --config classifier.yaml --out data/classifier.jsonl
```

Each record has a `label` of `malicious` or `benign`, and `meta.dataset` names the dataset and split it came from. `normalized_text` is lowercased with whitespace collapsed; if your model needs the original casing, train on `raw`, which `store_raw: true` adds to each record.

### Test your detector against disguised attacks

Attackers disguise a known prompt by inserting invisible zero-width characters, adding BiDi controls that reverse how text displays, swapping letters for look-alikes from other alphabets, or wrapping the payload in base64 or hex. Ordinary deduplication treats these as copies and throws them away. Ingestor keeps them and tags each one with:

- `meta.evasion_type`: `zero_width`, `bidi_override`, `zero_width_bidi`, `homoglyph` or `encoding_wrap`
- `meta.evasion_variant_of`: the id of the record it disguises

```bash
jq -c 'select(.meta.evasion_type)' data/classifier.jsonl > data/evasion.jsonl
```

Run your detector on each variant and on its original. A variant that gets through while its original is caught shows a gap in your detector. Ingestor only finds variants that already exist in your sources; it doesn't generate new ones. This needs `preserve_evasion_variants: true`, which is the default.

### Look up known attacks by hash

`prompt_hash` is a SHA-256 of the text after heavy normalization: compatibility forms folded, look-alike letters mapped to ASCII and accents removed. Prompts that differ only by look-alike letters, full-width letters, accents or BiDi controls get the same hash, so a set of hashes gives you a fast first check for attacks you've already seen:

```python
import hashlib
import json

from ingestor.normalization import normalize_text_heavy

with open("data/classifier.jsonl") as f:
    known = {r["prompt_hash"] for r in map(json.loads, f) if r["label"] == "malicious"}

def is_known_attack(prompt: str) -> bool:
    digest = hashlib.sha256(normalize_text_heavy(prompt).encode("utf-8")).hexdigest()
    return digest in known
```

This only catches exact copies. The hash is case-sensitive, and a zero-width space (U+200B) inside a word becomes a real space, which changes the hash. To catch reworded attacks, embed `normalized_text` into a vector index instead.

### Build a corpus you can use commercially

```yaml
enforce_license: true

hf:
  - deepset/prompt-injections                    # license read from the dataset card
git:
  - https://github.com/owner/prompt-corpus.git   # license read from LICENSE / COPYING
local:
  - "data/in-house/*.jsonl"

local_overrides:
  "data/in-house/*.jsonl":
    license: MIT   # local files carry no license; declare it
```

Only data under MIT, Apache-2.0, BSD-3-Clause, CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0 or Unlicense is kept, and each record states its license in `meta.license`. Everything else is rejected with reason `license`. CC-BY data still requires attribution, and CC-BY-SA also requires sharing under the same license; `meta.license` shows you which records carry those terms.

## General NLP

### Merge classification datasets with different label schemes

Nothing in Ingestor is specific to security. Any single-label text task works, such as spam, toxicity, sentiment, intent or ticket routing. Map each source's labels to your own set:

```yaml
local:
  - "data/raw/sms_spam.csv"          # columns: v1 (ham/spam), v2 (message)
  - "data/raw/email_spam/*.jsonl"    # columns: text, is_spam (true/false)

allowed_languages: [en]

local_overrides:
  "data/raw/sms_spam.csv":
    text_column: v2
    label_column: v1
  "data/raw/email_spam/*.jsonl":
    label_column: is_spam

local_label_maps:
  "data/raw/sms_spam.csv":
    ham: not_spam
    spam: spam
  "data/raw/email_spam/*.jsonl":
    true: spam
    false: not_spam
```

A label with no mapping is kept, converted to `lowercase_with_underscores`. Text and label columns with common names (`text`, `prompt`, `label`, `class` and others listed in the README) are found without an override.

### Clean your own text before labeling or fine-tuning

Exports of support tickets, chat logs or scraped pages tend to contain repeated messages, other languages and junk rows. Ingestor removes those before you pay annotators or spend GPU time on them. Rows don't need a label; unlabeled rows come out with `"label": null`.

```yaml
local:
  - "exports/tickets/**/*.jsonl"

store_raw: true
allowed_languages: [en]
min_length: 20
max_length: 5000

local_overrides:
  "exports/tickets/**/*.jsonl":
    text_column: message_text   # without this, a row with no known text column has all its fields joined
```

### Keep test data out of your training set

Ingestor removes duplicates across every source and split in one output, and the first copy in config order is the one kept. List your evaluation data first, so its rows stay and their copies in the training data are dropped:

```yaml
local:
  - "data/eval/*.jsonl"    # listed first: all its rows are kept
  - "data/train/*.jsonl"   # rows copying an eval row are dropped
```

Then split the output back apart with `meta.dataset`, which holds the configured glob for local files and `owner/name:split` for Hugging Face:

```bash
jq -c 'select(.meta.dataset == "data/eval/*.jsonl")' data/all.jsonl > data/eval.jsonl
jq -c 'select(.meta.dataset == "data/train/*.jsonl")' data/all.jsonl > data/train.jsonl
```

This only works within one output. Runs with different `--out` paths keep separate state and don't see each other's rows.

## Auditing datasets

Every decision is stored in the output's state file, `.state/<output-name>-<hash>.sqlite`. The queries below use `sqlite3` on that file.

### See what a dataset contains

```bash
# Counts, label distribution and sample rows per dataset; writes nothing
ingestor verify --config my.config.yaml

# Label, source, dataset, category, license and length summary of an output
python scripts/insights.py --input data/unified.jsonl --hash-stats

# Why rows were rejected
sqlite3 .state/unified-*.sqlite \
  "SELECT reason, COUNT(*) FROM rejected GROUP BY reason ORDER BY 2 DESC;"
```

Rejection reasons are `empty`, `entropy`, `length`, `language`, `license`, `duplicate_exact` and `near_duplicate`. To see everything a source contains, run it with the filters off, using the [inspection config](README.md#keep-almost-everything-for-inspection) in the README.

### Find prompts that datasets label differently

When the same prompt appears in two datasets with different labels, the output keeps only the first copy's label. The duplicate log records both:

```sql
SELECT kept_id, dropped_id, reason, label_kept, label_dropped
FROM duplicate_log
WHERE label_kept <> label_dropped;
```

```text
kept_id                       dropped_id                    reason           label_kept  label_dropped
local:data/test/eval.jsonl:0  local:data/train/train.csv:0  exact_duplicate  malicious   benign
```

Each row is a likely labeling error in one of the two sources. This needs `enable_duplicate_logging: true`, which is the default.

### Measure overlap between datasets

```sql
SELECT source_kept, source_dropped, reason, COUNT(*) AS n
FROM duplicate_log
GROUP BY 1, 2, 3
ORDER BY n DESC;
```

For Hugging Face, Kaggle and Git, the source names the dataset (`hf:owner/name:split`, `kaggle:owner/name`, `git:<url>`). Local files all appear as `local`; for those, use `kept_id` and `dropped_id`, which contain the file path. A pair with a high count means one dataset largely repeats the other.

### Grow a corpus over time

Add a source to the config and run the same command again. Records already decided are skipped and the new source's records are appended. An interrupted run resumes the same way. That makes Ingestor safe to run on a schedule or in CI:

- The exit code is 1 if any source failed (the output is still written) and 2 if the config is invalid or lists no sources.
- After removing a source or changing filters, label maps or overrides, run with `--fresh`. Otherwise earlier decisions stand.

## Where it doesn't fit

- **Conversations.** Each row becomes one text. Multi-turn chats are flattened (their fields are joined, or dumped as JSON), so turns and roles are lost.
- **Multi-label or structured targets.** Each record has one label.
- **Non-text data** such as images or audio.
- **Generating data.** Ingestor doesn't create train/test splits, paraphrases or new evasion variants. It only keeps and tags what your sources contain.
- **Web-scale pretraining corpora.** State lives in a single SQLite file, and near-duplicate detection only compares against the most recent `near_dup_memory_limit` records (1,000,000 by default). Exact duplicates are still caught beyond that limit.
