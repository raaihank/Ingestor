"""Shared constants across the ingestor modules."""

# File extensions for different data types
TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
STRUCTURED_EXTENSIONS = {
    ".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet", ".arrow"
}
CONFIG_EXTENSIONS = {".yaml", ".yml"}

# All supported file extensions
ALL_SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS | STRUCTURED_EXTENSIONS | CONFIG_EXTENSIONS
)
