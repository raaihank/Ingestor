#!/usr/bin/env python3
import argparse
import os
import sys

import pandas as pd


def convert_jsonl(input_file: str, output_file: str, output_format: str):
    try:
        df = pd.read_json(input_file, lines=True)

        if output_format == "csv":
            df.to_csv(output_file, index=False)
        elif output_format == "parquet":
            df.to_parquet(output_file, engine="pyarrow", index=False)
        else:
            print(f"Unsupported format: {output_format}")
            sys.exit(1)

        print(f"Converted {input_file} -> {output_file} [{output_format}]")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL file to CSV or Parquet"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output file"
    )
    parser.add_argument(
        "--output-format",
        required=True,
        choices=["csv", "parquet"],
        help="Output format: csv or parquet"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file does not exist: {args.input}")
        sys.exit(1)

    if not args.input.lower().endswith(".jsonl"):
        print("Input file must have .jsonl extension")
        sys.exit(1)

    if args.output_format == "csv" and not args.output.lower().endswith(".csv"):
        print("Output file must have .csv extension for csv format")
        sys.exit(1)

    if args.output_format == "parquet" and not args.output.lower().endswith(".parquet"):
        print("Output file must have .parquet extension for parquet format")
        sys.exit(1)

    convert_jsonl(args.input, args.output, args.output_format)


if __name__ == "__main__":
    main()
