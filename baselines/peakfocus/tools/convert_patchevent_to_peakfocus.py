#!/usr/bin/env python3
"""Convert PatchEvent event_series CSV into PeakFocus mixed CSV format."""

import argparse
import os
import pandas as pd


REQUIRED_INPUT_COLS = [
    "timestamp",
    "value",
    "is_peak",
]

OUTPUT_COLS = [
    "date_60min",
    "value_60min",
    "date_max",
    "value_max",
    "is_peak",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PatchEvent event_series CSV to PeakFocus mixed CSV"
    )
    parser.add_argument("--input", required=True, help="Path to *_event_series_v1.csv")
    parser.add_argument("--output", required=True, help="Path to output mixed CSV")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["wlel", "ett", "elc"],
        help="Dataset name for logging",
    )
    return parser.parse_args()


def validate_input_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_INPUT_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def convert(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date_60min": df["timestamp"],
            "value_60min": df["value"],
            "date_max": df["timestamp"],
            "value_max": df["value"],
            "is_peak": df["is_peak"],
        }
    )
    return out[OUTPUT_COLS]


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    validate_input_columns(df)

    converted = convert(df)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    converted.to_csv(args.output, index=False)

    peak_sum = int(converted["is_peak"].sum())
    print(f"[convert] dataset={args.dataset}")
    print(f"[convert] input={args.input}")
    print(f"[convert] output={args.output}")
    print(f"[convert] rows={len(converted)}")
    print(f"[convert] is_peak_sum={peak_sum}")


if __name__ == "__main__":
    main()
