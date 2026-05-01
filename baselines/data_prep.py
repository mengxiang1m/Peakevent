from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASET_SPECS = {
    "wlel": {
        "series_csv": "dataset/wlel/event_v1/data/wlel_event_series_v1.csv",
        "events_jsonl": "dataset/wlel/event_v1/data/wlel_events_v1.jsonl",
        "split_json": "dataset/wlel/event_v1/data/split_indices_v1.json",
    },
    "ett": {
        "series_csv": "dataset/ett/event_v1/data/ett_event_series_v1.csv",
        "events_jsonl": "dataset/ett/event_v1/data/ett_events_v1.jsonl",
        "split_json": "dataset/ett/event_v1/data/split_indices_v1.json",
    },
    "elc": {
        "series_csv": "dataset/electricity/event_v1/data/elc_event_series_v1.csv",
        "events_jsonl": "dataset/electricity/event_v1/data/elc_events_v1.jsonl",
        "split_json": "dataset/electricity/event_v1/data/split_indices_v1.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare STSEP comparison datasets")
    parser.add_argument("--seq_len", type=int, default=96, help="history length")
    parser.add_argument("--window_stride", type=int, default=4, help="stride for aligned windows")
    parser.add_argument(
        "--pred_lens",
        type=str,
        default="96,168,336",
        help="comma-separated prediction lengths",
    )
    return parser.parse_args()


def _window_count(start: int, end: int, seq_len: int, pred_len: int, stride: int) -> int:
    max_start = end + 1 - seq_len - pred_len
    if max_start < start:
        return 0
    offset = (stride - (start % stride)) % stride
    first = start + offset
    if first > max_start:
        return 0
    return (max_start - first) // stride + 1


def _split_summary(split_json_path: Path) -> dict:
    with open(split_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    summary = {}
    for split_key, ids_key in [("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids")]:
        ids = raw[ids_key]
        summary[split_key] = {
            "start": int(ids[0]),
            "end": int(ids[-1]),
            "count": int(len(ids)),
        }
    return summary


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    out_data_dir = here / "data"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    pred_lens = [int(x.strip()) for x in args.pred_lens.split(",") if x.strip()]
    config = {
        "version": "comparison_v1",
        "seq_len_default": int(args.seq_len),
        "window_stride_default": int(args.window_stride),
        "pred_lens_default": pred_lens,
        "datasets": {},
    }

    for name, spec in DATASET_SPECS.items():
        src_series = repo_root / spec["series_csv"]
        src_events = repo_root / spec["events_jsonl"]
        src_split = repo_root / spec["split_json"]

        df = pd.read_csv(src_series)
        if "timestamp" not in df.columns or "value" not in df.columns:
            raise KeyError(f"{src_series} must contain `timestamp` and `value` columns")
        out_df = df[["timestamp", "value"]].rename(columns={"timestamp": "date", "value": "OT"})
        out_csv = out_data_dir / f"{name}.csv"
        out_df.to_csv(out_csv, index=False)

        splits = _split_summary(src_split)
        test_cfg = splits["test"]
        window_counts = {
            f"pred_{pred_len}": _window_count(
                start=test_cfg["start"],
                end=test_cfg["end"],
                seq_len=args.seq_len,
                pred_len=pred_len,
                stride=args.window_stride,
            )
            for pred_len in pred_lens
        }

        config["datasets"][name] = {
            "csv": f"data/{name}.csv",
            "events_jsonl": str(Path("..") / spec["events_jsonl"]),
            "source_series_csv": str(Path("..") / spec["series_csv"]),
            "source_split_json": str(Path("..") / spec["split_json"]),
            "target": "OT",
            "date_col": "date",
            "length": int(len(out_df)),
            "splits": splits,
            "window_counts": window_counts,
            "event_meta": {
                "min_duration": 3,
                "max_flank_hours": 6,
                "tolerance": 3,
            },
        }

        print(
            f"[{name}] rows={len(out_df)} train={splits['train']['count']} "
            f"val={splits['val']['count']} test={splits['test']['count']}"
        )

    cfg_path = out_data_dir / "dataset_configs.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {cfg_path}")


if __name__ == "__main__":
    main()
