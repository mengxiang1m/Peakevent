"""
Walk every results/*/experiment_log.txt, extract training-time and final-metric
info into a tidy CSV (one row per setting/seed).

Usage:
    python tools/parse_epoch_time.py \
        --results_root results \
        --out paper_data/parsed_logs.csv
"""
from __future__ import annotations

import argparse
import os
import re
import csv
from datetime import datetime
from pathlib import Path

# Regexes for every field we want to extract
RE_EPOCH_COST = re.compile(r"<TIME> Epoch cost:\s*([\d.]+)s")
RE_PARAMS = re.compile(r"\[PARAMS\] Trainable:\s*([\d,]+)")
RE_MEM    = re.compile(r"\[MEMORY\] Peak GPU memory during train:\s*([\d.]+) MB")
RE_INFER  = re.compile(r"\[INFER\] Test inference:.*per_sample=([\d.]+)ms")
RE_START  = re.compile(r"Start Time:\s*(\S+\s+\S+)")
# Different tasks write different "done" strings — accept any of them.
RE_COMPLETED = re.compile(
    r"TRAINING AND TESTING COMPLETED|TESTING COMPLETED"
)

# Final test metrics block. The file is timestamped so each line starts with
# "[HH:MM:SS] " — we look for the metric label as a whole word anywhere on the line.
def _lbl(label: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(label)}:\s*([\d.]+)")

RE_MAE = _lbl("MAE")
RE_MSE = _lbl("MSE")
RE_RMSE = _lbl("RMSE")
RE_BPE = re.compile(r"Balanced_Peak_Error \(BPE\):\s*([\d.]+)")
RE_PIM = re.compile(r"Peak_Integrated_Metric \(PIM\):\s*([\d.]+)")
RE_F1  = re.compile(r"F1-Score:\s*([\d.]+)")
RE_P   = _lbl("Precision")
RE_R   = _lbl("Recall")
RE_TP_MSE = _lbl("TP_MSE")
RE_TP_MAE = _lbl("TP_MAE")


def extract(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")

    epochs = [float(x) for x in RE_EPOCH_COST.findall(text)]
    avg_epoch = sum(epochs) / len(epochs) if epochs else None
    total_train = sum(epochs) if epochs else None

    # Final metrics live in the "EVALUATION RESULTS" block at the very end.
    # Intermediate epoch prints also contain "MSE:"/"MAE:" etc., so we must
    # scope metric extraction to the tail.
    idx = text.rfind("EVALUATION RESULTS")
    tail_text = text[idx:] if idx >= 0 else text

    def _search_float(rx, txt=tail_text):
        m = rx.search(txt)
        return float(m.group(1).replace(",", "")) if m else None

    def _search_int(rx, txt=text):
        m = rx.search(txt)
        return int(m.group(1).replace(",", "")) if m else None

    # Wall-clock from header start to last timestamped line
    wall_s = None
    m_start = RE_START.search(text)
    if m_start:
        try:
            start = datetime.strptime(m_start.group(1), "%Y-%m-%d %H:%M:%S")
            # last [HH:MM:SS] timestamp in the file
            ts_matches = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", text)
            if ts_matches:
                last_t = ts_matches[-1]
                last = datetime.strptime(
                    start.strftime("%Y-%m-%d") + " " + last_t, "%Y-%m-%d %H:%M:%S"
                )
                if last < start:
                    # crossed midnight; approximate with +24h
                    last = last.replace(day=start.day + 1)
                wall_s = (last - start).total_seconds()
        except Exception:
            pass

    return dict(
        setting=log_path.parent.name,
        avg_epoch_s=avg_epoch,
        total_train_s=total_train,
        wall_clock_s=wall_s,
        num_params=_search_int(RE_PARAMS),
        peak_mem_mb=_search_float(RE_MEM),
        infer_ms=_search_float(RE_INFER),
        completed=bool(RE_COMPLETED.search(text)),
        MAE=_search_float(RE_MAE),
        MSE=_search_float(RE_MSE),
        RMSE=_search_float(RE_RMSE),
        BPE=_search_float(RE_BPE),
        PIM=_search_float(RE_PIM),
        F1=_search_float(RE_F1),
        Precision=_search_float(RE_P),
        Recall=_search_float(RE_R),
        TP_MSE=_search_float(RE_TP_MSE),
        TP_MAE=_search_float(RE_TP_MAE),
    )


def parse_setting_name(setting: str) -> dict:
    """
    Decode the auto-generated setting name produced by run.py:
        <task_name>_<data>_<model>_<model_id>_<seq_len>_<pred_len>_<ii>
    This is heuristic because model_id itself can contain underscores.
    """
    out = dict(task_name="", data="", model="", model_id="",
               seq_len=None, pred_len=None, seed=None)
    parts = setting.split("_")
    if len(parts) < 3:
        return out
    # The last 3 tokens are seq_len, pred_len, seed
    try:
        out["seed"] = int(parts[-1])
        out["pred_len"] = int(parts[-2])
        out["seq_len"] = int(parts[-3])
    except ValueError:
        return out
    head = parts[:-3]
    # task_name is one of:
    #   peak_detect_ltf, peak_detect_ltf_basic, peak_detect_ltf_seq2peak, seq2peak
    if len(head) >= 4 and head[0] == "peak" and head[1] == "detect":
        if head[2] == "ltf" and head[3] == "basic":
            out["task_name"] = "peak_detect_ltf_basic"
            head = head[4:]
        elif head[2] == "ltf" and head[3] == "seq2peak":
            out["task_name"] = "peak_detect_ltf_seq2peak"
            head = head[4:]
        else:
            out["task_name"] = "peak_detect_ltf"
            head = head[3:]
    elif head and head[0] == "seq2peak":
        # Old Seq2Peak runs saved the setting with just the "seq2peak_" prefix
        # (task_name truncated) — treat as the seq2peak task.
        out["task_name"] = "peak_detect_ltf_seq2peak"
        head = head[1:]
    # data = first 1-2 tokens. We know two datasets:
    #   load_data_mixed, electricity_mixed
    if head and head[0] == "load" and len(head) >= 3 and head[1] == "data":
        out["data"] = "load_data_mixed"
        head = head[3:]
    elif head and head[0] == "electricity":
        out["data"] = "electricity_mixed"
        head = head[2:]
    # model = next token (possibly a 2-token model name like "peak_Transformer")
    if head:
        if len(head) >= 2 and head[0] == "peak" and head[1] == "Transformer":
            out["model"] = "peak_Transformer"
            head = head[2:]
        else:
            out["model"] = head[0]
            head = head[1:]
    # remainder is model_id
    out["model_id"] = "_".join(head)
    # Display alias: Seq2Peak task with peak_Transformer backbone → "Seq2Peak"
    if out["task_name"] == "peak_detect_ltf_seq2peak" and \
            out["model"] == "peak_Transformer":
        out["model"] = "Seq2Peak"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results",
                    help="directory containing */experiment_log.txt")
    ap.add_argument("--out", default="paper_data/parsed_logs.csv")
    args = ap.parse_args()

    rows = []
    root = Path(args.results_root)
    for p in sorted(root.glob("*/experiment_log.txt")):
        try:
            base = extract(p)
            meta = parse_setting_name(base["setting"])
            base.update(meta)
            rows.append(base)
        except Exception as e:
            print(f"[warn] {p}: {e}")

    if not rows:
        print("No experiment_log.txt files found.")
        return

    fieldnames = ["setting", "task_name", "data", "model", "model_id",
                  "seq_len", "pred_len", "seed",
                  "completed", "num_params", "avg_epoch_s", "total_train_s",
                  "wall_clock_s", "peak_mem_mb", "infer_ms",
                  "MAE", "MSE", "RMSE", "BPE", "PIM",
                  "F1", "Precision", "Recall", "TP_MSE", "TP_MAE"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Parsed {len(rows)} runs → {out_path}")


if __name__ == "__main__":
    main()
