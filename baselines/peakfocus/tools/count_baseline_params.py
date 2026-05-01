"""
Compute parameter counts for every baseline model analytically.

Old experiment logs (the ones from the previous paper draft) do NOT contain
the [PARAMS] marker that the new instrumentation writes, so we need an
alternative way to populate num_params for the efficiency radar chart.

This script imports each baseline model with the same hyperparameters used
in `run.py`'s default command line (what the main-table experiments used)
and counts trainable parameters via sum(p.numel() for p in model.parameters()
if p.requires_grad).

Output: paper_data/baseline_params.csv
    model, dataset, horizon, num_params

Usage:
    python tools/count_baseline_params.py
"""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO_CODE = HERE.parent


# ---------- default hyper-parameters used by the main-table experiments ----
# These match the flags the old shell scripts passed to run.py. Any model
# that wants extra flags gets a per-model override via MODEL_OVERRIDES.
BASE_ARGS_WLEL = dict(
    task_name="peak_detect_ltf",
    seq_len=168, label_len=48, pred_len=336,
    enc_in=1, dec_in=1, c_out=1,
    d_model=512, d_ff=2048, e_layers=1, d_layers=1,
    n_heads=4, factor=3, dropout=0.1, activation="gelu",
    embed="timeF", freq="t",
    num_class=0, top_k=5, num_kernels=6, distil=True,
    moving_avg=25, seg_len=24, channel_independence=1,
    decomp_method="moving_avg", use_norm=1, down_sampling_layers=0,
    down_sampling_window=1, down_sampling_method=None,
    features="S", output_attention=False,
    mlp_layers=2,
)


# PatchTST in the OLD main-table run used d_model=512, d_ff=2048 (the runs
# tagged "maxIn_maxOut_MSE_244_23"), which is why a "fair" re-run is now in
# progress with d_model=256, d_ff=512. We record both configs so the
# radar chart can compare them.
MODEL_OVERRIDES = {
    "PatchTST_unfair": dict(model_name="PatchTST",
                            d_model=512, d_ff=2048, e_layers=1, n_heads=4,
                            patch_len=16, stride=8),
    "PatchTST_fair":   dict(model_name="PatchTST",
                            d_model=256, d_ff=512,  e_layers=1, n_heads=4,
                            patch_len=16, stride=8),
    "Transformer":     dict(d_model=512, d_ff=2048, e_layers=2, d_layers=1),
    "Informer":        dict(d_model=512, d_ff=2048, e_layers=2, d_layers=1),
    "TimeMixer":       dict(d_model=16,  d_ff=32,   e_layers=2),
    "SegRNN":          dict(d_model=512, d_ff=2048, seg_len=24),
    "CycleNet":        dict(d_model=512, cycle_len=168, use_revin=1,
                            model_type="linear"),
    "STID":            dict(d_model=32,  e_layers=3),
    "DLinear":         dict(d_model=512, d_ff=2048, individual=False),
    # Seq2Peak = peak_Transformer backbone + peak_detect_ltf_seq2peak task
    "Seq2Peak":        dict(model_name="peak_Transformer",
                            d_model=512, d_ff=2048, e_layers=2, d_layers=1,
                            n_heads=8),
    # PeakFocus (the proposed model) — match the default config from
    # shell/test_proposed_model.sh: d_model=256, d_ff=256, e_layers=1,
    # n_heads=4, n_scales=2, patch_len=16, stride=8.
    "PeakFocus":       dict(model_name="proposed_model",
                            d_model=256, d_ff=256, e_layers=1, n_heads=4,
                            n_scales=2, patch_len=16, stride=8,
                            use_external_features=0, endogenous_in=1,
                            num_external_features=0),
}

# We compute num_params for every (dataset, horizon) combo since the prediction
# length affects the parameter count of some models (e.g. DLinear, SegRNN).
DATASETS = [
    dict(dataset="WLEL", seq_len=168),
    dict(dataset="ELC",  seq_len=168),
]
HORIZONS = [336, 720]


def _args_for(model_label: str, pred_len: int) -> SimpleNamespace:
    """Build the argparse Namespace that each model's __init__ expects."""
    args = dict(BASE_ARGS_WLEL)
    args["pred_len"] = pred_len
    if model_label in MODEL_OVERRIDES:
        args.update(MODEL_OVERRIDES[model_label])
    # Default falls: some models expect these even when unused
    args.setdefault("use_gpu", False)
    args.setdefault("devices", "0")
    args.setdefault("device", "cpu")
    args.setdefault("embed_type", 0)
    args.setdefault("kernel_size", 25)
    args.setdefault("cycle", 168)        # CycleNet: one daily cycle
    args.setdefault("num_nodes", 1)       # STID: univariate
    args.setdefault("input_dim", 1)
    args.setdefault("node_dim", 32)
    args.setdefault("temp_dim_tid", 32)
    args.setdefault("temp_dim_diw", 32)
    args.setdefault("if_node", True)
    args.setdefault("if_T_i_D", True)
    args.setdefault("if_D_i_W", True)
    args.setdefault("if_time_in_day", True)
    args.setdefault("if_day_in_week", True)
    args.setdefault("time_of_day_size", 1440)
    args.setdefault("day_of_week_size", 7)
    args.setdefault("embed_dim", 32)
    args.setdefault("dim_static", 8)
    args.setdefault("n_patch", 8)
    args.setdefault("num_layer", 3)
    args.setdefault("if_padding", True)
    args.setdefault("padding_mode", "zero")
    args.setdefault("static_hidden", 32)
    args.setdefault("dim_input", 1)
    args.setdefault("dim_output", 1)
    args.setdefault("mpp_size", 16)
    args.setdefault("mpp_hidden", 64)
    args.setdefault("mpp_layers", 2)
    # peak_Transformer (Seq2Peak)
    args.setdefault("hour_day", "h")
    args.setdefault("p_hidden_dims", [128, 128])
    args.setdefault("p_hidden_layers", 2)
    args.setdefault("with_shift", False)
    args.setdefault("busy_decoder", False)
    args.setdefault("busy_len", 24)
    args.setdefault("busy_hidden", 128)
    # PeakFocus / proposed_model flags
    args.setdefault("if_msm_pl", 1)
    args.setdefault("if_lad", 1)
    args.setdefault("n_scales", 2)
    args.setdefault("patch_len", 16)
    args.setdefault("stride", 8)
    args.setdefault("c_in", 1)
    args.setdefault("use_external_features", 0)
    args.setdefault("endogenous_in", 1)
    args.setdefault("num_external_features", 0)
    args.setdefault("external_feature_mode", "concat")
    return SimpleNamespace(**args)


def _count_params(model_label: str, pred_len: int) -> int | None:
    """Import the matching model module from models/ and count its parameters.
    Returns None on any import/construction error (and prints the reason)."""
    import torch  # lazy so the script still runs without GPU/torch envs

    model_name = MODEL_OVERRIDES.get(model_label, {}).get("model_name",
                                                          model_label)
    try:
        mod = importlib.import_module(f"models.{model_name}")
    except Exception as e:
        print(f"  [skip] {model_label}: cannot import models.{model_name}: {e}")
        return None

    Model = getattr(mod, "Model", None)
    if Model is None:
        print(f"  [skip] {model_label}: no Model class in module")
        return None

    args = _args_for(model_label, pred_len)
    try:
        model = Model(args)
    except Exception as e:
        print(f"  [skip] {model_label}@H={pred_len}: "
              f"{type(e).__name__}: {e}")
        return None

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n


def main():
    sys.path.insert(0, str(REPO_CODE))

    models = [
        "PeakFocus",
        "Transformer", "Informer",
        "PatchTST_unfair", "PatchTST_fair",
        "TimeMixer", "SegRNN", "CycleNet", "STID",
        "Seq2Peak",
    ]

    rows: list[dict] = []
    for ds in DATASETS:
        for h in HORIZONS:
            print(f"--- {ds['dataset']}  H={h} ---")
            for m in models:
                n = _count_params(m, h)
                if n is None:
                    continue
                if m.startswith("PatchTST"):
                    display = "PatchTST"
                else:
                    display = m
                if m == "PatchTST_unfair":
                    config_tag = "unfair"
                elif m == "PatchTST_fair":
                    config_tag = "fair"
                else:
                    config_tag = "default"
                rows.append(dict(
                    model=display,
                    config=config_tag,
                    dataset=ds["dataset"],
                    horizon=h,
                    num_params=n,
                ))
                print(f"  {display:<13} ({config_tag:>7}) "
                      f"→ {n/1e6:6.3f} M params")

    out = REPO_CODE / "paper_data" / "baseline_params.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "config", "dataset",
                                          "horizon", "num_params"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
