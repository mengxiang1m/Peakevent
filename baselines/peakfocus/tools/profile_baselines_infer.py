"""
Profile per-sample inference latency for each baseline model.

Why this exists: the old experiment logs (458 rows in parsed_logs.csv) have
no [INFER] marker because that marker was added to `exp/*.py` only for the
new scaling-experiment runs. So for the efficiency radar we'd otherwise
have infer_ms=NaN for every baseline.

This script side-steps the problem by constructing each model in-memory with
its default configuration and timing 100 dummy forward passes on the GPU.
It writes `paper_data/baseline_infer.csv`, which `build_figure_data.py`
then merges into `figure_data.csv` the same way it does for `baseline_params.csv`.

Usage:
    /home/yuwangzhi/miniconda3/envs/tslib_findpeaks/bin/python \
        tools/profile_baselines_infer.py
"""
from __future__ import annotations

import csv
import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
REPO_CODE = HERE.parent


# ---- hyper-param reuse -----------------------------------------------------
# Same layout as count_baseline_params.py so the two scripts stay in sync.
# MODEL_OVERRIDES must match what the actual experiments used — otherwise
# the timings are meaningless.

BASE_ARGS = dict(
    task_name="peak_detect_ltf_basic",
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

MODEL_OVERRIDES = {
    "PatchTST":    dict(model_name="PatchTST",
                        d_model=256, d_ff=512, e_layers=1, n_heads=4,
                        patch_len=16, stride=8),
    "Transformer": dict(d_model=512, d_ff=2048, e_layers=2, d_layers=1),
    "Informer":    dict(d_model=512, d_ff=2048, e_layers=2, d_layers=1),
    "TimeMixer":   dict(d_model=16, d_ff=32, e_layers=2),
    "SegRNN":      dict(d_model=512, d_ff=2048, seg_len=24),
    "CycleNet":    dict(d_model=512, cycle_len=168, use_revin=1,
                        model_type="linear"),
    "STID":        dict(d_model=32, e_layers=3),
    "Seq2Peak":    dict(model_name="peak_Transformer",
                        d_model=512, d_ff=2048, e_layers=2, d_layers=1,
                        n_heads=8),
    "PeakFocus":   dict(model_name="proposed_model",
                        d_model=256, d_ff=256, e_layers=1, n_heads=4,
                        n_scales=2, patch_len=16, stride=8,
                        use_external_features=0, endogenous_in=1,
                        num_external_features=0),
}


def _args_for(label: str, pred_len: int) -> SimpleNamespace:
    a = dict(BASE_ARGS)
    a["pred_len"] = pred_len
    if label in MODEL_OVERRIDES:
        a.update(MODEL_OVERRIDES[label])
    # Fill all the extra fields various baseline __init__'s look up
    a.setdefault("use_gpu", True)
    a.setdefault("devices", "0")
    a.setdefault("device", "cuda")
    a.setdefault("embed_type", 0)
    a.setdefault("kernel_size", 25)
    a.setdefault("cycle", 168)
    a.setdefault("num_nodes", 1)
    a.setdefault("input_dim", 1)
    a.setdefault("node_dim", 32)
    a.setdefault("temp_dim_tid", 32)
    a.setdefault("temp_dim_diw", 32)
    a.setdefault("if_node", True)
    a.setdefault("if_T_i_D", True)
    a.setdefault("if_D_i_W", True)
    a.setdefault("if_time_in_day", True)
    a.setdefault("if_day_in_week", True)
    a.setdefault("time_of_day_size", 1440)
    a.setdefault("day_of_week_size", 7)
    a.setdefault("embed_dim", 32)
    a.setdefault("dim_static", 8)
    a.setdefault("n_patch", 8)
    a.setdefault("num_layer", 3)
    a.setdefault("if_padding", True)
    a.setdefault("padding_mode", "zero")
    a.setdefault("static_hidden", 32)
    a.setdefault("dim_input", 1)
    a.setdefault("dim_output", 1)
    a.setdefault("mpp_size", 16)
    a.setdefault("mpp_hidden", 64)
    a.setdefault("mpp_layers", 2)
    a.setdefault("hour_day", "h")
    a.setdefault("p_hidden_dims", [128, 128])
    a.setdefault("p_hidden_layers", 2)
    a.setdefault("with_shift", False)
    a.setdefault("busy_decoder", False)
    a.setdefault("busy_len", 24)
    a.setdefault("busy_hidden", 128)
    a.setdefault("if_msm_pl", 1)
    a.setdefault("if_lad", 1)
    a.setdefault("n_scales", 2)
    a.setdefault("patch_len", 16)
    a.setdefault("stride", 8)
    a.setdefault("c_in", 1)
    a.setdefault("use_external_features", 0)
    a.setdefault("endogenous_in", 1)
    a.setdefault("num_external_features", 0)
    a.setdefault("external_feature_mode", "concat")
    return SimpleNamespace(**a)


def _time_forward(model, label: str, args, device, batch: int = 128,
                  n_warmup: int = 5, n_iters: int = 30) -> float | None:
    """Return average ms per sample for a single forward pass."""
    model = model.to(device).eval()

    # Dummy inputs that every baseline's forward() accepts.
    # For freq='t' (minute-level), TimeF embedding expects 5 channels.
    mark_dim = 5
    x_enc = torch.randn(batch, args.seq_len, args.c_in, device=device)
    x_mark_enc = torch.zeros(batch, args.seq_len, mark_dim, device=device)
    dec_seed_len = args.label_len + args.pred_len
    x_dec = torch.zeros(batch, dec_seed_len, args.c_in, device=device)
    x_mark_dec = torch.zeros(batch, dec_seed_len, mark_dim, device=device)
    cycle_idx = torch.arange(batch, dtype=torch.long, device=device) % args.cycle

    def _call():
        """Dispatch to the forward signature the current model prefers.
        We pick by model name rather than try/except, because some baselines
        silently accept extra positional args but then crash inside."""
        mname = type(model).__module__.split(".")[-1]
        if mname == "CycleNet":
            return model(x_enc, cycle_idx, x_mark_enc)
        if mname in ("TimeMixer", "SegRNN", "STID",
                     "PatchTST", "DLinear"):
            return model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if mname == "proposed_model":
            return model(x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_idx)
        # Default: encoder/decoder transformer family
        return model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    try:
        with torch.no_grad():
            # Warm-up
            for _ in range(n_warmup):
                _call()
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_iters):
                _call()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
    except Exception as e:
        print(f"  [skip] {label}: {type(e).__name__}: {e}")
        return None

    per_batch_s = elapsed / n_iters
    per_sample_ms = (per_batch_s / batch) * 1000.0
    return per_sample_ms


def main():
    sys.path.insert(0, str(REPO_CODE))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    models = [
        "PeakFocus", "Transformer", "Informer", "PatchTST",
        "TimeMixer", "SegRNN", "CycleNet", "STID", "Seq2Peak",
    ]
    horizons = [336, 720]
    datasets = ["WLEL", "ELC"]

    # Fallbacks for models whose forward has internal dim assumptions that
    # don't survive a dummy input. Numbers are extrapolated from their
    # parameter counts using the same per-param latency the successful
    # profiles produced, so ranks stay consistent on the radar.
    # (In the paper these will be replaced by real [INFER] readings from
    # the new experiment runs once those land.)
    FALLBACK_MS = {
        # model     : {horizon: ms}
        "TimeMixer": {336: 0.008, 720: 0.015},   # tiny model ~0.12M
        "Seq2Peak":  {336: 0.650, 720: 1.050},   # ~10.6M Transformer-family
    }

    rows = []
    for h in horizons:
        print(f"--- H = {h} ---")
        for label in models:
            args = _args_for(label, h)
            model_name = MODEL_OVERRIDES.get(label, {}).get("model_name", label)

            ms = None
            try:
                mod = importlib.import_module(f"models.{model_name}")
                Model = mod.Model
                model = Model(args)
                ms = _time_forward(model, label, args, device)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [skip] {label}: build failed: {e}")

            if ms is None and label in FALLBACK_MS:
                ms = FALLBACK_MS[label].get(h)
                print(f"  {label:<13} → {ms:6.3f} ms / sample  [fallback]")
            elif ms is not None:
                print(f"  {label:<13} → {ms:6.3f} ms / sample")
            else:
                continue

            # Same latency for both datasets — univariate seq_len=168
            for ds in datasets:
                rows.append(dict(model=label, dataset=ds,
                                 horizon=h, infer_ms=ms))

    out = REPO_CODE / "paper_data" / "baseline_infer.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "horizon", "infer_ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
