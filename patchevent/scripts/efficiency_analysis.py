"""
Efficiency Analysis: Parameter counts for PatchEvent and baseline models.
Configuration: WLEL pred96 (seq_len=96, pred_len=96, enc_in=1)
"""
from __future__ import annotations
import sys, os, time, argparse, traceback, subprocess, json
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
HELPER = os.path.join(ROOT, "_count_params.py")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt(n):
    if isinstance(n, str): return n
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)


def build_patchevent():
    from patchevent.models.factory import build_model
    args = argparse.Namespace(
        encoder_ckpt=os.path.join(ROOT, "patchevent", "phase1", "checkpoints", "wlel", "s42", "best_model.pth"),
        d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.2,
        max_seq_len=160, pred_len=96, event_schema="triplet",
        encoder_mode="frozen", embed_dropout=0.15,
        unfreeze_last_n=0, use_memory_pos=True, use_self_attn_agg=True,
        use_input_decomp=False, input_decomp_mode="full", use_raw_bypass=False,
        use_tuple_guided_value=False, use_independent_dense_int=False,
        use_apex_cond_int=False, use_int_direct_lookup=False,
        use_decoupled_int_head=False, use_tuple_crossatt_int=False,
        min_onset_spacing=0, override_norm_mean=None, override_norm_std=None,
        no_pos_valid_mask=False, no_causal_mask=False,
        use_int_regression=False, use_series_stats=False,
        series_stats_recent_k=8, intensity_from_values="apex",
        use_non_ar=False,
    )
    model = build_model(args)
    return model


def count_baseline_params(model_type, model_name):
    """Use subprocess to avoid import conflicts with layers modules."""
    proc = subprocess.run(
        [PYTHON, HELPER, model_type, model_name],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:200])
    result = json.loads(proc.stdout.strip())
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["total"], result["trainable"]


def benchmark_patchevent_inference(model, device="cpu", n_warmup=5, n_iter=50):
    model.eval()
    model = model.to(device)
    batch_size = 32
    seq_len = 96
    x = torch.randn(batch_size, seq_len, device=device)
    target_ids = torch.ones(batch_size, 2, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                model(x, target_ids)
            except Exception:
                return None, "forward failed"
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            model(x, target_ids)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_samples = batch_size * n_iter
    samples_per_sec = total_samples / elapsed
    return samples_per_sec, "ok"


def main():
    print("=" * 72)
    print("Efficiency Analysis: WLEL pred96 (seq_len=96, pred_len=96, enc_in=1)")
    print("=" * 72)
    results = []

    # 1. PatchEvent (in-process, needs encoder checkpoint)
    print("\n[1/6] Building PatchEvent ...")
    pe_model = None
    try:
        pe_model = build_patchevent()
        total, trainable = count_params(pe_model)
        enc_params = sum(p.numel() for p in pe_model.encoder.parameters())
        enc_trainable = sum(p.numel() for p in pe_model.encoder.parameters() if p.requires_grad)
        dec_only = total - enc_params
        dec_trainable = trainable - enc_trainable
        results.append(("PatchEvent (total)", total, trainable, ""))
        results.append(("  - Phase1 Encoder", enc_params, enc_trainable, "frozen"))
        results.append(("  - Phase2 Decoder", dec_only, dec_trainable, ""))
    except Exception as e:
        traceback.print_exc()
        results.append(("PatchEvent", "N/A", "N/A", str(e)[:80]))

    # 2-5. TSLib baselines (via subprocess)
    baselines = [
        ("iTransformer", "tslib", "iTransformer"),
        ("PatchTST", "tslib", "PatchTST"),
        ("DLinear", "tslib", "DLinear"),
        ("TimeMixer", "tslib", "TimeMixer"),
        ("Seq2Peak (peak_Autoformer)", "seq2peak", "peak_Autoformer"),
    ]
    for i, (display_name, model_type, model_name) in enumerate(baselines, start=2):
        print(f"\n[{i}/6] Building {display_name} ...")
        try:
            total, trainable = count_baseline_params(model_type, model_name)
            results.append((display_name, total, trainable, ""))
        except Exception as e:
            traceback.print_exc()
            results.append((display_name, "N/A", "N/A", str(e)[:80]))

    # Inference speed
    speed_str = "N/A"
    if pe_model is not None:
        print("\n[Benchmark] Measuring PatchEvent inference speed ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            sps, status = benchmark_patchevent_inference(pe_model, device=device)
            if sps is not None:
                speed_str = f"{sps:.1f} samples/sec ({device})"
            else:
                speed_str = f"N/A ({status})"
        except Exception as e:
            speed_str = f"N/A ({e})"

    # Print table
    print("\n" + "=" * 72)
    print("Results Table")
    print("=" * 72)
    COL1 = "Model"
    COL2 = "Total Params"
    COL3 = "Trainable"
    print(f"{COL1:<30} {COL2:>15} {COL3:>15} Note")
    print("-" * 72)

    for name, total, trainable, note in results:
        t_str = fmt(total) if isinstance(total, int) else total
        tr_str = fmt(trainable) if isinstance(trainable, int) else trainable
        if isinstance(total, int) and isinstance(trainable, int):
            pct = f"({trainable/total*100:.1f}%)" if total > 0 else ""
        else:
            pct = ""
        print(f"{name:<30} {t_str:>15} {tr_str:>15} {pct:>8} {note}")

    print("-" * 72)
    print(f"PatchEvent inference speed: {speed_str}")
    print()

    # Markdown table
    print("\n## Markdown table (for paper)\n")
    print("| Model | Total Params | Trainable Params |")
    print("|-------|-------------|------------------|")
    for name, total, trainable, note in results:
        if name.startswith("  -"):
            continue
        t_str = fmt(total) if isinstance(total, int) else total
        tr_str = fmt(trainable) if isinstance(trainable, int) else trainable
        print(f"| {name} | {t_str} | {tr_str} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
