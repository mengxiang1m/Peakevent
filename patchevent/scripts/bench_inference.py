import argparse
import json
import os
import time
import traceback

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))

from patchevent.models.factory import build_model


def _resolve_path(root: str, value):
    if value is None or not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return value
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(root, value))


def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _load_model(root: str):
    ckpt_path = os.path.join(
        root,
        "patchevent", "phase2",
        "checkpoints",
        "wlel_arch_ablation_v2",
        "G0_full_s42",
        "best.pth",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "args" not in ckpt:
        raise KeyError(f"Checkpoint missing 'args': {ckpt_path}")

    args = argparse.Namespace(**dict(ckpt["args"]))

    for key in ("encoder_ckpt", "series_path", "events_path"):
        if hasattr(args, key):
            setattr(args, key, _resolve_path(root, getattr(args, key)))

    model = build_model(args)

    state_dict = None
    for key in ("model", "model_state_dict", "state_dict"):
        if key in ckpt:
            state_dict = ckpt[key]
            break
    if state_dict is None:
        raise KeyError("No model state dict found in checkpoint (expected one of: model/model_state_dict/state_dict)")

    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if hasattr(model, "norm_mean") and "norm_mean" in ckpt and ckpt["norm_mean"] is not None:
        try:
            model.norm_mean = float(ckpt["norm_mean"])
        except Exception:
            pass
    if hasattr(model, "norm_std") and "norm_std" in ckpt and ckpt["norm_std"] is not None:
        try:
            model.norm_std = float(ckpt["norm_std"])
        except Exception:
            pass

    return model, device


def _run_once(model, x: torch.Tensor, device: torch.device):
    greedy_fn = getattr(model, "greedy_generate", None)
    if callable(greedy_fn):
        return greedy_fn(x, max_len=getattr(model, "max_seq_len", 64))

    tokenizer = getattr(model, "tokenizer", None)
    vocab_size = getattr(tokenizer, "VOCAB_SIZE", None)
    if vocab_size is None:
        vocab_size = getattr(model, "vocab_size", 512)

    max_seq_len = int(getattr(model, "max_seq_len", 64))
    target_ids = torch.randint(
        low=3,
        high=int(vocab_size),
        size=(x.size(0), max_seq_len + 1),
        device=device,
        dtype=torch.long,
    )

    bos_id = int(getattr(tokenizer, "BOS_ID", 1))
    target_ids[:, 0] = bos_id

    return model(x, target_ids)


def benchmark():
    batch_size = 64
    seq_len = 96
    warmup_iters = 10
    bench_iters = 100

    model, device = _load_model(ROOT)
    x = torch.randn(batch_size, seq_len, device=device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _run_once(model, x, device)

        _cuda_sync_if_needed(device)
        t0 = time.perf_counter()

        for _ in range(bench_iters):
            _run_once(model, x, device)

        _cuda_sync_if_needed(device)
        t1 = time.perf_counter()

    elapsed = t1 - t0
    total_samples = batch_size * bench_iters
    samples_per_sec = total_samples / elapsed if elapsed > 0 else float("inf")
    ms_per_sample = (elapsed * 1000.0) / total_samples if total_samples > 0 else float("inf")

    result = {
        "model": "PatchEvent",
        "batch_size": batch_size,
        "samples_per_sec": samples_per_sec,
        "ms_per_sample": ms_per_sample,
    }
    if device.type != "cuda":
        result["note"] = "GPU not available; benchmark ran on CPU"

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        benchmark()
    except Exception:
        traceback.print_exc()
        raise
