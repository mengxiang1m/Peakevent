"""Sweep Hybrid DETR-AR exist threshold and onset NMS radius.

The script selects the best `(exist_threshold, hybrid_nms_onset_radius)` on the
validation split, then evaluates the selected pair on the test split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from patchevent.phase2.dataset import build_dataloaders
from patchevent.phase2.evaluate import evaluate_loader
from patchevent.phase2.model import build_model


DEFAULT_THRESHOLDS = [round(v / 10.0, 1) for v in range(1, 10)]
DEFAULT_NMS_RADII = [0, 1, 2, 3]


def _parse_float_list(text: str | None, default: list[float]) -> list[float]:
    if not text:
        return list(default)
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_int_list(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return list(default)
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _finite_or(value, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(value) or math.isinf(value):
        return fallback
    return value


def _selection_key(row: dict) -> tuple[float, float, float]:
    metrics = row["metrics"]
    f1 = _finite_or(metrics.get("event_f1"), -1.0)
    onset = _finite_or(metrics.get("onset_mae"), 1e9)
    precision = _finite_or(metrics.get("event_precision"), 0.0)
    return (f1, -onset, precision)


def _namespace_from_checkpoint(ckpt: dict, cli_args: argparse.Namespace) -> SimpleNamespace:
    args_dict = dict(ckpt.get("args") or {})
    state = ckpt.get("model", ckpt)
    args_dict["decoder_type"] = "hybrid"
    args_dict["series_path"] = cli_args.series_path or args_dict.get("series_path")
    args_dict["events_path"] = cli_args.events_path or args_dict.get("events_path")
    args_dict["batch_size"] = int(cli_args.batch_size or args_dict.get("batch_size", 64))
    args_dict["eval_batch_size"] = int(cli_args.eval_batch_size or args_dict.get("eval_batch_size", 128))
    args_dict["gpu"] = int(cli_args.gpu)
    args_dict.setdefault("event_schema", "quad")
    args_dict.setdefault("seq_len", 96)
    args_dict.setdefault("pred_len", 96)
    args_dict.setdefault("window_stride", 4)
    args_dict.setdefault("patch_len", 8)
    args_dict.setdefault("patch_stride", 4)
    args_dict.setdefault("backbone_layers", 2)
    args_dict.setdefault("hybrid_refine_order", "onset")
    args_dict.setdefault("hybrid_no_causal_refine_mask", False)
    args_dict.setdefault("hybrid_use_matched_refine_order", False)
    args_dict.setdefault("hybrid_nms_onset_radius", 0)
    args_dict.setdefault("hybrid_use_count_head", False)
    args_dict.setdefault("hybrid_use_count_decoding", False)
    args_dict.setdefault("hybrid_no_count_decoding", False)
    if "hybrid_time_head" not in args_dict:
        loc_weight = state.get("localization_head.3.weight") if isinstance(state, dict) else None
        args_dict["hybrid_time_head"] = (
            "structured"
            if loc_weight is not None and int(loc_weight.shape[0]) == 5
            else "independent"
        )
    args_dict.setdefault("max_event_count", 12)
    args_dict.setdefault("hybrid_structure_loss_weight", 0.0)
    args_dict.setdefault("exist_threshold", 0.5)
    if not args_dict.get("series_path") or not args_dict.get("events_path"):
        raise ValueError("series_path and events_path are required via checkpoint args or CLI.")
    return SimpleNamespace(**args_dict)


def _load_model(run_args: SimpleNamespace, ckpt: dict, device: torch.device):
    model = build_model(run_args).to(device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def _evaluate_combo(
    model,
    loader,
    device: torch.device,
    threshold: float,
    nms_radius: int,
    max_new_tokens: int,
    tolerance: int,
    intensity_scale: float,
    save_dir: str | None = None,
) -> dict:
    model.exist_threshold = float(threshold)
    model.hybrid_nms_onset_radius = int(nms_radius)
    return evaluate_loader(
        model,
        loader,
        device,
        max_new_tokens=max_new_tokens,
        tolerance=tolerance,
        verbose=False,
        save_dir=save_dir,
        intensity_scale=intensity_scale,
        hybrid_nms_onset_radius=nms_radius,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep Hybrid DETR-AR threshold/NMS calibration.")
    p.add_argument("--checkpoint_dir", required=True, help="Directory containing best.pth")
    p.add_argument("--checkpoint_name", default="best.pth")
    p.add_argument("--series_path", default=None)
    p.add_argument("--events_path", default=None)
    p.add_argument("--thresholds", default=None, help="Comma list; default 0.1,...,0.9")
    p.add_argument("--nms_radii", default=None, help="Comma list; default 0,1,2,3")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--tolerance", type=int, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_test_outputs", action="store_true")
    return p.parse_args()


def main() -> None:
    cli_args = parse_args()
    checkpoint_dir = Path(cli_args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / cli_args.checkpoint_name
    out_dir = checkpoint_dir / "calibrated_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{cli_args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[sweep] device={device}")
    print(f"[sweep] checkpoint={checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_args = _namespace_from_checkpoint(ckpt, cli_args)

    _, val_loader, test_loader, meta = build_dataloaders(
        series_path=run_args.series_path,
        events_path=run_args.events_path,
        seq_len=int(run_args.seq_len),
        pred_len=int(run_args.pred_len),
        event_schema=run_args.event_schema,
        window_stride=int(run_args.window_stride),
        batch_size=int(run_args.batch_size),
        eval_batch_size=int(run_args.eval_batch_size),
        patch_labels_path=getattr(run_args, "patch_labels_path", None),
    )
    run_args.norm_mean = meta["mean"]
    run_args.norm_std = meta["std"]
    model = _load_model(run_args, ckpt, device)

    thresholds = _parse_float_list(cli_args.thresholds, DEFAULT_THRESHOLDS)
    nms_radii = _parse_int_list(cli_args.nms_radii, DEFAULT_NMS_RADII)
    max_new_tokens = int(cli_args.max_new_tokens or getattr(run_args, "max_new_tokens", 60))
    tolerance = int(cli_args.tolerance or getattr(run_args, "tolerance", 3))

    candidates = []
    for threshold in thresholds:
        for nms_radius in nms_radii:
            print(f"[val] threshold={threshold:.2f} nms_radius={nms_radius}")
            metrics = _evaluate_combo(
                model,
                val_loader,
                device,
                threshold=threshold,
                nms_radius=nms_radius,
                max_new_tokens=max_new_tokens,
                tolerance=tolerance,
                intensity_scale=meta["mean"],
            )
            candidates.append({
                "exist_threshold": threshold,
                "hybrid_nms_onset_radius": nms_radius,
                "metrics": metrics,
            })

    best = max(candidates, key=_selection_key)
    val_payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(checkpoint_path),
        "split": "val",
        "thresholds": thresholds,
        "nms_radii": nms_radii,
        "selection_rule": "max event_f1, then min onset_mae, then max precision",
        "best_selection": {
            "exist_threshold": best["exist_threshold"],
            "hybrid_nms_onset_radius": best["hybrid_nms_onset_radius"],
            "metrics": best["metrics"],
        },
        "candidates": candidates,
    }
    val_path = out_dir / "threshold_sweep_val.json"
    val_path.write_text(json.dumps(val_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVE] {val_path}")

    test_save_dir = str(out_dir / "test_eval") if cli_args.save_test_outputs else None
    test_metrics = _evaluate_combo(
        model,
        test_loader,
        device,
        threshold=float(best["exist_threshold"]),
        nms_radius=int(best["hybrid_nms_onset_radius"]),
        max_new_tokens=max_new_tokens,
        tolerance=tolerance,
        intensity_scale=meta["mean"],
        save_dir=test_save_dir,
    )
    test_payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(checkpoint_path),
        "split": "test",
        "selected_from_val": {
            "exist_threshold": best["exist_threshold"],
            "hybrid_nms_onset_radius": best["hybrid_nms_onset_radius"],
            "val_metrics": best["metrics"],
        },
        "metrics": test_metrics,
    }
    test_path = out_dir / "threshold_sweep_test.json"
    test_path.write_text(json.dumps(test_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVE] {test_path}")


if __name__ == "__main__":
    main()
