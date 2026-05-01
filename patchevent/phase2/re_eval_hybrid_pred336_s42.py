from __future__ import annotations

import json
import os
from types import SimpleNamespace
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # TODO(R-future): migrate to patchevent package import

from patchevent.phase2.dataset import build_dataloaders
from patchevent.phase2.evaluate import evaluate_loader
from patchevent.phase2.model import build_model
from patchevent.phase2.train import set_seed

CKPT_PATH = os.path.join(
    ROOT_DIR,
    "patchevent", "phase2",
    "checkpoints",
    "wlel_pred336_hybrid",
    "TG1_triplet_guided_s42",
    "best.pth",
)


def _abs_from_root(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def main() -> None:
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"checkpoint not found: {CKPT_PATH}")

    print(f"[re-eval] loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    args_dict = dict(ckpt.get("args", {}))
    if not args_dict:
        raise ValueError("checkpoint missing args")

    args_dict["encoder_ckpt"] = _abs_from_root(args_dict["encoder_ckpt"])
    args_dict["series_path"] = _abs_from_root(args_dict["series_path"])
    args_dict["events_path"] = _abs_from_root(args_dict["events_path"])
    if args_dict.get("patch_labels_path"):
        args_dict["patch_labels_path"] = _abs_from_root(args_dict["patch_labels_path"])
    args_dict["output_dir"] = _abs_from_root(args_dict["output_dir"])
    args = SimpleNamespace(**args_dict)

    for k in ("encoder_ckpt", "series_path", "events_path"):
        if not os.path.exists(getattr(args, k)):
            raise FileNotFoundError(f"{k} not found: {getattr(args, k)}")

    set_seed(int(args.seed))
    device = torch.device(
        f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[re-eval] device = {device}")

    _, _, test_loader, meta = build_dataloaders(
        series_path=args.series_path,
        events_path=args.events_path,
        seq_len=int(args.seq_len),
        pred_len=int(args.pred_len),
        event_schema=args.event_schema,
        window_stride=int(args.window_stride),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        patch_labels_path=getattr(args, "patch_labels_path", None),
    )
    print(f"[re-eval] test batches = {len(test_loader)}")

    model = build_model(args).to(device)
    state = ckpt.get("model", ckpt)

    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = []
    for key, value in state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered_state[key] = value
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(
        f"[re-eval] loaded with missing={len(missing)} unexpected={len(unexpected)} "
        f"shape_skipped={len(skipped_shape)}"
    )

    if "norm_mean" in ckpt and hasattr(model, "norm_mean"):
        model.norm_mean.data = ckpt["norm_mean"].to(model.norm_mean.device)
    if "norm_std" in ckpt and hasattr(model, "norm_std"):
        model.norm_std.data = ckpt["norm_std"].to(model.norm_std.device)

    save_dir = os.path.join(args.output_dir, "re_eval_intmape_fix")
    os.makedirs(save_dir, exist_ok=True)

    metrics = evaluate_loader(
        model=model,
        test_loader=test_loader,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        tolerance=int(args.tolerance),
        verbose=True,
        save_dir=save_dir,
        intensity_scale=float(meta["mean"]),
    )

    metrics_path = os.path.join(save_dir, "metrics_stdout_copy.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[re-eval] metrics json saved -> {metrics_path}")

    print("[re-eval] metrics summary:")
    for key in sorted(metrics.keys()):
        print(f"{key}={metrics[key]}")


if __name__ == "__main__":
    main()
