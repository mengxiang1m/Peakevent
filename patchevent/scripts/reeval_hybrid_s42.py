"""
Re-evaluate Hybrid pred336 s42 with the fixed evaluate.py (IntMAPE bug fix).
"""
import sys, os, json
from argparse import Namespace
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)  # TODO(R-future): migrate to patchevent package import

import torch
from patchevent.phase2.dataset import build_dataloaders
from patchevent.phase2.model import build_model
from patchevent.phase2.evaluate import evaluate_loader

CKPT_DIR = os.path.join(ROOT, "patchevent/phase2/checkpoints/wlel_pred336_hybrid/TG1_triplet_guided_s42")
BEST_PTH = os.path.join(CKPT_DIR, "best.pth")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(BEST_PTH, map_location="cpu", weights_only=False)
    args_raw = ckpt.get("args")
    args = Namespace(**args_raw) if isinstance(args_raw, dict) else args_raw
    norm_mean = ckpt.get("norm_mean", 1.0)
    norm_std = ckpt.get("norm_std", 1.0)

    print(f"  pred_len={args.pred_len}, event_schema={getattr(args, 'event_schema', 'quad')}")
    print(f"  use_tuple_guided_value={getattr(args, 'use_tuple_guided_value', False)}")
    print(f"  norm_mean={norm_mean}, norm_std={norm_std}")

    # Build dataloaders (using same args as training)
    _, _, test_loader, meta = build_dataloaders(
        series_path=args.series_path,
        events_path=args.events_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        event_schema=getattr(args, 'event_schema', 'quad'),
        window_stride=getattr(args, 'window_stride', 4),
        batch_size=getattr(args, 'batch_size', 32),
        eval_batch_size=getattr(args, 'eval_batch_size', 128),
        patch_labels_path=getattr(args, 'patch_labels_path', None),
    )
    print(f"Test loader: {len(test_loader)} batches")

    # Build model and load weights
    model = build_model(args).to(device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Evaluate
    max_new_tokens = getattr(args, "max_new_tokens", 120)
    tolerance = getattr(args, "tolerance", 3)
    intensity_scale = meta['mean']

    print(f"\nEvaluating (max_new_tokens={max_new_tokens}, tolerance={tolerance}, intensity_scale={intensity_scale:.2f})...")
    metrics = evaluate_loader(
        model, test_loader, device,
        max_new_tokens=max_new_tokens,
        tolerance=tolerance,
        intensity_scale=intensity_scale,
        verbose=True,
        save_dir=os.path.join(CKPT_DIR, "test_eval_fixed"),
    )

    # Save
    out_path = os.path.join(CKPT_DIR, "test_summary_fixed.json")
    summary = {
        'test_event_f1': metrics.get('event_f1'),
        'test_event_precision': metrics.get('event_precision'),
        'test_event_recall': metrics.get('event_recall'),
        'test_apex_mae': metrics.get('apex_mae'),
        'test_onset_mae': metrics.get('onset_mae'),
        'test_duration_mae': metrics.get('duration_mae'),
        'test_intensity_mape': metrics.get('intensity_mape'),
        'test_parse_success_rate': metrics.get('parse_success_rate'),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Compare old vs new
    old_path = os.path.join(CKPT_DIR, "test_summary.json")
    if os.path.exists(old_path):
        with open(old_path) as f:
            old = json.load(f)
        print("\n  OLD vs NEW:")
        for k in ["test_event_f1", "test_onset_mae", "test_apex_mae", "test_intensity_mape"]:
            old_v = old.get(k, "N/A")
            new_v = summary.get(k, "N/A")
            if isinstance(old_v, float) and isinstance(new_v, float):
                print(f"    {k}: {old_v:.6f} -> {new_v:.6f}")
            else:
                print(f"    {k}: {old_v} -> {new_v}")

if __name__ == "__main__":
    main()
