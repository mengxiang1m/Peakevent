#!/usr/bin/env python
"""
Evaluate Phase 1 ablation checkpoints on WLEL test set.
Extracts ALL metrics: has_apex F1, d_to_apex MAE, phase CosSim, offset Acc.
Also evaluates the baseline (wlel/s{42,123,456}).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

_pkg = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_pkg)
if _root not in sys.path:
    sys.path.insert(0, _root)  # TODO(R-future): migrate to patchevent package import
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)  # TODO(R-future): migrate to patchevent package import

from dataset import build_dataloaders
from model import PatchEncoder
from loss import Phase1Loss


SEEDS = [42, 123, 456]
CKPT_BASE = os.path.join(_pkg, 'checkpoints')

CONFIGS = {
    'P0_baseline': 'wlel',
    'P1a_wo_has_apex': 'wlel_ablation/P1a_wo_has_apex',
    'P1b_wo_distance': 'wlel_ablation/P1b_wo_distance',
    'P1c_wo_phase': 'wlel_ablation/P1c_wo_phase',
    'P1d_wo_offset': 'wlel_ablation/P1d_wo_offset',
}

METRIC_KEYS = [
    'has_apex_f1', 'd_to_apex_mae', 'd_to_apex_r2',
    'phase_cosine_sim', 'phase_kl_mean', 'phase_event_f1',
    'offset_accuracy', 'offset_top2_accuracy',
]


def evaluate(model, loader, loss_fn, device):
    """Reuse the _evaluate logic from train.py."""
    from train import _evaluate
    _, _, metrics = _evaluate(model, loader, loss_fn, device, verbose=False)
    return metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for config_name, subdir in CONFIGS.items():
        results[config_name] = {'per_seed': {}, 'summary': {}}
        seed_metrics = []

        for seed in SEEDS:
            ckpt_path = os.path.join(CKPT_BASE, subdir, f's{seed}', 'best_model.pth')
            if not os.path.exists(ckpt_path):
                print(f'[SKIP] {config_name} s{seed}: {ckpt_path} not found')
                continue

            ckpt = torch.load(ckpt_path, map_location='cpu')
            args = ckpt['args']

            # Build test loader
            _, _, test_loader, mean, std = build_dataloaders(
                series_path='dataset/wlel/event_v1/data/wlel_event_series_v1.csv',
                patch_labels_path='dataset/wlel/event_v1/data/wlel_patch_labels_v1.csv',
                seq_len=args.get('seq_len', 96),
                patch_len=args.get('patch_len', 8),
                stride=args.get('stride', 4),
                batch_size=64,
                num_workers=0,
                log1p_d=bool(args.get('log1p_d', 1)),
            )

            # Build model
            model = PatchEncoder(
                seq_len=args.get('seq_len', 96),
                patch_len=args.get('patch_len', 8),
                stride=args.get('stride', 4),
                d_model=args.get('d_model', 128),
                n_heads=args.get('n_heads', 4),
                e_layers=args.get('e_layers', 2),
                d_ff=args.get('d_ff', 256),
                dropout=args.get('dropout', 0.1),
            ).to(device)
            model.load_state_dict(ckpt['model_state'], strict=False)

            # Build loss (use saved lambda values)
            loss_fn = Phase1Loss(
                lambda_bce=args.get('lambda_bce', 1.0),
                lambda_d=args.get('lambda_d', 0.5),
                lambda_kl=args.get('lambda_kl', 1.0),
                lambda_off=args.get('lambda_off', 0.5),
            )

            metrics = evaluate(model, test_loader, loss_fn, device)
            seed_result = {k: float(metrics.get(k, 0.0)) for k in METRIC_KEYS}
            results[config_name]['per_seed'][str(seed)] = seed_result
            seed_metrics.append(seed_result)

            print(f'[DONE] {config_name} s{seed}: '
                  f'OffAcc={seed_result["offset_accuracy"]:.3f} '
                  f'CosSim={seed_result["phase_cosine_sim"]:.3f} '
                  f'dMAE={seed_result["d_to_apex_mae"]:.3f} '
                  f'haF1={seed_result["has_apex_f1"]:.3f}')

        # Compute summary
        if seed_metrics:
            for k in METRIC_KEYS:
                vals = [s[k] for s in seed_metrics]
                results[config_name]['summary'][k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                }

    # Save results
    out_path = os.path.join(CKPT_BASE, 'p1_ablation_full_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')

    # Print summary table
    print('\n' + '=' * 100)
    print(f'{"Config":<22} {"has_apex F1":>12} {"dMAE":>10} {"CosSim":>10} {"PhaseKL":>10} '
          f'{"OffAcc":>10} {"OffTop2":>10}')
    print('-' * 100)
    for config_name in CONFIGS:
        s = results[config_name]['summary']
        if not s:
            continue
        print(f'{config_name:<22} '
              f'{s["has_apex_f1"]["mean"]:.3f}±{s["has_apex_f1"]["std"]:.3f}  '
              f'{s["d_to_apex_mae"]["mean"]:.3f}±{s["d_to_apex_mae"]["std"]:.3f}  '
              f'{s["phase_cosine_sim"]["mean"]:.3f}±{s["phase_cosine_sim"]["std"]:.3f}  '
              f'{s["phase_kl_mean"]["mean"]:.4f}±{s["phase_kl_mean"]["std"]:.4f}  '
              f'{s["offset_accuracy"]["mean"]:.3f}±{s["offset_accuracy"]["std"]:.3f}  '
              f'{s["offset_top2_accuracy"]["mean"]:.3f}±{s["offset_top2_accuracy"]["std"]:.3f}')
    print('=' * 100)


if __name__ == '__main__':
    main()
