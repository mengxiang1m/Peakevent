#!/usr/bin/env python
"""
Phase 1: Patch Encoder Pretraining — training entry point.

Usage (from repo root):
    conda activate tslib_findpeaks
    python patchevent/phase1/train.py

Or with explicit args:
    python patchevent/phase1/train.py \\
        --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \\
        --train_epochs 20 --batch_size 64 --learning_rate 1e-3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import random

import numpy as np
import torch
import torch.optim as optim

# ── path setup ────────────────────────────────────────────────────────────────
_pkg  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_pkg)
if _root not in sys.path:
    sys.path.insert(0, _root)  # TODO(R-future): migrate to patchevent package import
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)  # TODO(R-future): migrate to patchevent package import

from dataset import build_dataloaders, build_multi_domain_dataloaders, WlelPatchDataset
from model   import PatchEncoder
from loss    import Phase1Loss


def adjust_learning_rate(optimizer, epoch, args):
    """Cosine annealing learning rate schedule (内联自utils/tools.py)。"""
    import math
    if args.lradj == 'cosine':
        lr = args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))
    elif args.lradj == 'type1':
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
    else:
        lr = args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'Updating learning rate to {lr}')

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print('[Warning] sklearn not found — F1/P/R metrics will be skipped')

_DOMAIN_PRESETS = {
    'wlel': {
        'series_path': 'dataset/wlel/event_v1/data/wlel_event_series_v1.csv',
        'labels_path': 'dataset/wlel/event_v1/data/wlel_patch_labels_v1.csv',
    },
    'ett': {
        'series_path': 'dataset/ett/event_v1/data/ett_event_series_v1.csv',
        'labels_path': 'dataset/ett/event_v1/data/ett_patch_labels_v1.csv',
    },
    'electricity': {
        'series_path': 'dataset/electricity/event_v1/data/elc_event_series_v1.csv',
        'labels_path': 'dataset/electricity/event_v1/data/elc_patch_labels_v1.csv',
    },
    # Backward-compatible alias used by ablation launcher scripts.
    'elc': {
        'series_path': 'dataset/electricity/event_v1/data/elc_event_series_v1.csv',
        'labels_path': 'dataset/electricity/event_v1/data/elc_patch_labels_v1.csv',
    },
}


def _preset_cfg(name: str) -> dict:
    key = str(name).lower()
    if key not in _DOMAIN_PRESETS:
        raise ValueError(f'Unknown domain "{name}". Available: {list(_DOMAIN_PRESETS.keys())}')
    cfg = _DOMAIN_PRESETS[key]
    return {'name': key, 'series_path': cfg['series_path'], 'labels_path': cfg['labels_path']}


def _build_domain_configs(args) -> list[dict]:
    """Resolve training domain configs from CLI args.

    Priority:
    1) Explicit --series_path/--labels_path (+ optional --extra_* legacy format)
    2) Domain presets via --single_domain / --multi_domains
    """
    mode = str(args.train_mode).lower()
    if mode not in ('single', 'multi'):
        raise ValueError(f'Unsupported train_mode={args.train_mode}')

    # Legacy explicit path mode: primary + extras
    has_primary_explicit = bool(args.series_path) and bool(args.labels_path)
    has_extra = bool(args.extra_series_paths) or bool(args.extra_labels_paths)
    if has_extra:
        if not has_primary_explicit:
            # Fallback primary to single_domain preset if primary path not explicitly provided.
            primary = _preset_cfg(args.single_domain)
        else:
            primary = {
                'name': str(args.single_domain).lower() if args.single_domain else 'primary',
                'series_path': args.series_path,
                'labels_path': args.labels_path,
            }
        if not (args.extra_series_paths and args.extra_labels_paths):
            raise ValueError('Both --extra_series_paths and --extra_labels_paths are required together.')
        if len(args.extra_series_paths) != len(args.extra_labels_paths):
            raise ValueError('Length mismatch: --extra_series_paths vs --extra_labels_paths.')

        cfgs = [primary]
        for i, (sp, lp) in enumerate(zip(args.extra_series_paths, args.extra_labels_paths)):
            cfgs.append({'name': f'extra_{i}', 'series_path': sp, 'labels_path': lp})
        return cfgs if mode == 'multi' else [cfgs[0]]

    # Preset mode
    if mode == 'single':
        if has_primary_explicit:
            return [{
                'name': str(args.single_domain).lower() if args.single_domain else 'single',
                'series_path': args.series_path,
                'labels_path': args.labels_path,
            }]
        return [_preset_cfg(args.single_domain)]

    domains = list(args.multi_domains or [])
    if not domains:
        raise ValueError('train_mode=multi requires --multi_domains or legacy --extra_* paths.')
    return [_preset_cfg(name) for name in domains]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _EarlyStopper:
    """Minimal early stopping — stops when val_loss fails to improve."""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter  = 0
        self.best     = float('inf')
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f'  EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _evaluate(model: PatchEncoder, loader, loss_fn: Phase1Loss, device, verbose=False):
    """Run full evaluation pass with per-task metrics.
    Returns (avg_loss, avg_components, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    comps = {'bce': 0., 'd': 0., 'kl': 0., 'off': 0.}

    # 收集所有预测和标签
    ha_prob_all, ha_true_all = [], []
    d_pred_all, d_true_all = [], []
    phase_pred_all, phase_true_all = [], []
    off_pred_all, off_true_all = [], []
    apex_mask_all = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch['x'])
            loss, lc = loss_fn(pred, batch)
            total_loss += loss.item()
            for k in comps:
                comps[k] += lc[k]

            ha_prob_all.append(pred['has_apex'].cpu().numpy().ravel())
            ha_true_all.append(batch['has_apex'].cpu().numpy().ravel())
            d_pred_all.append(pred['d_to_apex'].cpu().numpy().ravel())
            d_true_all.append(batch['min_d'].cpu().numpy().ravel())
            phase_pred_all.append(pred['phase_dist'].cpu().numpy().reshape(-1, 4))
            phase_true_all.append(batch['phase_dist'].cpu().numpy().reshape(-1, 4))

            mask = batch['has_apex'].cpu().numpy().ravel() > 0.5
            apex_mask_all.append(mask)
            off_pred_all.append(pred['apex_offset'].cpu().numpy().reshape(-1, 8))
            off_true_all.append(batch['apex_offset'].cpu().numpy().ravel())

    n = max(len(loader), 1)
    avg_loss = total_loss / n
    avg_comps = {k: v / n for k, v in comps.items()}

    # ── Task 1: has_apex (二分类) ──
    ha_prob = np.concatenate(ha_prob_all)
    ha_true = np.concatenate(ha_true_all)
    ha_pred = (ha_prob > 0.5).astype(float)
    metrics = {}
    if _HAS_SKLEARN:
        from sklearn.metrics import accuracy_score, roc_auc_score
        metrics['has_apex_f1'] = float(f1_score(ha_true, ha_pred, zero_division=0))
        metrics['has_apex_precision'] = float(precision_score(ha_true, ha_pred, zero_division=0))
        metrics['has_apex_recall'] = float(recall_score(ha_true, ha_pred, zero_division=0))
        metrics['has_apex_accuracy'] = float(accuracy_score(ha_true, ha_pred))
        try:
            metrics['has_apex_auc'] = float(roc_auc_score(ha_true, ha_prob))
        except ValueError:
            metrics['has_apex_auc'] = 0.0

    # ── Task 2: d_to_apex (回归) ──
    d_pred = np.concatenate(d_pred_all)
    d_true = np.concatenate(d_true_all)
    metrics['d_to_apex_mae'] = float(np.abs(d_pred - d_true).mean())
    metrics['d_to_apex_rmse'] = float(np.sqrt(((d_pred - d_true) ** 2).mean()))
    ss_res = ((d_pred - d_true) ** 2).sum()
    ss_tot = ((d_true - d_true.mean()) ** 2).sum()
    metrics['d_to_apex_r2'] = float(1 - ss_res / max(ss_tot, 1e-8))

    # ── Task 3: phase_dist (4类软分布) ──
    # 注意: phase_dist是软分布标签(bg dominant ~99.7%), 不适合argmax硬分类评估
    phase_pred = np.concatenate(phase_pred_all)
    phase_true = np.concatenate(phase_true_all)
    phase_names = ['bg', 'rising', 'apex', 'falling']

    # 3a. Per-class MAE (软分布预测误差)
    for i, name in enumerate(phase_names):
        metrics[f'phase_mae_{name}'] = float(np.abs(phase_pred[:, i] - phase_true[:, i]).mean())

    # 3b. Cosine Similarity (分布形状相似度)
    dot = (phase_pred * phase_true).sum(axis=1)
    norm_p = np.sqrt((phase_pred ** 2).sum(axis=1)).clip(1e-8)
    norm_t = np.sqrt((phase_true ** 2).sum(axis=1)).clip(1e-8)
    metrics['phase_cosine_sim'] = float((dot / (norm_p * norm_t)).mean())

    # 3c. KL divergence
    phase_true_safe = np.clip(phase_true, 1e-8, 1.0)
    phase_pred_safe = np.clip(phase_pred, 1e-8, 1.0)
    kl_per_sample = (phase_true_safe * np.log(phase_true_safe / phase_pred_safe)).sum(axis=1)
    metrics['phase_kl_mean'] = float(kl_per_sample.mean())

    # 3d. 非背景阶段检测 (rising/apex/falling任一>0.2视为事件相关patch)
    event_thresh = 0.2
    true_event = (phase_true[:, 1:].max(axis=1) > event_thresh)
    pred_event = (phase_pred[:, 1:].max(axis=1) > event_thresh)
    if _HAS_SKLEARN:
        metrics['phase_event_f1'] = float(f1_score(true_event, pred_event, zero_division=0))
        metrics['phase_event_precision'] = float(precision_score(true_event, pred_event, zero_division=0))
        metrics['phase_event_recall'] = float(recall_score(true_event, pred_event, zero_division=0))

    # 3e. 整体分布准确率 (argmax一致性, 作为参考)
    if _HAS_SKLEARN:
        from sklearn.metrics import accuracy_score
        metrics['phase_argmax_accuracy'] = float(accuracy_score(
            phase_true.argmax(axis=1), phase_pred.argmax(axis=1)))

    # ── Task 4: apex_offset (8类分类, 仅apex patch) ──
    apex_mask = np.concatenate(apex_mask_all)
    off_pred = np.concatenate(off_pred_all)
    off_true = np.concatenate(off_true_all)
    if apex_mask.any():
        off_pred_cls = off_pred[apex_mask].argmax(axis=1)
        off_true_cls = off_true[apex_mask].astype(int)
        off_true_cls = np.clip(off_true_cls, 0, 7)
        metrics['offset_accuracy'] = float((off_pred_cls == off_true_cls).mean())
        # top-2 accuracy
        off_top2 = off_pred[apex_mask].argsort(axis=1)[:, -2:]
        top2_hit = np.array([off_true_cls[i] in off_top2[i] for i in range(len(off_true_cls))])
        metrics['offset_top2_accuracy'] = float(top2_hit.mean())
        # MAE (位置误差)
        metrics['offset_mae'] = float(np.abs(off_pred_cls - off_true_cls).mean())
    else:
        metrics['offset_accuracy'] = 0.0
        metrics['offset_top2_accuracy'] = 0.0
        metrics['offset_mae'] = 0.0

    # 兼容旧接口
    metrics['f1'] = metrics.get('has_apex_f1', 0.0)
    metrics['precision'] = metrics.get('has_apex_precision', 0.0)
    metrics['recall'] = metrics.get('has_apex_recall', 0.0)

    if verbose:
        print(f"  [has_apex]    F1={metrics['has_apex_f1']:.3f}  P={metrics['has_apex_precision']:.3f}  "
              f"R={metrics['has_apex_recall']:.3f}  AUC={metrics.get('has_apex_auc',0):.3f}  "
              f"Acc={metrics['has_apex_accuracy']:.3f}")
        print(f"  [d_to_apex]   MAE={metrics['d_to_apex_mae']:.3f}  RMSE={metrics['d_to_apex_rmse']:.3f}  "
              f"R2={metrics['d_to_apex_r2']:.3f}")
        print(f"  [phase_dist]  CosSim={metrics['phase_cosine_sim']:.3f}  KL={metrics['phase_kl_mean']:.4f}  "
              f"EventF1={metrics.get('phase_event_f1',0):.3f}  "
              f"MAE(bg/rise/apex/fall)="
              f"{metrics['phase_mae_bg']:.3f}/{metrics['phase_mae_rising']:.3f}/"
              f"{metrics['phase_mae_apex']:.3f}/{metrics['phase_mae_falling']:.3f}")
        print(f"  [apex_offset] Acc={metrics['offset_accuracy']:.3f}  Top2={metrics['offset_top2_accuracy']:.3f}  "
              f"MAE={metrics['offset_mae']:.2f}")

    return avg_loss, avg_comps, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── seed ──────────────────────────────────────────────────────────────────
    if getattr(args, 'seed', None) is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f'[Train] seed: {args.seed}')

    # ── device ────────────────────────────────────────────────────────────────
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'[Train] device: {device}')

    # ── data ──────────────────────────────────────────────────────────────────
    domain_configs = _build_domain_configs(args)
    print('[Train] domains:')
    for i, cfg in enumerate(domain_configs):
        print(f'  - [{i}] {cfg["name"]}:')
        print(f'      series={cfg["series_path"]}')
        print(f'      labels={cfg["labels_path"]}')

    primary_cfg = domain_configs[0]
    domain_norms = None
    domain_eval_configs = None

    if len(domain_configs) > 1:
        train_loader, val_loader, test_loader, domain_norms = build_multi_domain_dataloaders(
            domain_configs    = domain_configs,
            seq_len           = args.seq_len,
            patch_len         = args.patch_len,
            stride            = args.stride,
            batch_size        = args.batch_size,
            num_workers       = args.num_workers,
            log1p_d           = bool(args.log1p_d),
        )
        mean, std = domain_norms[0]  # primary domain norm for checkpoint compatibility
        domain_eval_configs = domain_configs
    else:
        train_loader, val_loader, test_loader, mean, std = build_dataloaders(
            series_path       = primary_cfg['series_path'],
            patch_labels_path = primary_cfg['labels_path'],
            seq_len           = args.seq_len,
            patch_len         = args.patch_len,
            stride            = args.stride,
            batch_size        = args.batch_size,
            num_workers       = args.num_workers,
            log1p_d           = bool(args.log1p_d),
        )
    print(f'[Train] batches — train:{len(train_loader)}  val:{len(val_loader)}  test:{len(test_loader)}')

    # ── model ─────────────────────────────────────────────────────────────────
    mask_ratio = getattr(args, 'mask_ratio', 0.0)
    model = PatchEncoder(
        seq_len   = args.seq_len,
        patch_len = args.patch_len,
        stride    = args.stride,
        d_model   = args.d_model,
        n_heads   = args.n_heads,
        e_layers  = args.e_layers,
        d_ff      = args.d_ff,
        dropout   = args.dropout,
        mask_ratio = mask_ratio,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Train] PatchEncoder params: {n_params:,}')

    # ── loss & optimiser ──────────────────────────────────────────────────────
    lambda_recon = getattr(args, 'lambda_recon', 0.0)
    lambda_npp = getattr(args, 'lambda_npp', 0.0)
    loss_fn   = Phase1Loss(
        lambda_bce = args.lambda_bce,
        lambda_d   = args.lambda_d,
        lambda_kl  = args.lambda_kl,
        lambda_off = args.lambda_off,
        lambda_recon = lambda_recon,
        lambda_npp = lambda_npp,
        kl_phase_weights = [1.0, 1.0, args.kl_apex_weight, args.kl_falling_weight],
    )
    use_mpr = mask_ratio > 0 and lambda_recon > 0
    if use_mpr:
        print(f'[Train] MPR enabled: mask_ratio={mask_ratio}, lambda_recon={lambda_recon}')
    if lambda_npp > 0:
        print(f'[Train] NPP enabled: lambda_npp={lambda_npp}')
    optimizer = optim.Adam(
        model.parameters(),
        lr           = args.learning_rate,
        weight_decay = args.weight_decay,
    )

    # ── output directory ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'best_model.pth')

    # ── training loop ─────────────────────────────────────────────────────────
    stopper = _EarlyStopper(patience=args.patience)
    history: dict = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_loss = float('inf')

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pred = model(batch['x'], use_mask=use_mpr)
            loss, _ = loss_fn(pred, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / max(len(train_loader), 1)
        val_loss, val_comps, val_det = _evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        history['train_loss'].append(avg_train)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_det['f1'])

        print(
            f'Epoch {epoch:03d}/{args.train_epochs}  '
            f'train={avg_train:.4f}  val={val_loss:.4f}  '
            f'(bce={val_comps["bce"]:.3f} d={val_comps["d"]:.3f} '
            f'kl={val_comps["kl"]:.3f} off={val_comps["off"]:.3f})  '
            f'F1={val_det["f1"]:.3f} P={val_det["precision"]:.3f} R={val_det["recall"]:.3f}  '
            f'{elapsed:.1f}s'
        )

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'args':        vars(args),
                'mean':        mean,
                'std':         std,
                'val_loss':    val_loss,
                'val_f1':      val_det['f1'],
            }
            if domain_norms is not None:
                save_dict['domain_norms'] = domain_norms
            torch.save(save_dict, ckpt_path)
            print(f'  [saved] best checkpoint (val_loss={val_loss:.4f})')

        adjust_learning_rate(optimizer, epoch, args)
        if stopper.step(val_loss):
            print('Early stopping triggered.')
            break

    # ── test evaluation ───────────────────────────────────────────────────────
    print('\n=== Final Test Evaluation (best checkpoint) ===')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    # 混合test评估
    test_loss, test_comps, test_metrics = _evaluate(model, test_loader, loss_fn, device, verbose=True)
    print(f'test_loss={test_loss:.4f}')

    # 联合训练时分域测试
    domain_test_metrics = {}
    if domain_norms is not None:
        all_configs = domain_eval_configs or []
        for idx, cfg in enumerate(all_configs):
            dm, ds = domain_norms[idx]
            domain_test = WlelPatchDataset(
                cfg['series_path'], cfg['labels_path'], 'test',
                args.seq_len, args.patch_len, args.stride,
                mean=dm, std=ds, log1p_d=bool(args.log1p_d),
            )
            domain_loader = torch.utils.data.DataLoader(
                domain_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
            )
            print(f'\n--- Domain: {cfg["name"]} (n={len(domain_test)}) ---')
            _, _, dm_metrics = _evaluate(model, domain_loader, loss_fn, device, verbose=True)
            domain_test_metrics[cfg['name']] = dm_metrics

    # ── save results ──────────────────────────────────────────────────────────
    history.update({
        'test_loss':      test_loss,
        'test_comps':     test_comps,
        'test_metrics':   test_metrics,
    })
    if domain_test_metrics:
        history['domain_test_metrics'] = domain_test_metrics
    # 兼容旧接口
    history['test_f1'] = test_metrics.get('f1', 0.0)
    history['test_precision'] = test_metrics.get('precision', 0.0)
    history['test_recall'] = test_metrics.get('recall', 0.0)

    hist_path = os.path.join(args.output_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'History saved → {hist_path}')

    return model, ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='Phase 1: Patch Encoder Pretraining')
    # train mode / domain presets
    p.add_argument('--train_mode', type=str, default='single', choices=['single', 'multi'],
                   help='single: one domain only; multi: joint training over multiple domains')
    p.add_argument('--single_domain', type=str, default='wlel',
                   choices=['wlel', 'ett', 'electricity', 'elc'],
                   help='Domain preset for single-domain training')
    p.add_argument('--multi_domains', type=str, nargs='+',
                   default=['wlel', 'ett', 'electricity'],
                   help='Domain presets for multi-domain training')

    # explicit data paths (optional; override presets)
    p.add_argument('--series_path', type=str, default=None,
                   help='Primary series CSV path; if omitted, resolved from preset domain')
    p.add_argument('--labels_path', type=str, default=None,
                   help='Primary patch-label CSV path; if omitted, resolved from preset domain')
    # legacy extra paths (still supported)
    p.add_argument('--extra_series_paths', type=str, nargs='*', default=None,
                   help='Legacy: additional series CSV paths for multi-domain joint training')
    p.add_argument('--extra_labels_paths', type=str, nargs='*', default=None,
                   help='Legacy: additional patch-label CSV paths (must match extra_series_paths)')
    p.add_argument('--output_dir',  type=str,
                   default='patchevent/phase1/checkpoints')

    # patch / window config
    p.add_argument('--seq_len',   type=int, default=96)
    p.add_argument('--patch_len', type=int, default=8)
    p.add_argument('--stride',    type=int, default=4)

    # model architecture
    p.add_argument('--d_model',  type=int,   default=128)
    p.add_argument('--n_heads',  type=int,   default=4)
    p.add_argument('--e_layers', type=int,   default=2)
    p.add_argument('--d_ff',     type=int,   default=256)
    p.add_argument('--dropout',  type=float, default=0.1)

    # loss weights
    p.add_argument('--lambda_bce', type=float, default=1.0)
    p.add_argument('--lambda_d',   type=float, default=0.5)
    p.add_argument('--lambda_kl',  type=float, default=1.0)
    p.add_argument('--lambda_off', type=float, default=0.5)
    p.add_argument('--kl_apex_weight',    type=float, default=2.0,
                   help='KL weight for Apex phase (default 2.0 to emphasise rare phase)')
    p.add_argument('--kl_falling_weight', type=float, default=2.0,
                   help='KL weight for Falling phase (default 2.0 to emphasise rare phase)')
    p.add_argument('--mask_ratio', type=float, default=0.0,
                   help='MPR: fraction of patches to mask (0=disabled, 0.4=PatchTST default)')
    p.add_argument('--lambda_recon', type=float, default=0.0,
                   help='MPR: weight for masked patch reconstruction loss (0=disabled)')
    p.add_argument('--lambda_npp', type=float, default=0.0,
                   help='NPP: weight for next-patch prediction loss (0=disabled)')

    # training
    p.add_argument('--train_epochs',  type=int,   default=50)
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--num_workers',   type=int,   default=0)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--patience',      type=int,   default=5)
    p.add_argument('--lradj',         type=str,   default='cosine')
    p.add_argument('--log1p_d',       type=int,   default=1,
                   help='Apply log1p to d_to_apex labels (1=yes, 0=no)')
    p.add_argument('--apex_threshold', type=float, default=0.5,
                   help='Decision threshold for has_apex binary prediction')
    p.add_argument('--use_gpu',       type=int,   default=1)
    p.add_argument('--seed',           type=int,   default=None,
                   help='Random seed for reproducibility (None=no seeding)')

    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    train(args)

