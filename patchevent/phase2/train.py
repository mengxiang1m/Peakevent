"""Phase-2 training entrypoint with modularized training components."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.optim import AdamW

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # TODO(R-future): migrate to patchevent package import

from patchevent.phase2.dataset import build_dataloaders
from patchevent.phase2.evaluate import evaluate_loader
from patchevent.phase2.model import build_model
from patchevent.losses.event_localization import compute_event_local_mse_loss_gpu
from patchevent.losses.intensity import compute_int_cls_loss, compute_intensity_loss
from patchevent.losses.value_losses import (
    compute_apex_mape_loss,
    compute_dense_value_loss,
    merge_value_loss,
)
from patchevent.optim.schedulers import build_warmup_cosine_scheduler
from patchevent.training.engine import train_one_epoch
from patchevent.training.eval_loop import validate
from patchevent.training.output_adapter import unpack_model_outputs
from patchevent.utils.reproducibility import set_seed

def parse_args():
    p = argparse.ArgumentParser(description='PatchEvent Phase-2 训练 (精简版)')

    # model
    p.add_argument('--decoder_type', type=str, default='hybrid',
                   choices=['hybrid', 'legacy_ar', 'legacy_detr'],
                   help='decoder类型: hybrid为新的DETR-AR融合主线; legacy_*仅用于旧checkpoint兼容')
    p.add_argument('--encoder_ckpt', type=str, default=None,
                   help='Legacy encoder checkpoint路径; hybrid端到端训练不需要')
    p.add_argument('--encoder_mode', type=str, default='frozen',
                   choices=['frozen', 'scratch', 'cnn', 'lstm', 'mlp'],
                   help='Legacy encoder模式; hybrid主线忽略该参数')
    p.add_argument('--encoder_lr', type=float, default=1e-5,
                   help='encoder学习率 (unfreeze/scratch模式)')
    p.add_argument('--d_model', type=int, default=128, help='模型维度')
    p.add_argument('--n_heads', type=int, default=4, help='注意力头数')
    p.add_argument('--n_layers', type=int, default=3, help='decoder层数')
    p.add_argument('--d_ff', type=int, default=256, help='FFN维度')
    p.add_argument('--dropout', type=float, default=0.2, help='dropout率')
    p.add_argument('--max_seq_len', type=int, default=64, help='最大生成序列长度')
    p.add_argument('--embed_dropout', type=float, default=0.15,
                   help='token+pos embedding后的dropout')
    p.add_argument('--event_schema', type=str, default='quad', choices=['quad', 'triplet'])
    p.add_argument('--prediction_mode', type=str, default=None, choices=['quad', 'triplet'],
                   help='显式切换4预测(quad)或3预测(triplet)，设置后覆盖event_schema')
    p.add_argument('--patch_len', type=int, default=8,
                   help='Hybrid backbone patch长度')
    p.add_argument('--patch_stride', type=int, default=4,
                   help='Hybrid backbone patch步长')
    p.add_argument('--backbone_layers', type=int, default=2,
                   help='Hybrid backbone Transformer层数')
    p.add_argument('--refine_layers', type=int, default=1,
                   help='Hybrid causal event refinement层数')
    p.add_argument('--hybrid_refine_order', type=str, default='onset',
                   choices=['onset', 'query'],
                   help='Hybrid refine token order: onset排序或原始query顺序')
    p.add_argument('--hybrid_no_causal_refine_mask', action='store_true',
                   help='Hybrid消融: 禁用refine模块内部causal mask')
    p.add_argument('--hybrid_nms_onset_radius', type=int, default=0,
                   help='Hybrid评估后处理: 按onset半径抑制重复query (0=禁用)')
    p.add_argument('--hybrid_time_head', type=str, default='structured',
                   choices=['structured', 'independent'],
                   help='Hybrid时间头: structured约束apex在onset-duration内部; independent兼容旧连续头')

    # E5模块开关
    p.add_argument('--unfreeze_last_n', type=int, default=0,
                   help='解冻encoder最后N层 (0=全冻结, 1=最后1层)')
    p.add_argument('--use_memory_pos', action='store_true',
                   help='Memory Position Encoding (MemPos)')
    p.add_argument('--use_self_attn_agg', action='store_true',
                   help='Self-Attention Aggregation (SA Agg)')
    p.add_argument('--use_input_decomp', action='store_true',
                   help='InputSeriesDecomposition (输入级分解)')
    p.add_argument('--input_decomp_mode', type=str, default='full',
                   choices=['full', 'trend_only', 'resid_only', 'no_inner_gate', 'no_fusion_gate'],
                   help='InputDecomp模式')
    p.add_argument('--use_raw_bypass', action='store_true',
                   help='Pre-Transformer patch bypass (concat raw patch features)')
    p.add_argument('--use_tuple_guided_value', action='store_true',
                   help='Triplet event decoding + tuple-guided dense future value prediction')
    p.add_argument('--use_independent_dense_int', action='store_true',
                   help='Triplet AR decoding + independent dense 96->pred_len + event-local MSE')
    p.add_argument('--use_apex_cond_int', action='store_true',
                   help='Apex-conditioned INT head using apex patch feature')
    p.add_argument('--use_int_direct_lookup', action='store_true',
                   help='Direct lookup: intensity = x[apex_pos]/mean at inference')
    p.add_argument('--use_decoupled_int_head', action='store_true',
                   help='Use an extra independent INT classification head')
    p.add_argument('--use_tuple_crossatt_int', action='store_true',
                   help='Use tuple-guided cross-attention head for intensity under quad decoding')
    p.add_argument('--use_series_stats', action='store_true',
                   help='Series Statistics Injection (mean/std/recent/max/min)')
    p.add_argument('--series_stats_recent_k', type=int, default=8,
                   help='Series stats中recent level使用最后k步均值')
    p.add_argument('--min_onset_spacing', type=int, default=0,
                   help='推理时onset最小间距约束')

    # Legacy AR-only switches (not used by the new hybrid mainline)
    p.add_argument('--no_pos_valid_mask', action='store_true',
                   help='消融: 禁用position valid mask约束')
    p.add_argument('--no_pos_aware_smoothing', action='store_true',
                   help='消融: 用标准label smoothing替代position-aware smoothing')
    p.add_argument('--no_causal_mask', action='store_true',
                   help='消融: 禁用causal mask (双向attention)')

    # Legacy DETR compatibility
    p.add_argument('--use_non_ar', action='store_true',
                   help='Legacy alias: 使用旧DETR分支; 新主线请使用默认hybrid')
    p.add_argument('--max_events', type=int, default=10,
                   help='最大event query数')
    p.add_argument('--exist_threshold', type=float, default=0.5,
                   help='DETR存在性阈值')

    # loss
    p.add_argument('--attr_loss_weights', type=str, default=None,
                   help='属性级loss权重, 如 "2.0,0.5,2.0,0.5"')
    p.add_argument('--label_smoothing', type=float, default=0.1,
                   help='position-aware label smoothing')
    p.add_argument('--use_int_regression', action='store_true',
                   help='用回归头替代INT分类 (Huber loss)')
    p.add_argument('--int_reg_weight', type=float, default=1.0,
                   help='intensity回归loss权重')
    p.add_argument('--use_int_ordinal_loss', action='store_true',
                   help='INT位置使用高斯soft-label序数损失')
    p.add_argument('--int_ordinal_sigma', type=float, default=2.0,
                   help='INT序数损失高斯sigma (bin单位)')
    p.add_argument('--int_head_loss', type=str, default='ordinal',
                   choices=['ordinal', 'ce', 'mix'],
                   help='Loss for decoupled INT head in finetune_int_head mode')
    p.add_argument('--value_loss_weight', type=float, default=1.0,
                   help='Weight for dense future value loss in tuple-guided mode')
    p.add_argument('--apex_weight', type=float, default=1.0,
                   help='Extra weight multiplier for apex position in event-local MSE (1.0=no boost)')
    p.add_argument('--apex_mape_weight', type=float, default=0.0,
                   help='Weight for auxiliary MAPE loss at apex positions (0=disabled)')
    p.add_argument('--intensity_loss_weight', type=float, default=1.0,
                   help='Weight for apex intensity regression loss from dedicated head')
    p.add_argument('--intensity_from_values', type=str, default='apex',
                   choices=['apex', 'span_max'],
                   help='How to recover event intensity from dense future values in tuple-guided mode')
    p.add_argument('--hybrid_objectness_weight', type=float, default=1.0)
    p.add_argument('--hybrid_no_object_weight', type=float, default=0.15)
    p.add_argument('--hybrid_onset_weight', type=float, default=2.0)
    p.add_argument('--hybrid_apex_weight', type=float, default=2.0)
    p.add_argument('--hybrid_duration_weight', type=float, default=0.5)
    p.add_argument('--hybrid_intensity_weight', type=float, default=0.5)
    p.add_argument('--hybrid_match_onset_weight', type=float, default=2.0)
    p.add_argument('--hybrid_match_apex_weight', type=float, default=2.0)
    p.add_argument('--hybrid_match_duration_weight', type=float, default=0.5)
    p.add_argument('--hybrid_match_intensity_weight', type=float, default=0.2)
    p.add_argument('--hybrid_object_focal_gamma', type=float, default=0.0)
    p.add_argument('--hybrid_refine_loss_weight', type=float, default=1.0)
    p.add_argument('--hybrid_proposal_loss_weight', type=float, default=0.3)
    p.add_argument('--hybrid_structure_loss_weight', type=float, default=0.0,
                   help='Hybrid结构约束loss: 惩罚apex落在onset-duration窗口外')

    # Event-Anchor / Count Head (方案A)
    p.add_argument('--hybrid_use_count_head', action='store_true',
                   help='Hybrid增强: 用refined query预测事件数以减少过多激活')
    p.add_argument('--hybrid_no_count_decoding', action='store_true',
                   help='Hybrid消融: 训练count head但推理不使用top-K事件数控制')
    p.add_argument('--hybrid_count_loss_weight', type=float, default=None,
                   help='Hybrid count head CE loss权重; 默认复用count_loss_weight')
    p.add_argument('--use_count_head', action='store_true',
                   help='Enable auxiliary count prediction head on encoder memory')
    p.add_argument('--max_event_count', type=int, default=12,
                   help='Max event count for count head classification')
    p.add_argument('--count_loss_weight', type=float, default=0.5,
                   help='Weight for count prediction CE loss')
    # Onset Focal Loss (YOLO-inspired)
    p.add_argument('--onset_focal_gamma', type=float, default=0.0,
                   help='Focal loss gamma for onset positions (0=disabled, 2=standard focal)')
    # Q-Bridge (Q-Former style Memory Bridge)
    p.add_argument('--use_qbridge', action='store_true',
                   help='Replace Memory Bridge with Q-Former style learnable queries')
    p.add_argument('--qbridge_n_queries', type=int, default=16,
                   help='Number of learnable query tokens in Q-Bridge')
    p.add_argument('--qbridge_n_layers', type=int, default=2,
                   help='Number of cross-attention layers in Q-Bridge')
    # Position Regression (方案C')
    p.add_argument('--use_position_regression', action='store_true',
                   help='Enable position regression head for onset/apex continuous prediction')
    p.add_argument('--pos_reg_weight', type=float, default=0.3,
                   help='Weight for position regression SmoothL1 loss')
    p.add_argument('--pos_reg_target', type=str, default='both',
                   choices=['both', 'onset', 'apex'],
                   help='Which positions to regress: both onset+apex, onset only, or apex only')

    # data
    p.add_argument('--series_path', type=str, required=True, help='时间序列CSV路径')
    p.add_argument('--events_path', type=str, required=True, help='事件JSONL路径')
    p.add_argument('--patch_labels_path', type=str, default=None,
                   help='patch labels CSV (联合训练模式)')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--window_stride', type=int, default=4)

    # training
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--train_epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=5e-4, help='decoder学习率')
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=15)

    # eval
    p.add_argument('--max_new_tokens', type=int, default=60)
    p.add_argument('--tolerance', type=int, default=3)

    # cross-domain norm override
    p.add_argument('--override_norm_mean', type=float, default=None,
                   help='Override encoder checkpoint norm_mean (for cross-domain eval)')
    p.add_argument('--override_norm_std', type=float, default=None,
                   help='Override encoder checkpoint norm_std (for cross-domain eval)')

    # misc
    p.add_argument('--output_dir', type=str, default='patchevent/phase2/checkpoints/exp1')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--finetune_int_head', action='store_true',
                   help='Freeze all params and finetune only int_cls_head')
    p.add_argument('--resume_from', type=str, default=None,
                   help='Checkpoint path to load before training')

    return p.parse_args()

def main():
    args = parse_args()
    if getattr(args, 'use_non_ar', False):
        args.decoder_type = 'legacy_detr'
    if getattr(args, 'prediction_mode', None) is not None:
        args.event_schema = args.prediction_mode
    if args.decoder_type == 'hybrid' and args.event_schema != 'quad':
        raise ValueError('Hybrid DETR-AR decoder requires event_schema=quad')
    if args.decoder_type in ('legacy_ar', 'legacy_detr') and not getattr(args, 'encoder_ckpt', None):
        raise ValueError(f'{args.decoder_type} requires --encoder_ckpt')
    if getattr(args, 'use_tuple_guided_value', False) and args.event_schema != 'triplet':
        print('[train] use_tuple_guided_value=True -> override event_schema=triplet')
        args.event_schema = 'triplet'
    if getattr(args, 'use_independent_dense_int', False) and args.event_schema != 'triplet':
        print('[train] use_independent_dense_int=True -> override event_schema=triplet')
        args.event_schema = 'triplet'
    if args.decoder_type == 'hybrid' and args.event_schema != 'quad':
        raise ValueError('Hybrid DETR-AR decoder requires event_schema=quad and does not use legacy tuple modes')
    if getattr(args, 'use_tuple_guided_value', False) and getattr(args, 'use_independent_dense_int', False):
        raise ValueError('use_tuple_guided_value and use_independent_dense_int are mutually exclusive')
    if getattr(args, 'use_tuple_guided_value', False) and args.decoder_type == 'legacy_detr':
        raise ValueError('use_tuple_guided_value is only implemented for the autoregressive decoder')
    if getattr(args, 'use_independent_dense_int', False) and args.decoder_type == 'legacy_detr':
        raise ValueError('use_independent_dense_int is only implemented for the autoregressive decoder')
    if getattr(args, 'use_tuple_crossatt_int', False) and args.event_schema != 'quad':
        raise ValueError('use_tuple_crossatt_int requires quad prediction mode')
    if getattr(args, 'use_tuple_crossatt_int', False) and args.decoder_type == 'legacy_detr':
        raise ValueError('use_tuple_crossatt_int is only implemented for the autoregressive decoder')
    if getattr(args, 'event_schema', 'quad') != 'quad' and args.decoder_type == 'legacy_detr':
        raise ValueError('legacy_detr currently requires event_schema=quad')
    if getattr(args, 'use_tuple_guided_value', False) and getattr(args, 'finetune_int_head', False):
        raise ValueError('finetune_int_head is incompatible with tuple-guided value mode')
    if getattr(args, 'use_independent_dense_int', False) and getattr(args, 'finetune_int_head', False):
        raise ValueError('finetune_int_head is incompatible with independent dense intensity mode')
    if getattr(args, 'use_tuple_crossatt_int', False) and getattr(args, 'finetune_int_head', False):
        raise ValueError('finetune_int_head is incompatible with tuple cross-att intensity mode')
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'[train] device = {device}')

    # 构建数据
    print('[train] building dataloaders...')
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        series_path=args.series_path,
        events_path=args.events_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        event_schema=args.event_schema,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        patch_labels_path=getattr(args, 'patch_labels_path', None),
    )
    args.norm_mean = (
        args.override_norm_mean
        if getattr(args, 'override_norm_mean', None) is not None
        else meta['mean']
    )
    args.norm_std = (
        args.override_norm_std
        if getattr(args, 'override_norm_std', None) is not None
        else meta['std']
    )

    # 构建模型
    print('[train] building model...')
    model = build_model(args).to(device)

    resume_from = getattr(args, 'resume_from', None)
    if resume_from:
        print(f'[train] loading checkpoint: {resume_from}')
        ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt)
        model_state = model.state_dict()
        filtered_state = {}
        skipped_shape = []
        for k, v in state.items():
            if k not in model_state:
                continue
            if model_state[k].shape != v.shape:
                skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                continue
            filtered_state[k] = v
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f'[train] loaded with missing={len(missing)} unexpected={len(unexpected)} '
              f'shape_skipped={len(skipped_shape)}')
        if skipped_shape:
            for name, src_shape, dst_shape in skipped_shape[:10]:
                print(f'  [shape-skip] {name}: ckpt{src_shape} -> model{dst_shape}')
            if len(skipped_shape) > 10:
                print(f'  ... and {len(skipped_shape) - 10} more shape mismatches')

    if getattr(args, 'finetune_int_head', False):
        if not hasattr(model, 'int_cls_head'):
            raise ValueError('finetune_int_head requires --use_decoupled_int_head')
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.int_cls_head.parameters():
            p.requires_grad_(True)
        n_ft = sum(p.numel() for p in model.int_cls_head.parameters())
        print(f'[train] finetune_int_head=True, trainable int_cls_head params: {n_ft:,}')

    # 优化器
    encoder_lr = getattr(args, 'encoder_lr', None)
    param_groups = model.get_param_groups(lr=args.lr, encoder_lr=encoder_lr)
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.train_epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)

    # 训练循环
    print('[train] start training...')
    train_log: list[dict] = []
    best_val_loss = float('inf')
    best_val_epoch = -1
    best_ckpt_path = os.path.join(args.output_dir, 'best.pth')
    patience_cnt = 0
    global_step = 0

    for epoch in range(1, args.train_epochs + 1):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args, global_step
        )
        val_loss = validate(model, val_loader, device, args)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            patience_cnt = 0
            save_encoder = (args.encoder_mode != 'frozen'
                            or getattr(args, 'unfreeze_last_n', 0) > 0)
            torch.save({
                'model': {k: v for k, v in model.state_dict().items()
                          if save_encoder or not k.startswith('encoder.')},
                'args': vars(args),
                'norm_mean': model.norm_mean.cpu(),
                'norm_std': model.norm_std.cpu(),
            }, best_ckpt_path)
            marker = ' *'
        else:
            patience_cnt += 1
            marker = ''

        train_log.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'lr': lr_now, 'global_step': global_step, 'time_s': elapsed,
        })
        print(f'Epoch {epoch:3d}/{args.train_epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'lr={lr_now:.2e}  time={elapsed:.1f}s{marker}')

        if patience_cnt >= args.patience:
            print(f'  Early stopping at epoch {epoch}.')
            break

    # 保存训练日志
    log_path = os.path.join(args.output_dir, 'train_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(train_log, f, indent=2, ensure_ascii=False)
    print(f'[train] best val_loss={best_val_loss:.4f} at epoch={best_val_epoch}')

    # 加载best checkpoint做测试评估
    print('[train] loading best checkpoint for test evaluation...')
    ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)

    metrics = evaluate_loader(
        model, test_loader, device,
        max_new_tokens=args.max_new_tokens,
        tolerance=args.tolerance,
        verbose=True,
        save_dir=os.path.join(args.output_dir, 'test_eval'),
        intensity_scale=meta['mean'],
        hybrid_nms_onset_radius=args.hybrid_nms_onset_radius,
    )

    # 保存测试摘要
    test_summary = {
        'best_val_epoch': best_val_epoch,
        'best_val_loss': best_val_loss,
        'test_event_f1': metrics.get('event_f1'),
        'test_event_precision': metrics.get('event_precision'),
        'test_event_recall': metrics.get('event_recall'),
        'test_apex_mae': metrics.get('apex_mae'),
        'test_onset_mae': metrics.get('onset_mae'),
        'test_duration_mae': metrics.get('duration_mae'),
        'test_intensity_mape': metrics.get('intensity_mape'),
        'test_parse_success_rate': metrics.get('parse_success_rate'),
    }
    summary_path = os.path.join(args.output_dir, 'test_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False)
    print(f'[train] test summary saved: {summary_path}')
    print('[train] done.')


if __name__ == '__main__':
    main()
