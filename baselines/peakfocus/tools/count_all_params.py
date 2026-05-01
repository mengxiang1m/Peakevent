"""
Count parameters of PeakFocus (proposed_model) and all baselines using the
exact hyperparameters recorded in scripts/peak_detect_based_on_LTF/load_data/*.sh.

Run from the Code/ directory:
    python tools/count_all_params.py
"""
import os
import sys
import argparse
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_basic import Exp_Basic


def default_args():
    """Mirror run.py defaults for every arg the models may touch."""
    d = dict(
        task_name="peak_detect_ltf",
        is_training=1,
        model_id="test",
        model="proposed_model",
        data="load_data_mixed",
        root_path="./dataset/load_data/hf_load_data/",
        data_path="hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv",
        features="S",
        target="OT",
        input_col="value_max",
        target_col="value_max",
        freq="t",
        checkpoints="./checkpoints/",
        seq_len=168,
        label_len=48,
        pred_len=336,
        seasonal_patterns="Monthly",
        inverse=False,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=96,
        output_attention=0,
        # CycleNet
        cycle=168,
        model_type="mlp",
        use_revin=1,
        # STID
        num_nodes=1,
        node_dim=32,
        embed_dim=64,
        num_layer=3,
        temp_dim_tid=32,
        temp_dim_diw=32,
        time_of_day_size=24,
        day_of_week_size=7,
        if_T_i_D=1,
        if_D_i_W=1,
        if_node=0,
        # optimization
        num_workers=0,
        itr=5,
        train_epochs=10,
        batch_size=64,
        patience=3,
        learning_rate=0.0001,
        des="test",
        loss="MSE",
        loss_alpha=0.5,
        lradj="type1",
        use_amp=False,
        # GPU
        use_gpu=False,
        gpu=0,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices="0",
        # projector
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        # augmentation
        augmentation_ratio=0,
        seed=2,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        # TimeXer
        patch_len=16,
        # MetaEformer
        d_low=10,
        mpp_update=50,
        sim_num=10,
        threshold=1.0,
        mpp_size=350,
        mp_len=96,
        kernel_size=25,
        dim_static=0,
        if_padding=1,
        # peak eval
        enable_peak_eval=1,
        eval_train_peaks=1,
        eval_val_peaks=1,
        eval_test_peaks=1,
        vis_train_peaks=0,
        vis_val_peaks=0,
        vis_test_peaks=1,
        peak_lookahead=3,
        peak_method="peakdetect",
        peak_workers=8,
        peak_tolerance=1,
        peak_threshold=0.4,
        use_soft_labels=1,
        soft_label_sigma=2.0,
        soft_label_tolerance=3,
        value_loss_weight=0.2,
        peak_loss_weight=0.4,
        tp_mse_loss_weight=0.4,
        focal_alpha=0.25,
        focal_gamma=2.0,
        seq2peak_value_loss_weight=0.5,
        seq2peak_peak_loss_weight=0.5,
        busy_decoder=0,
        busy_decoder_modal="m",
        hour_day="h",
        with_shift="ws",
        mlp_layers=2,
        if_msm_pl=1,
        if_lad=1,
        naive_type="persistence",
        movavg_window=24,
        seasonal_period=168,
        mask_type="soft",
        n_scales=2,
        use_external_features=0,
        external_feature_cols="",
        external_feature_mode="concat",
        num_external_features=0,
        endogenous_in=1,
        device=torch.device("cpu"),
    )
    return d


# Per-baseline overrides extracted from
# scripts/peak_detect_based_on_LTF/load_data/*.sh
BASELINE_OVERRIDES = {
    "PeakFocus (proposed_model)": dict(
        model="proposed_model",
        task_name="peak_detect_ltf",
        d_model=256,
        d_ff=256,
        e_layers=1,
        n_heads=4,
        factor=3,
        mlp_layers=2,
        if_lad=1,
        if_msm_pl=1,
        n_scales=2,
    ),
    "PatchTST": dict(
        model="PatchTST",
        task_name="peak_detect_ltf_basic",
        e_layers=1,
        d_layers=1,
        factor=3,
        n_heads=4,
        # no d_model / d_ff override in script -> use run.py defaults
    ),
    "SegRNN": dict(
        model="SegRNN",
        task_name="peak_detect_ltf_basic",
        seg_len=24,
        e_layers=1,
        d_layers=1,
        factor=3,
    ),
    "CycleNet": dict(
        model="CycleNet",
        task_name="peak_detect_ltf_basic",
        cycle=168,
        model_type="mlp",
        use_revin=1,
        d_model=1024,
    ),
    "Transformer": dict(
        model="Transformer",
        task_name="peak_detect_ltf_basic",
        e_layers=1,
        d_model=256,
        d_ff=512,
        n_heads=2,
    ),
    "Informer": dict(
        model="Informer",
        task_name="peak_detect_ltf_basic",
        e_layers=1,
        d_model=128,
        d_ff=256,
        factor=3,
        n_heads=8,
    ),
    "TimeMixer": dict(
        model="TimeMixer",
        task_name="peak_detect_ltf_basic",
        down_sampling_layers=3,
        down_sampling_method="avg",
        down_sampling_window=2,
    ),
    "STID": dict(
        model="STID",
        task_name="peak_detect_ltf",
        num_nodes=1,
        node_dim=32,
        embed_dim=512,
        num_layer=2,
        temp_dim_tid=32,
        temp_dim_diw=32,
        time_of_day_size=24,
        day_of_week_size=7,
        if_T_i_D=1,
        if_D_i_W=1,
        if_node=0,
    ),
}


class _LightExp(Exp_Basic):
    """Subclass that bypasses GPU and skips the optimizer/dataset plumbing."""

    def _acquire_device(self):
        return torch.device("cpu")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_len", type=int, default=336)
    opt = parser.parse_args()

    base = default_args()
    base["pred_len"] = opt.pred_len

    print("=" * 90)
    print(f"Parameter count (seq_len=168, pred_len={opt.pred_len}, univariate S features)")
    print("=" * 90)
    print(f"{'Model':<32} {'#Params':>14} {'Trainable':>14} {'Size (MB)':>12}")
    print("-" * 90)

    rows = []
    for name, overrides in BASELINE_OVERRIDES.items():
        cfg = dict(base)
        cfg.update(overrides)
        args = SimpleNamespace(**cfg)
        try:
            exp = _LightExp(args)
            total, trainable = count_params(exp.model)
            mb = total * 4 / 1024 / 1024  # float32
            rows.append((name, total, trainable, mb))
            print(f"{name:<32} {total:>14,} {trainable:>14,} {mb:>11.2f}M")
        except Exception as e:
            print(f"{name:<32} ERROR: {type(e).__name__}: {e}")

    print("=" * 90)
    return rows


if __name__ == "__main__":
    main()
