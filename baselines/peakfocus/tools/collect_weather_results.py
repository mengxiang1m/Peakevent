import numpy as np
import os

base = "results"
for pred_len in [336, 720]:
    print(f"\n=== pred_len={pred_len}, 3 runs ===")
    mse_arr, mae_arr, f1_arr, tp_arr, bpe_arr, pim_arr = [], [], [], [], [], []
    for i in range(3):
        name = f"peak_detect_ltf_load_data_mixed_proposed_model_maxIn_maxOut_MSE_244_23_weather_168_{pred_len}_{i}"
        mpath = os.path.join(base, name, "metrics.npy")
        ppath = os.path.join(base, name, "peak_metrics.npy")
        if not os.path.exists(mpath):
            print(f"  Run {i}: NOT FOUND")
            continue
        m = np.load(mpath)
        p = np.load(ppath, allow_pickle=True).item()
        mse, mae = float(m[0]), float(m[1])
        f1 = float(p["Peak_Cls_F1"])
        tp_mse = float(p["Peak_Cls_TP_MSE"])
        bpe = float(p["Peak_Cls_Balanced_Error"])
        pim = float(p["Peak_Cls_PIM"])
        print(f"  Run {i}: MSE={mse:.4f}  MAE={mae:.4f}  F1={f1:.4f}  TP_MSE={tp_mse:.4f}  BPE={bpe:.4f}  PIM={pim:.4f}")
        mse_arr.append(mse)
        mae_arr.append(mae)
        f1_arr.append(f1)
        tp_arr.append(tp_mse)
        bpe_arr.append(bpe)
        pim_arr.append(pim)
    if len(mse_arr) > 0:
        print(f"  Mean+-Std:")
        print(f"    MSE:    {np.mean(mse_arr):.4f} +- {np.std(mse_arr):.4f}")
        print(f"    MAE:    {np.mean(mae_arr):.4f} +- {np.std(mae_arr):.4f}")
        print(f"    F1:     {np.mean(f1_arr):.4f} +- {np.std(f1_arr):.4f}")
        print(f"    TP_MSE: {np.mean(tp_arr):.4f} +- {np.std(tp_arr):.4f}")
        print(f"    BPE:    {np.mean(bpe_arr):.4f} +- {np.std(bpe_arr):.4f}")
        print(f"    PIM:    {np.mean(pim_arr):.4f} +- {np.std(pim_arr):.4f}")
