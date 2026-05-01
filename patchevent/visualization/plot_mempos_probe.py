from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # TODO(R-future): migrate to patchevent package import

from patchevent.phase2.dataset import SmallDecoderDataset
from patchevent.phase2.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", type=str, default="elc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_idx", type=int, default=901)
    p.add_argument("--config_a", type=str, default="G0_full")
    p.add_argument("--config_b", type=str, default="GS1_wo_mempos")
    p.add_argument(
        "--out",
        type=str,
        default=str(
            ROOT
            / "papers"
            / "69c8ebe9537191b1512a6125"
            / "figs"
            / "fig7_mempos_probe.png"
        ),
    )
    return p.parse_args()


def resolve_path(p: str) -> str:
    p = p.replace("\\", os.sep)
    p = os.path.normpath(p)
    if os.path.isabs(p):
        return p
    return str((ROOT / p).resolve())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _remap_legacy_path(p: str) -> str:
    """Remap legacy checkpoint paths from pre-refactor layout."""
    # phase1_encoder_pretrain_final/checkpoints/... → patchevent/phase1/checkpoints/...
    legacy = "phase1_encoder_pretrain_final/checkpoints/"
    new = "patchevent/phase1/checkpoints/"
    # handle both forward and back slashes
    for sep in ("/", "\\"):
        old_frag = legacy.replace("/", sep)
        new_frag = new.replace("/", sep)
        if old_frag in p:
            p = p.replace(old_frag, new_frag)
    return p


def load_model_from_exp(exp_dir: Path, device: torch.device):
    best_path = exp_dir / "best.pth"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    args_dict = dict(best["args"])
    args_dict["encoder_ckpt"] = resolve_path(_remap_legacy_path(args_dict["encoder_ckpt"]))
    args_dict["series_path"] = resolve_path(args_dict["series_path"])
    args_dict["events_path"] = resolve_path(args_dict["events_path"])
    args = argparse.Namespace(**args_dict)
    model = build_model(args).to(device)
    state = best.get("model", best)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, args


def load_sample(args: argparse.Namespace, sample_idx: int) -> dict:
    ds = SmallDecoderDataset(
        series_path=args.series_path,
        events_path=args.events_path,
        split="test",
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        window_stride=args.window_stride,
    )
    return ds[sample_idx]


def cosine_matrix(memory: torch.Tensor) -> np.ndarray:
    x = memory.detach().cpu().float().numpy()
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    return x @ x.T


def distance_curve(cos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = cos.shape[0]
    dists = np.arange(1, n)
    vals = []
    for d in dists:
        vals.append(float(np.mean(np.diag(cos, k=d))))
    return dists, np.array(vals, dtype=np.float32)


def rel_events(rec: dict) -> list[dict]:
    pred_start = int(rec["pred_start"])
    out = []
    for ev in rec["pred_events"]:
        out.append({
            "onset": ev["onset_idx"] - pred_start,
            "end": ev["end_idx"] - pred_start,
            "apex": ev["apex_idx"] - pred_start,
            "intensity": float(ev.get("apex_intensity", float("nan"))),
        })
    return out


def rel_gt_events(rec: dict) -> list[dict]:
    pred_start = int(rec["pred_start"])
    out = []
    for ev in rec["gt_events"]:
        out.append({
            "onset": ev["onset_idx"] - pred_start,
            "end": ev["end_idx"] - pred_start,
            "apex": ev["apex_idx"] - pred_start,
            "intensity": float(ev.get("apex_intensity", float("nan"))),
        })
    return out


def _event_intensity(ev: dict) -> float | None:
    v = float(ev.get("intensity", float("nan")))
    if not np.isfinite(v) or v <= 0:
        return None
    return v


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = ROOT / "patchevent" / "phase2" / "checkpoints" / f"{args.domain}_arch_ablation_v2"
    exp_a = base / f"{args.config_a}_s{args.seed}"
    exp_b = base / f"{args.config_b}_s{args.seed}"

    model_a, model_args = load_model_from_exp(exp_a, device)
    model_b, _ = load_model_from_exp(exp_b, device)
    sample = load_sample(model_args, args.sample_idx)

    x = sample["x"].unsqueeze(0).to(device)
    x_future = sample["x_future"].detach().cpu().numpy()

    with torch.no_grad():
        mem_a = model_a._encode_patches(x)[0, : model_a.n_patches, :]
        mem_b = model_b._encode_patches(x)[0, : model_b.n_patches, :]

    cos_a = cosine_matrix(mem_a)
    cos_b = cosine_matrix(mem_b)
    d_a, curve_a = distance_curve(cos_a)
    d_b, curve_b = distance_curve(cos_b)

    rows_a = load_jsonl(exp_a / "test_eval" / "eval_outputs.jsonl")
    rows_b = load_jsonl(exp_b / "test_eval" / "eval_outputs.jsonl")
    rec_a = rows_a[args.sample_idx]
    rec_b = rows_b[args.sample_idx]
    gt_events = rel_gt_events(rec_a)
    pred_a = rel_events(rec_a)
    pred_b = rel_events(rec_b)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Comic Sans MS",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.55)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_h1 = fig.add_subplot(gs[1, 0])
    ax_h2 = fig.add_subplot(gs[1, 1])
    ax_curve = fig.add_subplot(gs[1, 2])

    xs = np.arange(len(x_future))
    ax_ts.plot(xs, x_future, color="black", lw=1.5, label="Future series")
    all_ys = list(x_future)
    for ev in gt_events:
        v = _event_intensity(ev)
        if v is not None:
            all_ys.append(v)
    for ev in pred_a:
        v = _event_intensity(ev)
        if v is not None:
            all_ys.append(v)
    for ev in pred_b:
        v = _event_intensity(ev)
        if v is not None:
            all_ys.append(v)
    y_min = float(np.min(all_ys))
    y_max = float(np.max(all_ys))
    y_span = max(y_max - y_min, 1e-6)
    ax_ts.set_ylim(y_min - y_span * 0.08, y_max + y_span * 0.16)
    y0, y1 = ax_ts.get_ylim()
    for ev in gt_events:
        ax_ts.axvspan(ev["onset"], ev["end"], color="#AED6F1", alpha=0.30)
        gt_y = _event_intensity(ev)
        if gt_y is None:
            gt_y = x_future[max(0, min(len(x_future) - 1, ev["apex"]))]
        ax_ts.scatter(ev["apex"], gt_y, marker="v", color="#1565C0", s=35, zorder=4)
    for ev in pred_a:
        rect = mpatches.Rectangle((ev["onset"], y0), ev["end"] - ev["onset"], y1 - y0, fill=False, ec="#1f77b4", lw=1.0, linestyle="--")
        ax_ts.add_patch(rect)
        pred_y = _event_intensity(ev)
        if pred_y is not None:
            ax_ts.scatter(ev["apex"], pred_y, marker="^", color="#1f77b4", s=28, zorder=5)
    for ev in pred_b:
        rect = mpatches.Rectangle((ev["onset"], y0), ev["end"] - ev["onset"], y1 - y0, fill=False, ec="#ff7f0e", lw=1.0, linestyle="--")
        ax_ts.add_patch(rect)
        pred_y = _event_intensity(ev)
        if pred_y is not None:
            ax_ts.scatter(ev["apex"], pred_y, marker="o", facecolors="none", edgecolors="#ff7f0e", s=28, zorder=5)

    legend_handles = [
        plt.Line2D([0], [0], color="black", lw=1.5, label="Future series"),
        mpatches.Patch(facecolor="#AED6F1", edgecolor="#1565C0", alpha=0.35, label="GT span"),
        plt.Line2D([0], [0], color="#1f77b4", lw=1.0, linestyle="--", label="PatchEvent"),
        plt.Line2D([0], [0], color="#ff7f0e", lw=1.0, linestyle="--", label="w/o MemPos"),
    ]
    ax_ts.legend(handles=legend_handles, loc="upper left", ncol=4, frameon=False)
    ax_ts.set_xlim(0, len(x_future) - 1)
    ax_ts.set_title(f"(a) PatchEvent TP={rec_a['tp']} vs w/o MemPos TP={rec_b['tp']}", fontsize=9, fontweight="bold")
    ax_ts.set_xlabel("Hour")
    ax_ts.set_ylabel("Load")

    im1 = ax_h1.imshow(cos_a, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower", aspect="auto")
    ax_h1.set_title("(b) PatchEvent", fontsize=9, fontweight="bold")
    ax_h1.set_xlabel("Patch index")
    ax_h1.set_ylabel("Patch index")

    im2 = ax_h2.imshow(cos_b, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower", aspect="auto")
    ax_h2.set_title("(c) w/o MemPos", fontsize=9, fontweight="bold")
    ax_h2.set_xlabel("Patch index")
    ax_h2.set_ylabel("")

    cbar = fig.colorbar(im2, ax=[ax_h1, ax_h2], fraction=0.046, pad=0.06,
                         shrink=0.92)
    cbar.set_label("Cosine sim.", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    ax_curve.plot(d_a, curve_a, color="#1f77b4", lw=1.5, label="PatchEvent")
    ax_curve.plot(d_b, curve_b, color="#ff7f0e", lw=1.5, linestyle="--", label="w/o MemPos")
    ax_curve.set_title("(d) Mean cosine", fontsize=9, fontweight="bold")
    ax_curve.set_xlabel("Patch distance")
    ax_curve.set_ylabel("Cosine similarity", labelpad=2)
    ax_curve.legend(frameon=False)
    ax_curve.grid(alpha=0.30, linewidth=0.3, linestyle="--")

    offdiag_a = float(np.mean(cos_a[~np.eye(cos_a.shape[0], dtype=bool)]))
    offdiag_b = float(np.mean(cos_b[~np.eye(cos_b.shape[0], dtype=bool)]))
    fig.suptitle(
        "MemPos probe on saved checkpoints",
        y=0.99,
        fontsize=10,
    )

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.90)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(json.dumps({
        "out": str(out_path),
        "sample_idx": args.sample_idx,
        "pred_start": int(sample["pred_start"]),
        "tp_patch_event": rec_a["tp"],
        "tp_wo_mempos": rec_b["tp"],
        "offdiag_patch_event": offdiag_a,
        "offdiag_wo_mempos": offdiag_b,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
