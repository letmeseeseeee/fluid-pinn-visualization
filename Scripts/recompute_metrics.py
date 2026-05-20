"""Recompute metrics.json for all existing web_export results with Relative L2 Error."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / "web_exports"


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    diff = pred - gt
    per_t_mse = np.mean(diff ** 2, axis=(1, 2))
    per_t_rmse = np.sqrt(per_t_mse)

    gt_l2 = np.sqrt(np.sum(gt ** 2, axis=(1, 2)))
    diff_l2 = np.sqrt(np.sum(diff ** 2, axis=(1, 2)))
    per_t_rel_l2 = np.where(gt_l2 > 0, diff_l2 / gt_l2, 0.0)

    gt_l2_global = float(np.sqrt(np.sum(gt ** 2)))
    diff_l2_global = float(np.sqrt(np.sum(diff ** 2)))
    rel_l2_global = diff_l2_global / gt_l2_global if gt_l2_global > 0 else 0.0

    return {
        "mse": float(np.mean(diff ** 2)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "rel_l2": rel_l2_global,
        "frame_mse": per_t_mse.tolist(),
        "frame_rmse": per_t_rmse.tolist(),
        "frame_rel_l2": per_t_rel_l2.tolist(),
    }


def process_run(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.json"
    short_pred_path = run_dir / "prediction_short.npy"
    short_gt_path = run_dir / "gt_short.npy"
    long_pred_path = run_dir / "prediction_long.npy"
    long_gt_path = run_dir / "gt_long.npy"

    if not short_pred_path.exists() or not short_gt_path.exists():
        return False

    short_pred = np.load(short_pred_path)
    short_gt = np.load(short_gt_path)
    short_metrics = compute_metrics(short_pred, short_gt)

    if long_pred_path.exists() and long_gt_path.exists():
        long_pred = np.load(long_pred_path)
        long_gt = np.load(long_gt_path)
        long_metrics = compute_metrics(long_pred, long_gt)
    else:
        long_metrics = {}

    metrics = {"short": short_metrics, "long": long_metrics}
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main():
    if not EXPORT_ROOT.exists():
        print("[skip] web_exports/ not found")
        return

    success = 0
    skip = 0
    for model_dir in sorted(EXPORT_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if process_run(run_dir):
                success += 1
            else:
                skip += 1

    print(f"[done] Updated {success} runs, skipped {skip} (missing data)")


if __name__ == "__main__":
    main()
