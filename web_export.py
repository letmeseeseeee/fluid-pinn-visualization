from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ExportPayload:
    prediction_short: np.ndarray
    gt_short: np.ndarray
    prediction_long: np.ndarray
    gt_long: np.ndarray
    model_name: str
    epoch: int
    dt: float
    dx: float
    dy: float
    run_name: Optional[str] = None
    extra_meta: Optional[Dict[str, Any]] = None


def _to_3d(arr: np.ndarray) -> np.ndarray:
    """Convert to [T, H, W] if input is [T, 1, H, W]."""
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected [T,H,W] or [T,1,H,W], got shape={arr.shape}")


def _series_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
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


def export_prediction_bundle(root_dir: str | Path, payload: ExportPayload) -> Path:
    """Export standardized files for web visualization.

    Output structure:
    web_exports/{model}/epoch_xxxxxx/
      prediction_short.npy
      gt_short.npy
      diff_short.npy
      prediction_long.npy
      gt_long.npy
      diff_long.npy
      meta.json
      metrics.json
    """
    root = Path(root_dir)
    run_name = payload.run_name or f"epoch_{payload.epoch:06d}"
    out_dir = root / "web_exports" / payload.model_name.lower() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_short = _to_3d(payload.prediction_short)
    gt_short = _to_3d(payload.gt_short)
    pred_long = _to_3d(payload.prediction_long)
    gt_long = _to_3d(payload.gt_long)

    if pred_short.shape != gt_short.shape:
        raise ValueError(f"short shape mismatch: pred={pred_short.shape}, gt={gt_short.shape}")
    if pred_long.shape != gt_long.shape:
        raise ValueError(f"long shape mismatch: pred={pred_long.shape}, gt={gt_long.shape}")

    diff_short = pred_short - gt_short
    diff_long = pred_long - gt_long

    np.save(out_dir / "prediction_short.npy", pred_short)
    np.save(out_dir / "gt_short.npy", gt_short)
    np.save(out_dir / "diff_short.npy", diff_short)
    np.save(out_dir / "prediction_long.npy", pred_long)
    np.save(out_dir / "gt_long.npy", gt_long)
    np.save(out_dir / "diff_long.npy", diff_long)

    meta: Dict[str, Any] = {
        "model": payload.model_name.lower(),
        "epoch": payload.epoch,
        "grid": {
            "height": int(pred_short.shape[1]),
            "width": int(pred_short.shape[2]),
            "dt": float(payload.dt),
            "dx": float(payload.dx),
            "dy": float(payload.dy),
        },
        "frames": {
            "short": int(pred_short.shape[0]),
            "long": int(pred_long.shape[0]),
        },
        "files": {
            "prediction_short": "prediction_short.npy",
            "gt_short": "gt_short.npy",
            "diff_short": "diff_short.npy",
            "prediction_long": "prediction_long.npy",
            "gt_long": "gt_long.npy",
            "diff_long": "diff_long.npy",
            "metrics": "metrics.json",
            "meta": "meta.json",
        },
    }

    if payload.extra_meta:
        meta["extra"] = payload.extra_meta

    metrics = {
        "short": _series_metrics(pred_short, gt_short),
        "long": _series_metrics(pred_long, gt_long),
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_dir
