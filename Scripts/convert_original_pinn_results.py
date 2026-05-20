from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from web_export import ExportPayload, export_prediction_bundle


def analytic_heat_gt(frames: int, grid_size: int, dt: float, nu: float = 1.0) -> np.ndarray:
    xx = np.linspace(0.0, 1.0, grid_size)
    yy = np.linspace(0.0, 1.0, grid_size)
    x_mesh, y_mesh = np.meshgrid(xx, yy)

    series = []
    for i in range(frames):
        t = i * dt
        frame = np.exp(-2 * np.pi ** 2 * nu * t) * np.sin(np.pi * x_mesh) * np.sin(np.pi * y_mesh)
        series.append(frame.astype(np.float32))
    return np.stack(series, axis=0)


def epoch_from_name(path: Path) -> int | None:
    match = re.search(r"result_(\d+)\.npy$", path.name)
    if not match:
        return None
    return int(match.group(1))


def convert_result_file(result_path: Path, root_dir: Path, dt: float, nu: float, model_name: str) -> Path:
    prediction = np.load(result_path).astype(np.float32)
    if prediction.ndim == 4 and prediction.shape[1] == 1:
        prediction = prediction[:, 0, ...]
    if prediction.ndim != 3:
        raise ValueError(f"Expected [T,H,W] or [T,1,H,W] array, got shape={prediction.shape} for {result_path}")

    frames, height, width = prediction.shape
    if height != width:
        raise ValueError(f"Expected square grid, got shape={prediction.shape} for {result_path}")

    gt = analytic_heat_gt(frames=frames, grid_size=height, dt=dt, nu=nu)
    epoch = epoch_from_name(result_path)
    if epoch is None:
        raise ValueError(f"Cannot parse epoch from filename: {result_path.name}")

    payload = ExportPayload(
        prediction_short=prediction,
        gt_short=gt,
        prediction_long=prediction,
        gt_long=gt,
        model_name=model_name,
        epoch=epoch,
        dt=dt,
        dx=1.0 / height,
        dy=1.0 / width,
        extra_meta={
            "source": "original_pinn_result",
            "source_file": result_path.name,
            "nu": nu,
        },
    )
    return export_prediction_bundle(root_dir, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert original Hea0.1.19.py result_*.npy files into web_exports format.")
    parser.add_argument("--result-dir", type=Path, required=True, help="Directory containing result_*.npy from original training")
    parser.add_argument("--root-dir", type=Path, required=True, help="Project root where web_exports will be written")
    parser.add_argument("--dt", type=float, default=1e-5, help="Time step used by the original test/export")
    parser.add_argument("--nu", type=float, default=1.0, help="Diffusion coefficient used for analytic ground truth")
    parser.add_argument("--model-name", type=str, default="pinn", help="Model folder name under web_exports")
    parser.add_argument("--limit", type=int, default=0, help="Optional positive limit on number of result files to convert")
    parser.add_argument("--clean-target", action="store_true", help="Delete existing converted epoch_* directories before conversion")
    args = parser.parse_args()

    result_files = sorted(
        [path for path in args.result_dir.glob("result_*.npy") if epoch_from_name(path) is not None],
        key=lambda path: epoch_from_name(path) or 0,
    )
    if args.limit > 0:
        result_files = result_files[: args.limit]

    target_root = args.root_dir / "web_exports" / args.model_name.lower()
    target_root.mkdir(parents=True, exist_ok=True)

    if args.clean_target:
        for epoch_dir in target_root.glob("epoch_*"):
            if epoch_dir.is_dir():
                for item in epoch_dir.iterdir():
                    item.unlink()
                epoch_dir.rmdir()

    print(f"Converting {len(result_files)} result files from {args.result_dir}")
    for index, result_file in enumerate(result_files, start=1):
        out_dir = convert_result_file(result_file, args.root_dir, args.dt, args.nu, args.model_name)
        print(f"[{index}/{len(result_files)}] {result_file.name} -> {out_dir}")


if __name__ == "__main__":
    main()
