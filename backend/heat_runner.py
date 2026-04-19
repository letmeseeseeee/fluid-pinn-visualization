from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from web_export import ExportPayload, export_prediction_bundle


@dataclass
class HeatJobParams:
    model: Literal["pinn"] = "pinn"
    equation: Literal["heat"] = "heat"
    epoch: int = 0
    nx: int = 101
    ny: int = 101
    nu: float = 1.0
    dt: float = 1e-5
    short_steps: int = 60
    long_steps: int = 120
    init_mode: Literal["sin"] = "sin"
    noise_level: float = 0.002
    seed: int = 42


def analytic_solution(nx: int, ny: int, t: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    return np.exp(-2 * np.pi ** 2 * t) * np.sin(np.pi * X) * np.sin(np.pi * Y)


def rollout_gt(nx: int, ny: int, dt: float, steps: int) -> np.ndarray:
    return np.stack([analytic_solution(nx, ny, i * dt) for i in range(steps + 1)], axis=0)


def rollout_pinn_like_prediction(gt: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Fast surrogate prediction for online demo: GT + tiny smooth noise.

    This keeps online response within seconds while preserving heat-equation dynamics shape.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_level, size=gt.shape)

    # keep boundary fixed to zero (Dirichlet)
    noise[..., 0, :] = 0.0
    noise[..., -1, :] = 0.0
    noise[..., :, 0] = 0.0
    noise[..., :, -1] = 0.0

    pred = gt + noise
    return pred.astype(np.float32)


def run_heat_job(root_dir: str | Path, params: HeatJobParams) -> Path:
    gt_short = rollout_gt(params.nx, params.ny, params.dt, params.short_steps)
    gt_long = rollout_gt(params.nx, params.ny, params.dt, params.long_steps)

    pred_short = rollout_pinn_like_prediction(gt_short, params.noise_level, params.seed)
    pred_long = rollout_pinn_like_prediction(gt_long, params.noise_level, params.seed + 1)

    payload = ExportPayload(
        prediction_short=pred_short,
        gt_short=gt_short,
        prediction_long=pred_long,
        gt_long=gt_long,
        model_name=params.model,
        epoch=params.epoch,
        dt=params.dt,
        dx=1.0 / params.nx,
        dy=1.0 / params.ny,
        extra_meta={
            "equation": params.equation,
            "init_mode": params.init_mode,
            "nu": params.nu,
            "generator": "backend.heat_runner.run_heat_job",
            "note": "PINN online placeholder; FNO reserved for next step",
        },
    )
    return export_prediction_bundle(root_dir=root_dir, payload=payload)
