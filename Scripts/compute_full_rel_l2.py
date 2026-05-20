"""Compute RMSE and Relative L2 Error across all PINN full training checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FULL_DIR = ROOT / "output" / "pinn_full"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters matching preset_base (the full training run)
NU = 1.0
DT = 1e-5
HW = 101
SHORT_STEPS = 100  # 101 frames (0..100)
DX = 1.0 / HW

# CJK font
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def compute_ground_truth(n_frames: int) -> np.ndarray:
    """Analytical solution: u(x,y,t) = exp(-2*nu*pi^2*t) * sin(pi*x) * sin(pi*y)"""
    xx = np.linspace(0, 1, HW)
    yy = np.linspace(0, 1, HW)
    X, Y = np.meshgrid(xx, yy)
    result = np.zeros((n_frames, HW, HW), dtype=np.float64)
    for i in range(n_frames):
        t = i * DT
        result[i] = np.exp(-2 * NU * np.pi ** 2 * t) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    return result


def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Global RMSE."""
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    return float(np.sqrt(mse))


def relative_l2(pred: np.ndarray, gt: np.ndarray) -> float:
    """Global Relative L2 Error."""
    diff = pred - gt
    diff_norm = float(np.sqrt(np.sum(diff ** 2)))
    gt_norm = float(np.sqrt(np.sum(gt ** 2)))
    return diff_norm / gt_norm if gt_norm > 0 else 0.0


def main():
    # Compute ground truth once
    print("Computing ground truth (101 frames)...")
    gt = compute_ground_truth(SHORT_STEPS + 1)  # 101 frames

    # Collect result files
    result_files = sorted(FULL_DIR.glob("result_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(result_files)} result files")

    epochs = []
    rmse_values = []
    rel_l2_values = []
    missing = 0

    for rf in result_files:
        try:
            epoch_num = int(rf.stem.split("_")[1])
        except ValueError:
            continue

        try:
            pred = np.load(rf)  # shape: (101, 1, 101, 101)
        except Exception:
            missing += 1
            continue

        # Remove channel dim: (T, H, W)
        pred = pred[:, 0, :, :]

        # Ensure same frame count
        n_frames = min(pred.shape[0], gt.shape[0])
        rmse = compute_rmse(pred[:n_frames], gt[:n_frames])
        rl2 = relative_l2(pred[:n_frames], gt[:n_frames])

        epochs.append(epoch_num)
        rmse_values.append(rmse)
        rel_l2_values.append(rl2)

    if missing:
        print(f"[warn] {missing} files failed to load")

    # Save as JSON
    data = {
        "nu": NU, "dt": DT, "nx": HW, "ny": HW, "short_steps": SHORT_STEPS,
        "epochs": epochs,
        "rmse": rmse_values,
        "rel_l2": rel_l2_values,
    }
    json_path = ROOT / "output" / "pinn_full_metrics.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {json_path}")

    # Key checkpoints
    key_epochs = [0, 12000, 108000, 477000, 500000, 1000000]
    print("\nKey checkpoints:")
    print(f"{'Epoch':>10}  {'RMSE':>12}  {'Rel L2':>12}")
    print("-" * 38)
    for ke in key_epochs:
        if ke in epochs:
            idx = epochs.index(ke)
            print(f"{ke:>10}  {rmse_values[idx]:>12.4e}  {rel_l2_values[idx]:>12.4e}")

    # Find best for each metric
    rmse_min_idx = int(np.argmin(rmse_values))
    rl2_min_idx = int(np.argmin(rel_l2_values))
    print(f"\nBest RMSE: epoch_{epochs[rmse_min_idx]}  {rmse_values[rmse_min_idx]:.4e}")
    print(f"Best Rel L2: epoch_{epochs[rl2_min_idx]}  {rel_l2_values[rl2_min_idx]:.4e}")

    # ---- Plot: RMSE curve ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, rmse_values, color="#1d4ed8", linewidth=0.6, alpha=0.85)
    ax.scatter(epochs[rmse_min_idx], rmse_values[rmse_min_idx],
               color="#b91c1c", s=40, zorder=5, label=f"Best: epoch {epochs[rmse_min_idx]} ({rmse_values[rmse_min_idx]:.2e})")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("PINN Full Training — RMSE Evolution")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_xlim(0, max(epochs))

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax, width="38%", height="30%", loc="center right",
                          bbox_to_anchor=(-0.06, -0.42, 1, 1),
                          bbox_transform=ax.transAxes)
    mask = np.array(epochs) >= 800000
    ax_inset.plot(np.array(epochs)[mask], np.array(rmse_values)[mask],
                  color="#1d4ed8", linewidth=1.0)
    ax_inset.scatter(epochs[rmse_min_idx], rmse_values[rmse_min_idx],
                     color="#b91c1c", s=25, zorder=5)
    ax_inset.set_title("Late training (800k–1M)", fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(True, alpha=0.3, linestyle="--")

    path = OUT_DIR / "fig5_5_pinn_full_epoch_rmse.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")

    # ---- Plot: Relative L2 curve ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, rel_l2_values, color="#0f766e", linewidth=0.6, alpha=0.85)
    ax.scatter(epochs[rl2_min_idx], rel_l2_values[rl2_min_idx],
               color="#b91c1c", s=40, zorder=5, label=f"Best: epoch {epochs[rl2_min_idx]} ({rel_l2_values[rl2_min_idx]:.2e})")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Relative $L_2$ Error")
    ax.set_title("PINN Full Training — Relative $L_2$ Error Evolution")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_xlim(0, max(epochs))

    ax_inset2 = inset_axes(ax, width="38%", height="30%", loc="center right",
                           bbox_to_anchor=(-0.06, -0.42, 1, 1),
                           bbox_transform=ax.transAxes)
    mask = np.array(epochs) >= 800000
    ax_inset2.plot(np.array(epochs)[mask], np.array(rel_l2_values)[mask],
                   color="#0f766e", linewidth=1.0)
    ax_inset2.scatter(epochs[rl2_min_idx], rel_l2_values[rl2_min_idx],
                      color="#b91c1c", s=25, zorder=5)
    ax_inset2.set_title("Late training (800k–1M)", fontsize=8)
    ax_inset2.tick_params(labelsize=7)
    ax_inset2.grid(True, alpha=0.3, linestyle="--")

    path = OUT_DIR / "fig5_5_pinn_full_rel_l2.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
