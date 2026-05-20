from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str) -> None:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.15",
            linewidth=1.6,
            edgecolor="#24476b",
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=13)

    box(0.7, 5.0, 3.0, 1.4, "Cloud Training\nPINN / FNO results", "#eef6ff")
    box(0.7, 1.7, 3.0, 1.4, "Online Solve\nOriginal Hea0.1.19.py", "#fff2e9")
    box(5.0, 3.4, 4.0, 1.6, "Unified Result Layer\nweb_exports / metrics / meta", "#eefbf0")
    box(10.1, 5.0, 3.0, 1.4, "FastAPI Service\nmodels / fields / jobs", "#f5f0ff")
    box(10.1, 1.7, 3.0, 1.4, "Vue + ECharts\nheatmaps / curves / forms", "#fff9e8")

    ax.annotate("", xy=(5.0, 4.35), xytext=(3.7, 5.7), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(5.0, 4.15), xytext=(3.7, 2.4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(10.1, 5.7), xytext=(9.0, 4.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(10.1, 2.4), xytext=(9.0, 4.0), arrowprops=dict(arrowstyle="->", lw=2))

    ax.text(7.0, 6.9, "Dual-path architecture for training results and online solve", ha="center", fontsize=16, weight="bold")
    ax.text(7.0, 0.8, "Both historical runs and new runtime runs are normalized into the same data format.", ha="center", fontsize=11, color="#4c5a67")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_1_system_architecture.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_runtime_workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    steps = [
        (0.7, 3.0, 2.2, 1.1, "User input\nparameters"),
        (3.3, 3.0, 2.3, 1.1, "POST /api/jobs"),
        (6.0, 3.0, 2.3, 1.1, "Cache check"),
        (8.7, 4.5, 2.6, 1.1, "Cache hit\nreturn run"),
        (8.7, 1.5, 2.6, 1.1, "Cache miss\nlaunch solver"),
        (11.6, 1.5, 2.5, 1.1, "Export to\nweb_exports"),
        (11.6, 4.5, 2.5, 1.1, "Frontend refresh\nshow result"),
    ]

    for x, y, w, h, label in steps:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.4,
            edgecolor="#314e6e",
            facecolor="#f9fbfd",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=12)

    arrow = dict(arrowstyle="->", lw=1.8, color="#314e6e")
    ax.annotate("", xy=(3.3, 3.55), xytext=(2.9, 3.55), arrowprops=arrow)
    ax.annotate("", xy=(6.0, 3.55), xytext=(5.6, 3.55), arrowprops=arrow)
    ax.annotate("", xy=(8.7, 5.05), xytext=(8.3, 3.85), arrowprops=arrow)
    ax.annotate("", xy=(8.7, 2.05), xytext=(8.3, 3.25), arrowprops=arrow)
    ax.annotate("", xy=(11.6, 2.05), xytext=(11.3, 2.05), arrowprops=arrow)
    ax.annotate("", xy=(12.85, 4.5), xytext=(12.85, 2.6), arrowprops=arrow)
    ax.annotate("", xy=(11.6, 5.05), xytext=(11.3, 5.05), arrowprops=arrow)

    ax.text(7.5, 6.3, "Runtime solve workflow with cache reuse", ha="center", fontsize=16, weight="bold")
    ax.text(7.5, 0.7, "The online module reuses historical runs whenever the same parameter signature already exists.", ha="center", fontsize=11, color="#4c5a67")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_1_runtime_workflow.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_figure() -> None:
    pinn = np.load(ROOT / "web_exports" / "pinn" / "epoch_108000" / "prediction_short.npy")
    fno = np.load(ROOT / "web_exports" / "fno" / "epoch_108000" / "prediction_short.npy")
    gt = np.load(ROOT / "web_exports" / "pinn" / "epoch_108000" / "gt_short.npy")

    frame_idx = 100
    pinn_frame = pinn[frame_idx]
    fno_frame = fno[frame_idx]
    gt_frame = gt[frame_idx]
    diff_frame = pinn_frame - fno_frame

    vmin = min(float(pinn_frame.min()), float(fno_frame.min()), float(gt_frame.min()))
    vmax = max(float(pinn_frame.max()), float(fno_frame.max()), float(gt_frame.max()))
    diff_lim = float(np.max(np.abs(diff_frame)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Heat field comparison at short rollout frame 100", fontsize=15, weight="bold")

    im0 = axes[0, 0].imshow(pinn_frame, cmap="YlOrRd", origin="lower", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("PINN prediction")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    im1 = axes[0, 1].imshow(fno_frame, cmap="Greens", origin="lower", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("FNO prediction")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    im2 = axes[1, 0].imshow(gt_frame, cmap="YlOrRd", origin="lower", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Analytical ground truth")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    im3 = axes[1, 1].imshow(diff_frame, cmap="RdBu_r", origin="lower", vmin=-diff_lim, vmax=diff_lim)
    axes[1, 1].set_title("PINN - FNO")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    cbar0 = fig.colorbar(im2, ax=[axes[0, 0], axes[0, 1], axes[1, 0]], fraction=0.03, pad=0.02)
    cbar0.set_label("Temperature")
    cbar1 = fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar1.set_label("Difference")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_1_heatmap_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metric_curve_figure() -> None:
    pinn_metrics = load_json(ROOT / "web_exports" / "pinn" / "epoch_108000" / "metrics.json")
    fno_metrics = load_json(ROOT / "web_exports" / "fno" / "epoch_108000" / "metrics.json")

    x = np.arange(len(pinn_metrics["short"]["frame_rmse"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("Frame-wise error curves on the short rollout", fontsize=15, weight="bold")

    axes[0].plot(x, pinn_metrics["short"]["frame_rmse"], label="PINN", color="#db6d00", linewidth=2)
    axes[0].plot(x, fno_metrics["short"]["frame_rmse"], label="FNO", color="#1b7f5a", linewidth=2)
    axes[0].set_title("Frame RMSE")
    axes[0].set_xlabel("Frame index")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, pinn_metrics["short"]["frame_mse"], label="PINN", color="#db6d00", linewidth=2)
    axes[1].plot(x, fno_metrics["short"]["frame_mse"], label="FNO", color="#1b7f5a", linewidth=2)
    axes[1].set_title("Frame MSE")
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_2_error_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_architecture_figure()
    save_runtime_workflow_figure()
    save_heatmap_figure()
    save_metric_curve_figure()


if __name__ == "__main__":
    main()
