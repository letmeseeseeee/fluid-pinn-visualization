"""Generate Relative L2 Error figures for thesis Chapter 5."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / "web_exports"
FIGURES_DIR = ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# CJK-capable font setup for Windows
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# matplotlib style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

PRESET_SPECS = {
    "preset_base":     {"label": r"$\nu=1.0,\;\Delta t=1\times10^{-5}$ (基线)", "color": "#1d4ed8", "ls": "-"},
    "preset_low_nu":   {"label": r"$\nu=0.5,\;\Delta t=1\times10^{-5}$ (低扩散)", "color": "#0f766e", "ls": "--"},
    "preset_small_dt": {"label": r"$\nu=1.0,\;\Delta t=5\times10^{-6}$ (小步长)", "color": "#d97706", "ls": "-."},
    "preset_high_nu":  {"label": r"$\nu=1.5,\;\Delta t=1\times10^{-5}$ (高扩散)", "color": "#b91c1c", "ls": ":"},
}

MODEL_COMPARE = {
    "pinn": {"run": "epoch_108000", "label": "PINN (epoch 108000)", "color": "#1d4ed8", "ls": "-"},
    "fno":  {"run": "epoch_108000", "label": "FNO (epoch 108000)",  "color": "#b91c1c", "ls": "--"},
}


def load_frame_rel_l2(model: str, run: str) -> np.ndarray | None:
    metrics_path = EXPORT_ROOT / model / run / "metrics.json"
    if not metrics_path.exists():
        print(f"[warn] missing: {metrics_path}")
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return np.array(metrics["short"]["frame_rel_l2"])


def fig_preset_rel_l2():
    """Figure: per-frame Rel L2 for 4 PINN preset groups."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for preset_name, spec in PRESET_SPECS.items():
        arr = load_frame_rel_l2("pinn", preset_name)
        if arr is None:
            continue
        frames = np.arange(len(arr))
        ax.plot(frames, arr, color=spec["color"], linestyle=spec["ls"],
                linewidth=1.6, label=spec["label"])

    ax.set_xlabel("Frame index $t$")
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title("Per-frame Relative $L_2$ Error — PINN Preset Groups")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    path = FIGURES_DIR / "fig5_3_rel_l2_presets.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


def fig_pinn_vs_fno_rel_l2():
    """Figure: per-frame Rel L2 — PINN vs FNO at epoch_108000."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model, spec in MODEL_COMPARE.items():
        arr = load_frame_rel_l2(model, spec["run"])
        if arr is None:
            print(f"[warn] missing data for {model}/{spec['run']}")
            continue
        frames = np.arange(len(arr))
        ax.plot(frames, arr, color=spec["color"], linestyle=spec["ls"],
                linewidth=1.8, label=spec["label"])

    ax.set_xlabel("Frame index $t$")
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title("Per-frame Relative $L_2$ Error — PINN vs FNO")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    path = FIGURES_DIR / "fig5_6_rel_l2_pinn_fno.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


def fig_fno_epoch_rel_l2():
    """Figure: Rel L2 across FNO training epochs (3 checkpoints)."""
    fno_runs = ["epoch_012000", "epoch_060000", "epoch_108000"]
    epochs = [12000, 60000, 108000]
    rmse_vals = []
    rel_l2_vals = []

    for run in fno_runs:
        metrics_path = EXPORT_ROOT / "fno" / run / "metrics.json"
        if not metrics_path.exists():
            print(f"[warn] missing: {metrics_path}")
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        rmse_vals.append(m["short"]["rmse"])
        rel_l2_vals.append(m["short"]["rel_l2"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Training epoch")
    ax1.set_ylabel("RMSE", color="#1d4ed8")
    ax1.plot(epochs, rmse_vals, color="#1d4ed8", marker="o", linewidth=2.0,
             markersize=8, label="RMSE")
    ax1.tick_params(axis="y", labelcolor="#1d4ed8")
    ax1.set_xlim(0, 120000)
    ax1.set_ylim(0, max(rmse_vals) * 1.15)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Relative $L_2$ Error", color="#b91c1c")
    ax2.plot(epochs, rel_l2_vals, color="#b91c1c", marker="s", linewidth=2.0,
             markersize=8, linestyle="--", label="Relative $L_2$")
    ax2.tick_params(axis="y", labelcolor="#b91c1c")
    ax2.set_ylim(0, max(rel_l2_vals) * 1.15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.85)

    ax1.set_title("FNO — RMSE and Relative $L_2$ Error across Training Epochs")

    # Annotate values
    for i, (ep, rmse, rl2) in enumerate(zip(epochs, rmse_vals, rel_l2_vals)):
        ax1.annotate(f"{rmse:.2e}", (ep, rmse), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color="#1d4ed8")
        ax2.annotate(f"{rl2:.2e}", (ep, rl2), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color="#b91c1c")

    path = FIGURES_DIR / "fig5_4_fno_epoch_rel_l2.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


if __name__ == "__main__":
    fig_preset_rel_l2()
    fig_pinn_vs_fno_rel_l2()
    fig_fno_epoch_rel_l2()
    print("[done]")
