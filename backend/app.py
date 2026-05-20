from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.pinn_original_runner import (
    PinnOriginalJobParams,
    build_run_name,
    cancel_pinn_job,
    find_cached_pinn_run,
    run_original_pinn_job,
)

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / "web_exports"
PINN_PRESET_PREFIX = "preset_"

PINN_PRESET_CATALOG = {
    "preset_base": {
        "title": "基线预设",
        "note": "nu=1.0, dt=1e-5, epochs=12000, lr=1e-5",
        "sort": 0,
    },
    "preset_low_nu": {
        "title": "低扩散预设",
        "note": "nu=0.5, dt=1e-5, epochs=12000, lr=1e-5",
        "sort": 1,
    },
    "preset_small_dt": {
        "title": "小步长预设",
        "note": "nu=1.0, dt=5e-6, epochs=12000, lr=1e-5",
        "sort": 2,
    },
    "preset_high_nu": {
        "title": "高扩散预设",
        "note": "nu=1.5, dt=1e-5, epochs=12000, lr=1e-5",
        "sort": 3,
    },
}

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class SolveRequest(BaseModel):
    equation: str = "heat"
    nx: int = 101
    ny: int = 101
    nu: float = 1.0
    dt: float = 1e-5
    short_steps: int = 100
    long_steps: int = 100
    epochs: int = 500000
    learning_rate: float = 1e-5
    seed: int = 50976
    network_type: str = "transformer"
    transformer_hidden_channels: int = 128
    patch_size: int = 16
    num_heads: int = 4
    num_layers: int = 4
    loss_phy_weight: float = 1.0
    loss_data_weight: float = 0.0


def _params_from_request(body: SolveRequest) -> PinnOriginalJobParams:
    return PinnOriginalJobParams(
        equation=body.equation,
        nx=body.nx,
        ny=body.ny,
        nu=body.nu,
        dt=body.dt,
        short_steps=body.short_steps,
        long_steps=body.long_steps,
        epochs=body.epochs,
        learning_rate=body.learning_rate,
        seed=body.seed,
        network_type=body.network_type,
        transformer_hidden_channels=body.transformer_hidden_channels,
        patch_size=body.patch_size,
        num_heads=body.num_heads,
        num_layers=body.num_layers,
        loss_phy_weight=body.loss_phy_weight,
        loss_data_weight=body.loss_data_weight,
        use_cache=True,
    )


def _run_job_background(job_id: str, params: PinnOriginalJobParams) -> None:
    try:
        out_dir, cached = run_original_pinn_job(str(ROOT), params)
        with _jobs_lock:
            _jobs[job_id] = {
                "job_id": job_id,
                "status": "completed",
                "cached": cached,
                "run_name": out_dir.name,
                "total_epochs": params.epochs,
                "completed_at": time.time(),
            }
    except Exception as exc:
        with _jobs_lock:
            existing = _jobs.get(job_id)
            if existing and existing.get("status") == "cancelled":
                return
            _jobs[job_id] = {
                "job_id": job_id,
                "status": "failed",
                "error": str(exc),
                "total_epochs": params.epochs,
                "completed_at": time.time(),
            }


def _read_job_progress(job_id: str, total_epochs: int) -> dict | None:
    log_path = ROOT / "output" / "pinn_jobs" / f"{job_id}.log"
    if not log_path.exists():
        return None
    try:
        with log_path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            tail = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    matches = re.findall(r"It:\s*(\d+)\s+Loss:\s*([\d.e+\-]+)", tail)
    if not matches:
        return None

    current_iter = int(matches[-1][0])
    current_loss = float(matches[-1][1])
    return {
        "iteration": current_iter,
        "total_epochs": total_epochs,
        "loss": current_loss,
        "pct": round(current_iter / max(total_epochs, 1) * 100, 1),
    }


app = FastAPI(title="Fluid Simulation Visualization API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model_dir(model: str) -> Path:
    directory = EXPORT_ROOT / model.lower()
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
    return directory


def _run_dir(model: str, epoch: str) -> Path:
    directory = _model_dir(model) / epoch
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Run '{model}/{epoch}' not found")
    return directory


def _load_meta(run_dir: Path) -> dict | None:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/models")
def get_models():
    if not EXPORT_ROOT.exists():
        return {"models": []}
    models = sorted([path.name for path in EXPORT_ROOT.iterdir() if path.is_dir()])
    return {"models": models}


@app.get("/api/models/{model}/runs")
def get_runs(model: str):
    model_dir = _model_dir(model)
    runs = sorted([path.name for path in model_dir.iterdir() if path.is_dir()])
    return {"model": model, "runs": runs}


@app.get("/api/pinn/presets")
def get_pinn_presets():
    model_dir = _model_dir("pinn")
    presets = []
    for run_dir in sorted([path for path in model_dir.iterdir() if path.is_dir()]):
        if not run_dir.name.startswith(PINN_PRESET_PREFIX):
            continue
        if run_dir.name not in PINN_PRESET_CATALOG:
            continue
        meta = _load_meta(run_dir)
        if not meta:
            continue
        runtime = ((meta.get("extra") or {}).get("runtime_config")) or {}
        if not runtime:
            continue
        catalog = PINN_PRESET_CATALOG[run_dir.name]
        presets.append(
            {
                "title": catalog["title"],
                "note": catalog["note"],
                "run": run_dir.name,
                "sort": int(catalog["sort"]),
                "params": {
                    "equation": str(runtime.get("equation", "heat")),
                    "nx": int(runtime.get("nx", 101)),
                    "ny": int(runtime.get("ny", 101)),
                    "nu": float(runtime.get("nu", 1.0)),
                    "dt": float(runtime.get("dt", 1e-5)),
                    "short_steps": int(runtime.get("short_steps", 100)),
                    "long_steps": int(runtime.get("long_steps", 100)),
                    "epochs": int(runtime.get("epochs", 12000)),
                    "learning_rate": float(runtime.get("learning_rate", 1e-5)),
                    "seed": int(runtime.get("seed", 50976)),
                    "network_type": str(runtime.get("network_type", "transformer")),
                    "transformer_hidden_channels": int(runtime.get("transformer_hidden_channels", 128)),
                    "patch_size": int(runtime.get("patch_size", 16)),
                    "num_heads": int(runtime.get("num_heads", 4)),
                    "num_layers": int(runtime.get("num_layers", 4)),
                    "loss_phy_weight": float(runtime.get("loss_phy_weight", 1.0)),
                    "loss_data_weight": float(runtime.get("loss_data_weight", 0.0)),
                },
            }
        )
    presets.sort(key=lambda item: (item["sort"], item["run"]))
    return {"presets": presets}


@app.get("/api/models/{model}/{epoch}/meta")
def get_meta(model: str, epoch: str):
    meta_path = _run_dir(model, epoch) / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="meta.json not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@app.get("/api/models/{model}/{epoch}/metrics")
def get_metrics(model: str, epoch: str):
    metrics_path = _run_dir(model, epoch) / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.get("/api/models/{model}/{epoch}/field")
def get_field(
    model: str,
    epoch: str,
    kind: Literal[
        "prediction_short",
        "gt_short",
        "diff_short",
        "prediction_long",
        "gt_long",
        "diff_long",
    ] = Query("prediction_short"),
    t: int = Query(0, ge=0),
):
    field_path = _run_dir(model, epoch) / f"{kind}.npy"
    if not field_path.exists():
        raise HTTPException(status_code=404, detail=f"{kind}.npy not found")
    array = np.load(field_path)
    if t >= array.shape[0]:
        raise HTTPException(status_code=400, detail=f"t out of range, max={array.shape[0] - 1}")
    return {
        "model": model,
        "epoch": epoch,
        "kind": kind,
        "t": t,
        "shape": list(array.shape),
        "field": array[t].tolist(),
    }


@app.get("/api/compare/{epoch}")
def compare_models(
    epoch: str,
    left: str = Query("pinn"),
    right: str = Query("fno"),
    t: int = Query(0, ge=0),
    split: Literal["short", "long"] = Query("short"),
):
    left_path = _run_dir(left, epoch) / f"prediction_{split}.npy"
    right_path = _run_dir(right, epoch) / f"prediction_{split}.npy"
    if not left_path.exists() or not right_path.exists():
        return {
            "epoch": epoch,
            "left": left,
            "right": right,
            "split": split,
            "t": t,
            "available": False,
            "message": "Comparison result unavailable: one model export is missing",
        }

    left_array = np.load(left_path)
    right_array = np.load(right_path)
    if left_array.shape != right_array.shape:
        raise HTTPException(status_code=400, detail="shape mismatch between compared arrays")
    if t >= left_array.shape[0]:
        raise HTTPException(status_code=400, detail=f"t out of range, max={left_array.shape[0] - 1}")

    diff = left_array[t] - right_array[t]
    return {
        "epoch": epoch,
        "left": left,
        "right": right,
        "split": split,
        "t": t,
        "available": True,
        "shape": list(left_array.shape),
        "left_field": left_array[t].tolist(),
        "right_field": right_array[t].tolist(),
        "diff_field": diff.tolist(),
    }


@app.post("/api/pinn/solve")
def solve_pinn(body: SolveRequest):
    params = _params_from_request(body)
    job_id = build_run_name(params)

    with _jobs_lock:
        existing = _jobs.get(job_id)
        if existing and existing["status"] == "running":
            return existing

    cached = find_cached_pinn_run(str(ROOT), params)
    if cached:
        return {
            "job_id": job_id,
            "status": "completed",
            "cached": True,
            "run_name": cached.name,
        }

    with _jobs_lock:
        if _jobs.get(job_id, {}).get("status") == "running":
            return _jobs[job_id]
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "cached": False,
            "total_epochs": params.epochs,
        }

    thread = threading.Thread(
        target=_run_job_background,
        args=(job_id, params),
        daemon=True,
    )
    thread.start()

    return _jobs[job_id]


@app.get("/api/pinn/jobs/{job_id}")
def get_job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] == "running":
        total = job.get("total_epochs", 0)
        progress = _read_job_progress(job_id, total)
        return {**job, "progress": progress}

    return job


@app.delete("/api/pinn/jobs/{job_id}")
def cancel_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job["status"] != "running":
        raise HTTPException(status_code=400, detail="Job is not running")

    killed = cancel_pinn_job(job_id)
    with _jobs_lock:
        _jobs[job_id] = {
            **job,
            "status": "cancelled",
            "completed_at": time.time(),
        }

    return {
        "job_id": job_id,
        "status": "cancelled",
        "process_killed": killed,
    }
