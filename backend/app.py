from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Dict, Literal

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.heat_runner import HeatJobParams, run_heat_job

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / "web_exports"

app = FastAPI(title="Fluid PINN/FNO Visualization API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: Dict[str, dict] = {}
JOBS_LOCK = Lock()


class HeatJobRequest(BaseModel):
    model: Literal["pinn"] = "pinn"
    equation: Literal["heat"] = "heat"
    nx: int = Field(default=101, ge=21, le=201)
    ny: int = Field(default=101, ge=21, le=201)
    nu: float = Field(default=1.0, ge=0.01, le=2.0)
    dt: float = Field(default=1e-5, gt=0, le=1e-3)
    short_steps: int = Field(default=60, ge=10, le=300)
    long_steps: int = Field(default=120, ge=20, le=600)
    init_mode: Literal["sin"] = "sin"
    noise_level: float = Field(default=0.002, ge=0.0, le=0.05)
    seed: int = Field(default=42, ge=0, le=10**9)


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "success", "failed"]
    created_at: float


def _validate_heat_params(req: HeatJobRequest) -> None:
    dx = 1.0 / req.nx
    dy = 1.0 / req.ny
    # 2D explicit stability bound: dt <= dx^2*dy^2 / (2*nu*(dx^2+dy^2))
    stable_dt = (dx * dx * dy * dy) / (2.0 * req.nu * (dx * dx + dy * dy))
    if req.dt > stable_dt:
        raise HTTPException(
            status_code=400,
            detail=f"dt too large for stability. got {req.dt:.3e}, require <= {stable_dt:.3e}",
        )
    if req.long_steps < req.short_steps:
        raise HTTPException(status_code=400, detail="long_steps must be >= short_steps")


def _model_dir(model: str) -> Path:
    d = EXPORT_ROOT / model.lower()
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
    return d


def _run_dir(model: str, epoch: str) -> Path:
    d = _model_dir(model) / epoch
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Run '{model}/{epoch}' not found")
    return d


def _execute_job(job_id: str, req: HeatJobRequest) -> None:
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
    try:
        params = HeatJobParams(
            model=req.model,
            equation=req.equation,
            epoch=int(job_id[-6:]),
            nx=req.nx,
            ny=req.ny,
            nu=req.nu,
            dt=req.dt,
            short_steps=req.short_steps,
            long_steps=req.long_steps,
            init_mode=req.init_mode,
            noise_level=req.noise_level,
            seed=req.seed,
        )
        out_dir = run_heat_job(ROOT, params)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "success"
            JOBS[job_id]["finished_at"] = time.time()
            JOBS[job_id]["run"] = out_dir.name
            JOBS[job_id]["model"] = req.model
            JOBS[job_id]["output_dir"] = str(out_dir)
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["finished_at"] = time.time()
            JOBS[job_id]["error"] = str(e)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/jobs", response_model=JobResponse)
def create_job(req: HeatJobRequest, bg: BackgroundTasks):
    _validate_heat_params(req)
    epoch = int(time.time() * 1000) % 1000000
    job_id = f"job_{epoch:06d}"
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "request": req.model_dump(),
        }
    bg.add_task(_execute_job, job_id, req)
    return JobResponse(job_id=job_id, status="queued", created_at=JOBS[job_id]["created_at"])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    return j


@app.get("/api/models")
def get_models():
    if not EXPORT_ROOT.exists():
        return {"models": []}
    models = sorted([p.name for p in EXPORT_ROOT.iterdir() if p.is_dir()])
    return {"models": models}


@app.get("/api/models/{model}/runs")
def get_runs(model: str):
    model_dir = _model_dir(model)
    runs = sorted([p.name for p in model_dir.iterdir() if p.is_dir()])
    return {"model": model, "runs": runs}


@app.get("/api/models/{model}/{epoch}/meta")
def get_meta(model: str, epoch: str):
    p = _run_dir(model, epoch) / "meta.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="meta.json not found")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/models/{model}/{epoch}/metrics")
def get_metrics(model: str, epoch: str):
    p = _run_dir(model, epoch) / "metrics.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return json.loads(p.read_text(encoding="utf-8"))


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
    p = _run_dir(model, epoch) / f"{kind}.npy"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{kind}.npy not found")
    arr = np.load(p)
    if t >= arr.shape[0]:
        raise HTTPException(status_code=400, detail=f"t out of range, max={arr.shape[0]-1}")
    return {
        "model": model,
        "epoch": epoch,
        "kind": kind,
        "t": t,
        "shape": list(arr.shape),
        "field": arr[t].tolist(),
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
            "message": "FNO comparison placeholder: right model export missing",
        }

    left_arr = np.load(left_path)
    right_arr = np.load(right_path)
    if left_arr.shape != right_arr.shape:
        raise HTTPException(status_code=400, detail="shape mismatch between compared arrays")
    if t >= left_arr.shape[0]:
        raise HTTPException(status_code=400, detail=f"t out of range, max={left_arr.shape[0]-1}")

    diff = left_arr[t] - right_arr[t]
    return {
        "epoch": epoch,
        "left": left,
        "right": right,
        "split": split,
        "t": t,
        "available": True,
        "shape": list(left_arr.shape),
        "left_field": left_arr[t].tolist(),
        "right_field": right_arr[t].tolist(),
        "diff_field": diff.tolist(),
    }
