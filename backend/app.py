from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / "web_exports"

app = FastAPI(title="Fluid PINN/FNO Visualization API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/api/health")
def health():
    return {"ok": True}


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
        raise HTTPException(status_code=404, detail="comparison arrays not found")

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
        "shape": list(left_arr.shape),
        "left_field": left_arr[t].tolist(),
        "right_field": right_arr[t].tolist(),
        "diff_field": diff.tolist(),
    }
