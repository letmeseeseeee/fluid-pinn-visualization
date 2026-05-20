from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Tuple


PINN_JOB_LOCK = Lock()
_running_processes: dict[str, "subprocess.Popen[bytes] | subprocess.Popen[str]"] = {}
_proc_lock = Lock()


@dataclass
class PinnOriginalJobParams:
    equation: str = "heat"
    nx: int = 101
    ny: int = 101
    nu: float = 1.0
    dt: float = 1e-5
    short_steps: int = 100
    long_steps: int = 100
    epochs: int = 108000
    learning_rate: float = 1e-5
    seed: int = 50976
    transformer_hidden_channels: int = 128
    patch_size: int = 16
    num_heads: int = 4
    num_layers: int = 4
    network_type: str = "transformer"
    loss_phy_weight: float = 1.0
    loss_data_weight: float = 0.0
    use_cache: bool = True


def _normalize_float(value: float) -> float:
    return float(f"{float(value):.12g}")


def _float_close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)


def runtime_signature(params: PinnOriginalJobParams) -> Dict[str, Any]:
    return {
        "equation": str(params.equation).lower(),
        "nx": int(params.nx),
        "ny": int(params.ny),
        "nu": _normalize_float(params.nu),
        "dt": _normalize_float(params.dt),
        "short_steps": int(params.short_steps),
        "long_steps": int(params.long_steps),
        "epochs": int(params.epochs),
        "learning_rate": _normalize_float(params.learning_rate),
        "seed": int(params.seed),
        "network_type": str(params.network_type).lower(),
        "transformer_hidden_channels": int(params.transformer_hidden_channels),
        "patch_size": int(params.patch_size),
        "num_heads": int(params.num_heads),
        "num_layers": int(params.num_layers),
        "loss_phy_weight": _normalize_float(params.loss_phy_weight),
        "loss_data_weight": _normalize_float(params.loss_data_weight),
    }


def _legacy_signature(meta: Dict[str, Any]) -> Dict[str, Any] | None:
    extra = meta.get("extra") or {}
    if extra.get("source") not in {"Hea0.1.19.py/test", "pinn_offline_export", "original_pinn_result"}:
        return None
    grid = meta.get("grid") or {}
    frames = meta.get("frames") or {}
    return {
        "equation": "heat",
        "nx": int(grid.get("width", 101)),
        "ny": int(grid.get("height", 101)),
        "nu": _normalize_float(extra.get("nu", 1.0)),
        "dt": _normalize_float(grid.get("dt", 1e-5)),
        "short_steps": int(frames.get("short", 101)) - 1,
        "long_steps": int(frames.get("long", 101)) - 1,
        "epochs": int(meta.get("epoch", 108000)),
        "learning_rate": _normalize_float(1e-5),
        "seed": 50976,
        "network_type": "transformer",
        "transformer_hidden_channels": 128,
        "patch_size": 16,
        "num_heads": 4,
        "num_layers": 4,
        "loss_phy_weight": _normalize_float(1.0),
        "loss_data_weight": _normalize_float(0.0),
    }


def _meta_signature(meta: Dict[str, Any]) -> Dict[str, Any] | None:
    extra = meta.get("extra") or {}
    runtime = extra.get("runtime_config")
    if runtime:
        return {
            "equation": str(runtime.get("equation", "heat")).lower(),
            "nx": int(runtime.get("nx", meta.get("grid", {}).get("width", 101))),
            "ny": int(runtime.get("ny", meta.get("grid", {}).get("height", 101))),
            "nu": _normalize_float(runtime.get("nu", extra.get("nu", 1.0))),
            "dt": _normalize_float(runtime.get("dt", meta.get("grid", {}).get("dt", 1e-5))),
            "short_steps": int(runtime.get("short_steps", meta.get("frames", {}).get("short", 101) - 1)),
            "long_steps": int(runtime.get("long_steps", meta.get("frames", {}).get("long", 101) - 1)),
            "epochs": int(runtime.get("epochs", meta.get("epoch", 0))),
            "learning_rate": _normalize_float(runtime.get("learning_rate", 1e-5)),
            "seed": int(runtime.get("seed", 50976)),
            "network_type": str(runtime.get("network_type", "transformer")).lower(),
            "transformer_hidden_channels": int(runtime.get("transformer_hidden_channels", 128)),
            "patch_size": int(runtime.get("patch_size", 16)),
            "num_heads": int(runtime.get("num_heads", 4)),
            "num_layers": int(runtime.get("num_layers", 4)),
            "loss_phy_weight": _normalize_float(runtime.get("loss_phy_weight", 1.0)),
            "loss_data_weight": _normalize_float(runtime.get("loss_data_weight", 0.0)),
        }
    return _legacy_signature(meta)


def _signatures_match(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    for key, value in expected.items():
        other = actual.get(key)
        if isinstance(value, float):
            if other is None or not _float_close(value, other):
                return False
        else:
            if other != value:
                return False
    return True


def _safe_float_token(value: float) -> str:
    token = f"{float(value):.6g}"
    token = token.replace(".", "p").replace("+", "")
    token = token.replace("-", "m")
    return token


def build_run_name(params: PinnOriginalJobParams) -> str:
    parts = [
        f"epoch_{int(params.epochs):06d}",
        f"{params.equation.lower()}",
        f"n{int(params.nx)}",
        f"nu{_safe_float_token(params.nu)}",
        f"dt{_safe_float_token(params.dt)}",
        f"s{int(params.short_steps)}",
        f"l{int(params.long_steps)}",
        f"lr{_safe_float_token(params.learning_rate)}",
        f"seed{int(params.seed)}",
        f"ps{int(params.patch_size)}",
        f"h{int(params.num_heads)}",
        f"lay{int(params.num_layers)}",
    ]
    return "__".join(parts)


def _python_executable(root_dir: Path) -> Path:
    preferred = root_dir / ".venv-cuda" / "Scripts" / "python.exe"
    if preferred.exists():
        return preferred
    return Path("python")


def find_cached_pinn_run(root_dir: str | Path, params: PinnOriginalJobParams) -> Path | None:
    root = Path(root_dir)
    model_dir = root / "web_exports" / "pinn"
    if not model_dir.exists():
        return None

    expected = runtime_signature(params)
    candidates = sorted([p for p in model_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in candidates:
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        actual = _meta_signature(meta)
        if actual and _signatures_match(expected, actual):
            return run_dir
    return None


def run_original_pinn_job(root_dir: str | Path, params: PinnOriginalJobParams) -> Tuple[Path, bool]:
    root = Path(root_dir)
    if params.use_cache:
        cached = find_cached_pinn_run(root, params)
        if cached:
            return cached, True

    run_name = build_run_name(params)
    out_dir = root / "web_exports" / "pinn" / run_name
    if out_dir.exists():
        return out_dir, True

    logs_dir = root / "output" / "pinn_jobs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}.log"

    command = [
        str(_python_executable(root)),
        str(root / "Hea0.1.19.py"),
        "--equation", params.equation,
        "--nx", str(params.nx),
        "--ny", str(params.ny),
        "--nu", str(params.nu),
        "--dt", str(params.dt),
        "--short-steps", str(params.short_steps),
        "--long-steps", str(params.long_steps),
        "--epochs", str(params.epochs),
        "--learning-rate", str(params.learning_rate),
        "--seed", str(params.seed),
        "--transformer-hidden-channels", str(params.transformer_hidden_channels),
        "--patch-size", str(params.patch_size),
        "--num-heads", str(params.num_heads),
        "--num-layers", str(params.num_layers),
        "--network-type", str(params.network_type),
        "--loss-phy-weight", str(params.loss_phy_weight),
        "--loss-data-weight", str(params.loss_data_weight),
        "--run-name", run_name,
        "--export-interval", "0",
    ]

    with PINN_JOB_LOCK:
        with _proc_lock:
            _running_processes[run_name] = None

        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                command,
                cwd=str(root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with _proc_lock:
                _running_processes[run_name] = proc

        try:
            completed_code = proc.wait()
        finally:
            with _proc_lock:
                _running_processes.pop(run_name, None)

        if completed_code != 0:
            tail_lines = []
            try:
                with log_path.open("r", encoding="utf-8") as lf:
                    lines = lf.readlines()
                    tail_lines = [line.rstrip() for line in lines[-8:]]
            except Exception:
                pass
            tail = "\n".join(tail_lines) if tail_lines else "(log empty or unreadable)"
            raise RuntimeError(
                f"PINN solver task failed (exit code {completed_code}).\n"
                f"Log tail:\n{tail}\n"
                f"Full log: {log_path}"
            )
    if not out_dir.exists():
        raise RuntimeError(f"Expected output run not found after solve: {out_dir}")
    return out_dir, False


def serialize_runtime_signature(params: PinnOriginalJobParams) -> Dict[str, Any]:
    return runtime_signature(params)


def cancel_pinn_job(job_id: str) -> bool:
    with _proc_lock:
        proc = _running_processes.get(job_id)
    if proc is None or proc.poll() is not None:
        return False
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return True
