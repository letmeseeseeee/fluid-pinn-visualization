"""Self-contained platform performance benchmark — stdlib only, no extra deps."""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "http://127.0.0.1:8000/api"
ROOT = Path(__file__).resolve().parents[1]
WARMUP = 5
SAMPLES = 100
CONCURRENT_LEVELS = [1, 5, 10, 20, 50]


def fetch(url: str, timeout: float = 30) -> tuple[float, int, str | None]:
    """Return (latency_sec, status_code, error_message)."""
    start = time.perf_counter()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            latency = time.perf_counter() - start
            return latency, resp.status, None
    except urllib.error.HTTPError as e:
        return time.perf_counter() - start, e.code, str(e)
    except Exception as e:
        return time.perf_counter() - start, 0, str(e)


def percentiles(values: list[float]) -> dict:
    s = sorted(values)
    n = len(s)
    return {
        "min": s[0],
        "p50": s[n // 2],
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
        "mean": statistics.mean(s),
        "stdev": statistics.stdev(s) if n > 1 else 0,
    }


def warmup(url: str, n: int = WARMUP):
    for _ in range(n):
        try:
            urllib.request.urlopen(url, timeout=10)
        except Exception:
            pass


def run_latency_bench(url: str, label: str, n: int = SAMPLES) -> dict:
    print(f"  [{label}] warming up ({WARMUP} req)...")
    warmup(url)
    print(f"  [{label}] sampling ({n} req)...")
    latencies = []
    errors = 0
    for i in range(n):
        lat, code, err = fetch(url)
        if err:
            errors += 1
        else:
            latencies.append(lat)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n}...")
    result = percentiles(latencies)
    result["samples"] = len(latencies)
    result["errors"] = errors
    result["error_rate"] = errors / n
    print(f"  [{label}] p50={result['p50']*1000:.1f}ms p95={result['p95']*1000:.1f}ms p99={result['p99']*1000:.1f}ms")
    return result


def run_concurrency_bench(url: str, label: str, concurrency: int, req_per_client: int = 20) -> dict:
    print(f"  [{label}] concurrency={concurrency} ({concurrency*req_per_client} total req)...")
    errors = 0
    latencies = []

    def worker():
        nonlocal errors
        for _ in range(req_per_client):
            lat, code, err = fetch(url, timeout=60)
            if err:
                errors += 1
            else:
                latencies.append(lat)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - start

    total_req = concurrency * req_per_client
    result = percentiles(latencies) if latencies else {}
    result["concurrency"] = concurrency
    result["total_req"] = total_req
    result["errors"] = errors
    result["error_rate"] = errors / total_req if total_req else 1
    result["elapsed_s"] = elapsed
    result["throughput_req_per_s"] = total_req / elapsed if elapsed > 0 else 0
    print(f"  [{label}] conc={concurrency} throughput={result['throughput_req_per_s']:.1f} req/s p95={result.get('p95', 0)*1000:.1f}ms")
    return result


def main():
    # Ensure server is up
    print("=== Checking server ===")
    server_proc = None

    # Use .venv (has uvicorn+fastapi), not .venv-cuda (torch only)
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = sys.executable

    try:
        urllib.request.urlopen(f"{API_BASE}/health", timeout=5)
        print("Server: UP")
    except Exception:
        print("Server is DOWN. Starting...")
        server_proc = subprocess.Popen(
            [str(venv_python), "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(ROOT),
        )
        time.sleep(3)
        for attempt in range(10):
            try:
                urllib.request.urlopen(f"{API_BASE}/health", timeout=3)
                print(f"Server started (attempt {attempt+1})")
                break
            except Exception:
                time.sleep(2)
        else:
            print("ERROR: server failed to start")
            server_proc.kill()
            return

    # Discover available endpoints
    print("\n=== Discovering endpoints ===")
    _, _, _ = fetch(f"{API_BASE}/models")
    _, _, _ = fetch(f"{API_BASE}/pinn/presets")

    # Resolve a field URL for benchmarking
    field_url = None
    meta_single_url = None
    meta_url = f"{API_BASE}/models/pinn/runs"
    try:
        req = urllib.request.Request(meta_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            runs = data.get("runs", [])
            if runs:
                run = runs[0]
                field_url = f"{API_BASE}/models/pinn/{run}/field?kind=prediction_short&t=0"
                meta_single_url = f"{API_BASE}/models/pinn/{run}/meta"
                print(f"  Using run: {run}")
    except Exception as e:
        print(f"  Failed to discover runs: {e}")

    # ---- Latency Benchmarks ----
    print("\n=== 1. API Latency Benchmarks ===")
    latency_results = {}

    latency_results["health"] = run_latency_bench(f"{API_BASE}/health", "health")

    if meta_url:
        latency_results["meta_list"] = run_latency_bench(meta_url, "models/pinn/runs")

    latency_results["presets"] = run_latency_bench(f"{API_BASE}/pinn/presets", "pinn/presets")

    if meta_single_url:
        latency_results["meta_single"] = run_latency_bench(meta_single_url, "models/pinn/{run}/meta")

    if field_url:
        latency_results["field"] = run_latency_bench(field_url, "models/pinn/{run}/field (101×101, t=0)")

    # ---- Concurrency Benchmarks ----
    print("\n=== 2. Concurrency / Throughput Benchmarks ===")
    concurrency_results = []
    target_url = f"{API_BASE}/health"
    for c in CONCURRENT_LEVELS:
        r = run_concurrency_bench(target_url, "health", c, req_per_client=30)
        concurrency_results.append(r)

    if field_url:
        print("\n  Field data under concurrency...")
        for c in [1, 5, 10]:
            r = run_concurrency_bench(field_url, "field", c, req_per_client=10)
            concurrency_results.append(r)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("=== BENCHMARK SUMMARY ===")
    print("=" * 60)

    print("\n--- API Latency (ms) ---")
    print(f"{'Endpoint':<35} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8} {'err%':>6}")
    print("-" * 75)
    for name, res in latency_results.items():
        print(f"{name:<35} {res['p50']*1000:>7.1f} {res['p95']*1000:>7.1f} {res['p99']*1000:>7.1f} {res['mean']*1000:>7.1f} {res['error_rate']*100:>5.1f}%")

    print(f"\n--- Throughput (req/s) ---")
    print(f"{'Endpoint':<35} {'conc':>6} {'req/s':>8} {'p95(ms)':>8} {'err%':>6}")
    print("-" * 70)
    for res in concurrency_results:
        name = res.get("endpoint", "health")[:33]
        print(f"{name:<35} {res['concurrency']:>5} {res['throughput_req_per_s']:>7.1f} {res.get('p95', 0)*1000:>7.1f} {res['error_rate']*100:>5.1f}%")

    # Save report
    report = {
        "latency": latency_results,
        "concurrency": concurrency_results,
    }

    class BenchEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            return super().default(obj)

    report_path = ROOT / "output" / "benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, cls=BenchEncoder), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    # Shutdown server if we started it
    try:
        if server_proc is not None:
            print("\nShutting down server...")
            server_proc.terminate()
            server_proc.wait(timeout=5)
    except NameError:
        pass  # server was already running, leave it


if __name__ == "__main__":
    main()
