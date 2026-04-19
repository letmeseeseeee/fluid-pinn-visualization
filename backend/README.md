# Backend (FastAPI) - Heat Equation Online Simulation (PINN v1)

## Run

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

## API

### 1) Submit online heat-equation job (PINN v1)
- `POST /api/jobs`
- body example:

```json
{
  "model": "pinn",
  "equation": "heat",
  "nx": 101,
  "ny": 101,
  "nu": 1.0,
  "dt": 1e-5,
  "short_steps": 60,
  "long_steps": 120,
  "init_mode": "sin",
  "noise_level": 0.002,
  "seed": 42
}
```

### 2) Query job status
- `GET /api/jobs/{job_id}`

### 3) Visualization data APIs
- `GET /api/models`
- `GET /api/models/{model}/runs`
- `GET /api/models/{model}/{epoch}/meta`
- `GET /api/models/{model}/{epoch}/metrics`
- `GET /api/models/{model}/{epoch}/field?kind=prediction_short&t=0`
- `GET /api/compare/{epoch}?left=pinn&right=fno&t=0&split=short`

## Notes
- Current version focuses on `heat + pinn` online task flow.
- FNO compare endpoint is reserved and returns placeholder when FNO data is not available.
