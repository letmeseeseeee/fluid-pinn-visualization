# Backend (FastAPI)

## Run

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

## API
- `GET /api/models`
- `GET /api/models/{model}/runs`
- `GET /api/models/{model}/{epoch}/meta`
- `GET /api/models/{model}/{epoch}/metrics`
- `GET /api/models/{model}/{epoch}/field?kind=prediction_short&t=0`
- `GET /api/compare/{epoch}?left=pinn&right=fno&t=0&split=short`
