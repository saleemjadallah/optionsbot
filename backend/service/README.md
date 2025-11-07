# Ensemble Service

This package exposes the `ModelEnsemble` universe scanner as a standalone FastAPI
service suitable for deployment on platforms like Railway or fly.io. It allows
the Streamlit frontend (or any other client) to POST prepared option chain data
and receive fully structured trade ideas without importing the heavy ensemble
modules inside the UI process.

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `ENSEMBLE_SERVICE_URL` | Client-side URL to reach the service (used by the frontend). | `""` |
| `ENSEMBLE_RISK_LEVEL` | Risk appetite (`low`, `moderate`, or `high`). | `moderate` |
| `SERVICE_HOST` | Host binding for uvicorn. | `0.0.0.0` |
| `SERVICE_PORT` | Listener port. | `8000` |
| `SERVICE_RELOAD` | Enable autoreload (for local development). | `false` |
| `SERVICE_LOG_LEVEL` | Uvicorn log level. | `info` |
| `CORS_ALLOW_ORIGINS` | Comma-separated origins allowed for CORS. | `*` |

## Running Locally

```bash
cd backend
poetry install  # or pip install -r requirements.txt
export SERVICE_PORT=8000
export TASTYTRADE_USERNAME="your_email"
export TASTYTRADE_PASSWORD="your_password"
python -m service.main
```

The service starts with two endpoints:

- `GET /health` – lightweight readiness check
- `POST /ensemble/ideas` – accepts a `UniverseScanRequest` payload and returns
  a list of `EnsembleIdea` objects.
- `POST /market-data/options` – fetches trimmed option chains plus DXLink quotes/Greeks
  for the requested symbols. Requires Tastytrade credentials via environment variables.

## Railway Deployment

1. Create a new Railway service and attach this repository.
2. Set the start command to:
   ```bash
   python -m service.main
   ```
3. Configure environment variables:
   - `SERVICE_PORT=8000`
   - `ENSEMBLE_RISK_LEVEL=moderate` (or desired risk level)
   - `CORS_ALLOW_ORIGINS=https://your-frontend-domain`
4. Optionally set `SERVICE_LOG_LEVEL=debug` during initial rollouts.

Expose the generated Railway URL via `ENSEMBLE_SERVICE_URL` in the frontend
deployment so the Streamlit app knows where to POST universe scan requests.
