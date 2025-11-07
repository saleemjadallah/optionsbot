# Fix Guide: Enable Universe Scanning for Trade Ideas

## Problem Summary

The universe scan is currently disabled because neither of the two required configurations are active:
- `ENSEMBLE_SERVICE_URL` is not set (microservice path)
- `ENABLE_FRONTEND_ENSEMBLE` is not enabled (local fallback path)

This causes `build_universe_recommendations()` to return an empty list immediately, which is why you see "Loaded 91 symbols..." but get 0 proactive trades.

## Solution: Choose One of Two Options

### Option 1: Use the FastAPI Microservice (RECOMMENDED)

This approach provides better performance, isolation, and uses DXLink for real-time streaming.

#### Step 1: Start the Backend Service

```bash
# Navigate to the backend service directory
cd backend/service

# Start the service on port 8000
SERVICE_PORT=8000 python3 -m service.main
```

You should see output like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Step 2: Verify the Service is Running

Open another terminal and test the health endpoint:

```bash
curl http://localhost:8000/health
```

You should get:
```json
{
  "status": "ok",
  "scanner": "ready",
  "timestamp": "2025-11-07T..."
}
```

#### Step 3: Configure the Frontend

Edit `frontend/.env` (already created for you) and uncomment this line:

```bash
ENSEMBLE_SERVICE_URL=http://localhost:8000
```

Your `frontend/.env` should have:
```bash
ENSEMBLE_SERVICE_URL=http://localhost:8000
ENSEMBLE_SERVICE_TIMEOUT=20
ENSEMBLE_RISK_LEVEL=moderate
ENSEMBLE_EDGE_THRESHOLD=0.00
ENSEMBLE_CONFIDENCE_THRESHOLD=0.00
```

#### Step 4: Launch Streamlit

```bash
cd frontend
streamlit run app.py
```

#### Step 5: Verify It's Working

Watch the backend service logs. When you trigger a universe scan in Streamlit, you should see:
```
INFO: Analyzing X underlying assets
```

If you still get zero ideas, check the service response to see what predictions were returned.

---

### Option 2: Enable Built-in Ensemble (Fallback)

This runs the ensemble locally within Streamlit - heavier but no HTTP dependency.

#### Step 1: Verify Backend Dependencies

Ensure you have the backend dependencies installed:

```bash
cd backend
pip install -r requirements.txt
```

#### Step 2: Configure the Frontend

Edit `frontend/.env` and uncomment this line:

```bash
ENABLE_FRONTEND_ENSEMBLE=true
```

Your `frontend/.env` should have:
```bash
ENABLE_FRONTEND_ENSEMBLE=true
ENSEMBLE_RISK_LEVEL=moderate
ENSEMBLE_EDGE_THRESHOLD=0.00
ENSEMBLE_CONFIDENCE_THRESHOLD=0.00
```

#### Step 3: Launch Streamlit

```bash
cd frontend
streamlit run app.py
```

The ensemble will now run locally within the Streamlit process.

---

## Verifying the Fix

Once configured with either option:

1. **Ensemble loads**: Check that ModelEnsemble imports successfully
2. **Universe scan executes**: Log messages should show "Analyzing X symbols"
3. **Trade ideas appear**: With edge/confidence at 0%, any model output should surface

## Key Files Modified

- **Created**: `frontend/.env` - Frontend environment configuration
- **Reference**: See [frontend/utils/strategy_engine.py:28](frontend/utils/strategy_engine.py#L28) for `ENABLE_FRONTEND_ENSEMBLE` check
- **Reference**: See [frontend/utils/strategy_engine.py:151](frontend/utils/strategy_engine.py#L151) for `ensemble_service_url` check

## Current Threshold Settings

I've already set your thresholds to zero in the `.env` file to allow any predictions through:

```bash
ENSEMBLE_EDGE_THRESHOLD=0.00      # Was 0.02 (2%)
ENSEMBLE_CONFIDENCE_THRESHOLD=0.00 # Was 0.4 (40%)
```

Once the ensemble is active, these zero thresholds will let all model output surface, confirming the pipeline works.

## Troubleshooting

### Backend Service Won't Start

Check if port 8000 is already in use:
```bash
lsof -i :8000
```

Change to a different port:
```bash
SERVICE_PORT=8001 python3 -m service.main
# Then update ENSEMBLE_SERVICE_URL=http://localhost:8001
```

### Import Errors with Local Ensemble

Ensure your Python path includes the backend:
```bash
export PYTHONPATH=/Users/saleemjadallah/Desktop/OptionsTrader/backend:$PYTHONPATH
```

Or the code at [strategy_engine.py:27-31](frontend/utils/strategy_engine.py#L27-L31) should handle this automatically.

### Still Getting Zero Ideas

1. Check service logs for actual predictions returned
2. Verify option chains are being fetched (not empty)
3. Check that symbols in your universe are valid
4. Enable debug logging to see what the ensemble produces

## Next Steps

After enabling either option and confirming the pipeline runs:

1. **Inspect actual predictions**: See what edge/confidence values the models produce
2. **Adjust thresholds**: Set realistic thresholds once you see the model output distribution
3. **Check option chains**: Ensure they're being populated with valid data
4. **Monitor performance**: Watch service logs for errors or slow responses

---

**Configuration Status**: ✅ Frontend `.env` file created with both options ready to enable
