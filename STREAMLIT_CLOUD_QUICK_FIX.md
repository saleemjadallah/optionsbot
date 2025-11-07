# Quick Fix: Enable Universe Scanning on Streamlit Cloud

## The Issue
Your universe scan returns 0 trades because neither `ENSEMBLE_SERVICE_URL` nor `ENABLE_FRONTEND_ENSEMBLE` is configured.

## The Solution for Streamlit Cloud

Since you're on **Streamlit Cloud** (not running locally), follow these steps:

---

## Step 1: Add Secrets in Streamlit Cloud

1. Go to your app dashboard: https://share.streamlit.io/
2. Find your "Options Trading Bot" app
3. Click **⚙️** (menu) → **Settings** → **Secrets**
4. Add this configuration (TOML format):

```toml
# Enable built-in ensemble
ENABLE_FRONTEND_ENSEMBLE = "true"

# Set thresholds to zero to see all predictions
ENSEMBLE_EDGE_THRESHOLD = "0.00"
ENSEMBLE_CONFIDENCE_THRESHOLD = "0.00"

# Risk level
ENSEMBLE_RISK_LEVEL = "moderate"
```

5. Click **Save**
6. Your app will automatically restart

---

## Step 2: Update Frontend Requirements

Your `frontend/requirements.txt` needs additional dependencies for the ensemble to work.

Add these lines to [frontend/requirements.txt](frontend/requirements.txt):

```txt
# Required for ModelEnsemble (backend integration)
scikit-learn>=1.3.0
scipy>=1.10.0
torch>=2.0.0
joblib>=1.3.0
```

**Why?** The built-in ensemble imports `ModelEnsemble` from the backend, which requires ML libraries.

---

## Step 3: Commit and Push

After updating `frontend/requirements.txt`:

```bash
git add frontend/requirements.txt
git commit -m "Add ML dependencies for ensemble support"
git push
```

Streamlit Cloud will automatically redeploy with the new dependencies.

---

## Step 4: Verify It's Working

After redeployment:

1. Open your Streamlit Cloud app
2. Navigate to the Universe Scan section
3. Click "Scan" or "Refresh"
4. **Check logs**: Click **Manage app** → **Logs**

You should see:
```
INFO: Analyzing X symbols...
```

And trade ideas should appear!

---

## If It Still Doesn't Work

### Issue: Import Errors

**Symptom**: Logs show `ModuleNotFoundError: No module named 'strategies'`

**Cause**: Streamlit Cloud can't find the backend folder

**Fix**: The code at [strategy_engine.py:27-31](frontend/utils/strategy_engine.py#L27-L31) should handle this automatically. Verify your repo structure:

```
OptionsTrader/
├── frontend/
│   ├── app.py
│   └── utils/
└── backend/
    └── strategies/
        └── model_ensemble.py
```

### Issue: Memory Limit

**Symptom**: App crashes or freezes during scan

**Cause**: Free tier memory limit (1GB)

**Fix**:
1. Reduce universe size to 20-30 symbols
2. Or upgrade to Streamlit Cloud paid tier

### Issue: Still Zero Ideas

**Symptom**: Scan completes but returns 0 ideas

**Cause**: Models may not be producing predictions with sufficient edge/confidence

**Fix**:
1. Check logs for actual predictions
2. Verify option chains are loading
3. Test with liquid symbols (SPY, AAPL, MSFT)

---

## Alternative: Deploy Backend Service

If the built-in ensemble is too slow or memory-intensive, deploy the backend separately:

### Quick Railway Deployment

1. Go to [Railway.app](https://railway.app)
2. Click **"New Project"** → **"Deploy from GitHub"**
3. Select your repo → choose `backend/service/` folder
4. Railway will auto-detect FastAPI and deploy
5. Copy the service URL (e.g., `https://your-app.railway.app`)

### Update Streamlit Cloud Secrets

Replace `ENABLE_FRONTEND_ENSEMBLE` with:

```toml
# Use external service instead
ENSEMBLE_SERVICE_URL = "https://your-app.railway.app"
ENSEMBLE_SERVICE_TIMEOUT = "20"

# Remove or comment out:
# ENABLE_FRONTEND_ENSEMBLE = "false"
```

This offloads the heavy ensemble work to Railway's servers.

---

## Recommended: Minimal Frontend Changes

To ensure the frontend works on Streamlit Cloud with the built-in ensemble, I recommend updating `frontend/requirements.txt`:

### Current State
```txt
# Missing ML dependencies
pandas>=1.5.3
numpy>=1.24.3
```

### Add These
```txt
# ML dependencies for ModelEnsemble
scikit-learn>=1.3.0
scipy>=1.10.0
torch>=2.0.0
joblib>=1.3.0
```

**Why torch?** The backend's `ModelEnsemble` likely uses PyTorch models. Without it, imports will fail.

---

## Summary: What You Need to Do

✅ **Streamlit Cloud Secrets** (add via web UI):
```toml
ENABLE_FRONTEND_ENSEMBLE = "true"
ENSEMBLE_EDGE_THRESHOLD = "0.00"
ENSEMBLE_CONFIDENCE_THRESHOLD = "0.00"
```

✅ **Update `frontend/requirements.txt`** (commit & push):
```txt
scikit-learn>=1.3.0
scipy>=1.10.0
torch>=2.0.0
joblib>=1.3.0
```

✅ **Push to GitHub** → Streamlit Cloud auto-redeploys

✅ **Test** → Should now generate trade ideas!

---

## Need Help?

See the full guide: [STREAMLIT_CLOUD_ENSEMBLE_SETUP.md](STREAMLIT_CLOUD_ENSEMBLE_SETUP.md)
