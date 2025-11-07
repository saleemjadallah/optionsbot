# Streamlit Cloud: Enable Universe Scanning

## For Streamlit Cloud Deployment

Since Streamlit Cloud runs only your frontend app (you can't run a separate backend service there), you must use **Option 2: Built-in Ensemble**.

---

## Configuration Steps

### 1. Add Environment Variables in Streamlit Cloud

Go to your Streamlit Cloud dashboard:
- Navigate to your app settings
- Click **"⚙️ Settings"** → **"Secrets"**
- Add the following in **TOML format**:

```toml
# Enable the built-in ensemble (required for universe scanning)
ENABLE_FRONTEND_ENSEMBLE = "true"

# Ensemble configuration
ENSEMBLE_RISK_LEVEL = "moderate"
ENSEMBLE_EDGE_THRESHOLD = "0.00"
ENSEMBLE_CONFIDENCE_THRESHOLD = "0.00"

# Your existing Tastytrade credentials
TASTYTRADE_USERNAME = "tastyslim"
TASTYTRADE_EMAIL = "saleem86@gmail.com"
# Add your Tastytrade password if not already set
```

### 2. Verify Backend Dependencies

Ensure your `frontend/requirements.txt` includes the backend dependencies needed for the ensemble. Check if these are present:

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

If not, you may need to add them or reference the backend requirements.

### 3. Update Python Path (if needed)

The code at [frontend/utils/strategy_engine.py:27-31](frontend/utils/strategy_engine.py#L27-L31) automatically adds the backend to the Python path when `ENABLE_FRONTEND_ENSEMBLE` is true:

```python
BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
ENABLE_FRONTEND_ENSEMBLE = os.getenv("ENABLE_FRONTEND_ENSEMBLE", "").strip().lower() in {"1", "true", "yes"}

if ENABLE_FRONTEND_ENSEMBLE and BACKEND_ROOT.exists() and str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))
```

This should work automatically in Streamlit Cloud.

---

## Important Notes

### Repository Structure

Your Streamlit Cloud deployment must have access to **both** `frontend/` and `backend/` directories:

```
OptionsTrader/
├── frontend/
│   ├── app.py              # Main file path in Streamlit Cloud
│   ├── utils/
│   │   └── strategy_engine.py
│   └── requirements.txt
└── backend/
    └── strategies/
        └── model_ensemble.py  # Imported by frontend when enabled
```

### Why Not Option 1 (Microservice)?

Streamlit Cloud:
- ❌ Only runs your Streamlit app
- ❌ Can't run separate FastAPI services
- ❌ Can't easily connect to external services behind firewalls

If you want to use the microservice approach, you'd need to:
1. Deploy the backend service separately (Railway, Render, Heroku, etc.)
2. Set `ENSEMBLE_SERVICE_URL` to point to that service
3. Handle CORS and authentication

---

## Streamlit Cloud Secrets Format

**Critical**: Streamlit Cloud uses TOML format, not `.env` format!

✅ **Correct** (TOML):
```toml
ENABLE_FRONTEND_ENSEMBLE = "true"
ENSEMBLE_EDGE_THRESHOLD = "0.00"
```

❌ **Wrong** (.env format won't work):
```bash
ENABLE_FRONTEND_ENSEMBLE=true
ENSEMBLE_EDGE_THRESHOLD=0.00
```

---

## Step-by-Step: Add Secrets to Streamlit Cloud

1. **Go to your Streamlit Cloud dashboard**
   - URL: https://share.streamlit.io/

2. **Select your app**
   - Find "Options Trading Bot" in your apps list

3. **Open Settings**
   - Click the **"⚙️"** menu
   - Select **"Settings"**

4. **Add Secrets**
   - Click **"Secrets"** tab
   - Paste the TOML configuration:

   ```toml
   # Universe Scanning Configuration
   ENABLE_FRONTEND_ENSEMBLE = "true"
   ENSEMBLE_RISK_LEVEL = "moderate"
   ENSEMBLE_EDGE_THRESHOLD = "0.00"
   ENSEMBLE_CONFIDENCE_THRESHOLD = "0.00"

   # Tastytrade Credentials (if not already set)
   TASTYTRADE_USERNAME = "tastyslim"
   TASTYTRADE_EMAIL = "saleem86@gmail.com"
   ```

5. **Save**
   - Click **"Save"**
   - Your app will automatically restart with the new configuration

---

## Verification

After deploying with `ENABLE_FRONTEND_ENSEMBLE = "true"`:

### Check Logs

In Streamlit Cloud, click **"Manage app"** → **"Logs"** and look for:

```
INFO: Loading ModelEnsemble...
INFO: Analyzing X symbols from universe...
```

### Test Universe Scan

1. Navigate to the "Universe Scan" or "Trade Ideas" section
2. Click "Scan Universe" or "Refresh Ideas"
3. You should see trade recommendations appear

### If You Get Zero Ideas

Even with edge/confidence at 0%, if you still get no ideas:

1. **Check backend imports**: Look for import errors in logs
2. **Verify option chains**: Ensure market data is fetching correctly
3. **Check model initialization**: ModelEnsemble must load successfully
4. **Inspect symbols**: Verify your universe contains valid tickers

---

## Performance Considerations

### Built-in Ensemble on Streamlit Cloud

⚠️ **Important Performance Notes**:

- **Slower**: Running the ensemble in-process is heavier than the microservice
- **Memory**: May hit Streamlit Cloud's memory limits with large universes
- **Cold starts**: First scan after wake-up will be slower

### Optimization Tips

1. **Limit universe size**: Keep to 20-50 symbols max
2. **Cache aggressively**: The code should cache option chains
3. **Monitor memory**: Watch Streamlit Cloud memory usage
4. **Consider upgrading**: Streamlit Cloud paid tiers have more resources

---

## Alternative: Deploy Backend Service Separately

If the built-in ensemble is too slow or hits memory limits, consider:

### Option A: Deploy Backend to Railway/Render

1. Deploy `backend/service/` as a separate FastAPI app
2. Get the service URL (e.g., `https://your-backend.railway.app`)
3. In Streamlit Cloud secrets, use:

```toml
ENSEMBLE_SERVICE_URL = "https://your-backend.railway.app"
# Remove or set to false:
# ENABLE_FRONTEND_ENSEMBLE = "false"
```

### Option B: Use Railway for Both

Deploy both frontend and backend on Railway with internal networking.

---

## Troubleshooting

### Import Error: "No module named 'strategies'"

**Cause**: Backend dependencies not installed or path issue

**Fix**:
1. Check that `backend/` folder is in your repository
2. Verify the path resolution at [strategy_engine.py:27](frontend/utils/strategy_engine.py#L27)
3. Ensure backend dependencies are in requirements.txt

### Memory Limit Exceeded

**Cause**: Ensemble + large universe too heavy for free tier

**Fix**:
1. Reduce universe size (try 20 symbols)
2. Upgrade to Streamlit Cloud paid tier
3. Or deploy backend separately

### Still Getting Zero Ideas

**Cause**: Models may not be producing predictions

**Fix**:
1. Check logs for actual model output
2. Verify option chains are populating
3. Test with a small universe (SPY, AAPL, etc.)
4. Enable debug logging

---

## Summary: Quick Setup

For Streamlit Cloud, use this minimal configuration:

**Streamlit Cloud Secrets** (⚙️ Settings → Secrets):
```toml
ENABLE_FRONTEND_ENSEMBLE = "true"
ENSEMBLE_EDGE_THRESHOLD = "0.00"
ENSEMBLE_CONFIDENCE_THRESHOLD = "0.00"
```

**Save** → App auto-restarts → Universe scanning should now work!

---

## Next Steps After Enabling

1. ✅ Verify ensemble loads without errors
2. ✅ Run a test scan with a small universe (5-10 symbols)
3. ✅ Check that trade ideas appear
4. ✅ Inspect the actual edge/confidence values produced
5. ✅ Gradually increase universe size and adjust thresholds

Once confirmed working, you can tune the thresholds back to reasonable values like:
```toml
ENSEMBLE_EDGE_THRESHOLD = "0.02"      # 2%
ENSEMBLE_CONFIDENCE_THRESHOLD = "0.40"  # 40%
```
