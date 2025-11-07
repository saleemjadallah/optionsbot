# Streamlit Frontend Deployment Guide

Your backend API is now running on Railway at:
**https://optionsbot-production.up.railway.app**

Now you need to deploy the Streamlit frontend. Here are your options:

---

## Option 1: Streamlit Community Cloud (Recommended - FREE)

### Why This Option?
- ✅ **Free hosting** for public repos
- ✅ **Easy setup** (3 clicks)
- ✅ **Auto-deploys** on git push
- ✅ **Built for Streamlit** apps
- ✅ **SSL/Custom domains** included

### Steps:

1. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Select repository: `saleemjadallah/optionsbot`
   - Branch: `main`
   - Main file path: `frontend/app.py`

3. **Set Secrets (Environment Variables)**
   Click "Advanced settings" → "Secrets" and paste:
   ```toml
   TRADING_BOT_API_URL = "https://optionsbot-production.up.railway.app"
   TASTYTRADE_USERNAME = "saleem86@gmail.com"
   TASTYTRADE_PASSWORD = "olaabdel88"
   ```

   **Important:** Must be in TOML format with quotes!

4. **Deploy!**
   - Click "Deploy"
   - Wait 2-3 minutes
   - Your app will be live at: `https://your-app-name.streamlit.app`

---

## Option 2: Railway (Second Service)

### Why This Option?
- ✅ Both frontend & backend on same platform
- ✅ Easier environment variable sharing
- ⚠️ Costs money (but only ~$5-10/month)

### Steps:

1. **In Railway Dashboard**
   - Go to your project
   - Click "+ New" → "Empty Service"

2. **Connect to GitHub**
   - Connect same repo: `saleemjadallah/optionsbot`
   - Railway will auto-detect Python

3. **Configure Service**
   - Go to Settings → Environment
   - Add variables:
     ```
     TRADING_BOT_API_URL=https://optionsbot-production.up.railway.app
     TASTYTRADE_USERNAME=saleem86@gmail.com
     TASTYTRADE_PASSWORD=olaabdel88
     ```

4. **Use Custom Config**
   Railway needs to know which config file to use for frontend.

   In Settings → Build:
   - Set custom build command: `pip install -r frontend/requirements.txt`
   - Set custom start command: `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`

5. **Deploy**
   - Railway will auto-deploy
   - Get your URL from Settings → Domains

---

## Option 3: Run Locally (Development)

### Steps:

1. **Set Environment Variables**
   ```bash
   export TRADING_BOT_API_URL=https://optionsbot-production.up.railway.app
   export TASTYTRADE_USERNAME=saleem86@gmail.com
   export TASTYTRADE_PASSWORD=olaabdel88
   ```

2. **Install Dependencies**
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

3. **Run Streamlit**
   ```bash
   streamlit run app.py
   ```

4. **Access**
   - Open browser to: http://localhost:8501
   - The frontend will connect to your Railway backend

---

## Current Architecture

```
┌─────────────────────────────────────────┐
│  Streamlit Frontend                     │
│  (Not deployed yet - choose option)     │
│                                         │
│  - User Interface                       │
│  - Tastytrade Auth                      │
│  - Strategy Engine                      │
│  - Watchlist Management                 │
└────────────┬────────────────────────────┘
             │
             │ HTTP Requests
             │
             ▼
┌─────────────────────────────────────────┐
│  Railway Backend API                    │
│  https://optionsbot-production.up...    │
│                                         │
│  ✅ FastAPI Service                     │
│  ✅ DXLink Streaming                    │
│  ✅ Greeks Calculation                  │
│  ✅ Model Ensemble                      │
│  ✅ Market Data Service                 │
└────────────┬────────────────────────────┘
             │
             │ WebSocket + REST
             │
             ▼
┌─────────────────────────────────────────┐
│  Tastytrade API                         │
│  (Production)                           │
│                                         │
│  - Real-time Market Data                │
│  - Options Chains                       │
│  - Greeks & Theoretical Pricing         │
└─────────────────────────────────────────┘
```

---

## Recommended Setup

**For Production:**
1. **Backend**: Railway (already done ✅)
2. **Frontend**: Streamlit Community Cloud (free & easy)

**For Development:**
1. **Backend**: Railway (already done ✅)
2. **Frontend**: Local (run with streamlit run)

---

## Testing Your Setup

Once frontend is deployed:

1. **Visit Frontend URL**
   - Streamlit Cloud: `https://your-app.streamlit.app`
   - Railway: `https://your-frontend.railway.app`
   - Local: `http://localhost:8501`

2. **Test Authentication**
   - Go to Tastytrade Auth page
   - Login with credentials
   - Should see account info

3. **Test Market Data**
   - Add symbols to watchlist (AAPL, MSFT, etc.)
   - Click "Refresh Now"
   - Should see options strategies appear

4. **Check Backend Connection**
   - Frontend will try: `POST /market-data/options` to Railway
   - If successful, you'll see Greeks and real-time data
   - If it fails, check CORS settings in backend

---

## Troubleshooting

### Frontend can't connect to backend
- Check `TRADING_BOT_API_URL` is set correctly
- Verify CORS origins in backend `app.py` includes your frontend URL

### Authentication fails
- Check `TASTYTRADE_USERNAME` and `TASTYTRADE_PASSWORD` are set
- Verify credentials are for **production** (not sandbox)

### No market data appears
- Check Railway backend logs for errors
- Verify backend `/health` endpoint returns "ok"
- Test backend directly:
  ```bash
  curl https://optionsbot-production.up.railway.app/health
  ```

---

## Cost Breakdown

### FREE Option:
- Backend: Railway (~$5-10/month)
- Frontend: Streamlit Cloud (FREE)
- **Total: $5-10/month**

### All Railway:
- Backend: Railway (~$5-10/month)
- Frontend: Railway (~$5-10/month)
- **Total: $10-20/month**

### Local Development:
- Backend: Railway (~$5-10/month)
- Frontend: Local (FREE)
- **Total: $5-10/month** (best for testing)

---

**I recommend Streamlit Community Cloud for the frontend - it's free and built specifically for Streamlit apps!**

🚀 Your backend is ready. Deploy the frontend using Option 1 above!
