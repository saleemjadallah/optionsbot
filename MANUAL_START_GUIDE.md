# Manual Start Guide - Tastytrade Integration

## ✅ Issue Fixed!

The Streamlit config file had a syntax error. **This has been fixed.**

---

## 🚀 Start Everything (3 Commands)

### Terminal 1 - Start Backend

```bash
cd /Users/saleemjadallah/Desktop/OptionsTrader
source options_trader_env/bin/activate
cd backend
uvicorn api.endpoints_tastytrade_only:app --reload --port 8000
```

**You should see:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Terminal 2 - Start Frontend

```bash
cd /Users/saleemjadallah/Desktop/OptionsTrader
source options_trader_env/bin/activate
cd frontend
streamlit run app.py
```

**You should see:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Step 3 - Open Browser

Open: **http://localhost:8501**

---

## 🔐 Authenticate

1. Click the **"Tastytrade"** tab
2. Click **"Connect to Tastytrade"**
3. Click the authentication link
4. Login:
   - Username: `tastyslim`
   - Password: `saleemjadallah1986`
5. Click **"Authorize"**
6. Return to dashboard
7. Click **"Refresh Status"**

---

## ✅ What Was Fixed

### Problem
The `.streamlit/config.toml` file had invalid TOML syntax on line 83:
```toml
clientSettings = {
    "rtc_config": {...}  # Invalid nested syntax
}
```

### Solution
Commented out the WebRTC configuration section since it wasn't needed and had syntax errors.

---

## 🐛 If Something Still Doesn't Work

### Backend Won't Start

**Check if port 8000 is in use:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Test backend import:**
```bash
cd backend
python -c "from api.endpoints_tastytrade_only import app"
```

### Frontend Won't Start

**Check if port 8501 is in use:**
```bash
lsof -ti:8501 | xargs kill -9
```

**Test streamlit:**
```bash
streamlit --version
```

**Check for errors:**
```bash
cd frontend
streamlit run app.py --server.headless true
```

### Missing Dependencies

**Install required packages:**
```bash
pip install fastapi uvicorn streamlit requests pydantic python-dotenv pandas plotly streamlit-option-menu
```

---

## 📊 Verify Everything Works

Run the test script:
```bash
./test_startup.sh
```

Or use the automated startup script:
```bash
./start_tastytrade.sh
```

---

## 🎯 Quick Checks

**Backend Health:**
```bash
curl http://localhost:8000/health
```

**Backend API Docs:**
Open: http://localhost:8000/docs

**Frontend:**
Open: http://localhost:8501

**Tastytrade API Status:**
```bash
curl http://localhost:8000/api/tastytrade/health
```

---

## ✅ Everything Should Now Work!

The config error is fixed. Just start both services and authenticate!

**Quick start:**
```bash
# Terminal 1
cd backend && uvicorn api.endpoints_tastytrade_only:app --port 8000

# Terminal 2
cd frontend && streamlit run app.py
```

Then open http://localhost:8501 and click the Tastytrade tab!
