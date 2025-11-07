# 🚀 Quick Start - Tastytrade Authentication

## ✅ URL Fixed!

The correct sandbox URL is now configured: `https://api.cert.tastyworks.com`

## One Command to Start Everything

```bash
cd /Users/saleemjadallah/Desktop/OptionsTrader
./start_tastytrade.sh
```

This lightweight script starts:
- ✅ Tastytrade API backend (port 8000)
- ✅ Streamlit dashboard (port 8501)
- ✅ No heavy dependencies required!

---

## What's Different?

This uses a **simplified backend** (`endpoints_tastytrade_only.py`) that:
- Only includes Tastytrade integration
- No trading bot dependencies
- Fast startup
- Perfect for authentication testing

---

## Authenticate in 3 Steps

### 1. Start the App
```bash
./start_tastytrade.sh
```

### 2. Open Dashboard
Navigate to: **http://localhost:8501**

### 3. Connect Tastytrade
1. Click **"Tastytrade"** tab
2. Click **"Connect to Tastytrade"**
3. Login:
   - Username: `tastyslim`
   - Password: `saleemjadallah1986`
4. Click **"Authorize"**
5. Click **"Refresh Status"** back in dashboard

---

## ✅ You Should See

After authenticating:
- ✅ Account number displayed
- ✅ Environment: SANDBOX
- ✅ Account info tab
- ✅ Positions tab
- ✅ Balance tab

---

## 🐛 If Something Goes Wrong

**Backend won't start:**
```bash
# Check logs
tail -f logs/tastytrade_backend.log

# Try manual start
cd backend
uvicorn api.endpoints_tastytrade_only:app --port 8000
```

**Frontend won't connect:**
```bash
# Check if backend is running
curl http://localhost:8000/health

# Check frontend logs
tail -f logs/tastytrade_frontend.log
```

**Authentication fails:**
```bash
# Verify credentials in .env
cat backend/.env | grep TASTYTRADE

# Clear tokens and retry
rm backend/credentials/tokens.json
```

---

## 📚 Full Documentation

- **Complete Guide**: `TASTYTRADE_COMPLETE_SUMMARY.md`
- **API Reference**: `TASTYTRADE_API_DOCUMENTATION.md`
- **Frontend Integration**: `TASTYTRADE_FRONTEND_INTEGRATION.md`

---

## 🔧 Using the Full Backend

To use the complete trading bot backend (with all features):

1. Ensure all dependencies are installed
2. Create missing modules if needed
3. Use: `uvicorn backend.api.endpoints:app --port 8000`

For now, the lightweight version (`endpoints_tastytrade_only.py`) is recommended for testing Tastytrade integration!

---

**Ready? Run `./start_tastytrade.sh` and authenticate! 🎉**
