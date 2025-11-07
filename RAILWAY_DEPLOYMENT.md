# Railway Deployment Guide

## Quick Start

Your backend service is now configured for Railway deployment with the following files:
- `railway.toml` - Railway-specific configuration
- `nixpacks.toml` - Nixpacks build configuration
- `Procfile` - Process file for web service

## Required Environment Variables

Set these in your Railway project settings:

### Required (Production)
```bash
TASTYTRADE_USERNAME=saleem86@gmail.com
TASTYTRADE_PASSWORD=olaabdel88
TASTYTRADE_SANDBOX=false
```

### Optional Configuration
```bash
# CORS Configuration (comma-separated origins)
CORS_ALLOW_ORIGINS=https://your-frontend.railway.app,http://localhost:8501

# Ensemble Service Configuration
ENSEMBLE_RISK_LEVEL=moderate

# Service Configuration (Railway sets PORT automatically)
SERVICE_HOST=0.0.0.0
SERVICE_LOG_LEVEL=info
```

## Deployment Steps

### 1. Connect to Railway

If you haven't already:
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link to your project (or create new)
railway link
```

### 2. Set Environment Variables

In Railway Dashboard:
1. Go to your project
2. Click on "Variables" tab
3. Add the required environment variables listed above
4. Click "Deploy"

Or via CLI:
```bash
railway variables set TASTYTRADE_USERNAME=saleem86@gmail.com
railway variables set TASTYTRADE_PASSWORD=olaabdel88
railway variables set TASTYTRADE_SANDBOX=false
railway variables set CORS_ALLOW_ORIGINS=*
```

### 3. Deploy

```bash
# Push to deploy
git push

# Or use Railway CLI
railway up
```

## Verify Deployment

Once deployed, test the endpoints:

### Health Check
```bash
curl https://your-app.railway.app/health
```

Expected response:
```json
{
  "status": "ok",
  "scanner": "ready",
  "timestamp": "2025-11-07T..."
}
```

### Market Data Endpoint
```bash
curl -X POST https://your-app.railway.app/market-data/options \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "max_options": 10,
    "strike_span": 0.10
  }'
```

### Ensemble Ideas Endpoint
```bash
curl -X POST https://your-app.railway.app/ensemble/ideas \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "option_chains": {},
    "market_data": {},
    "risk_level": "moderate",
    "min_edge": 0.02,
    "min_confidence": 0.4
  }'
```

## Troubleshooting

### Build Fails

**Error: "No start command was found"**
- Solution: Make sure `railway.toml`, `nixpacks.toml`, or `Procfile` exists in root
- Verify the files were committed and pushed to git

**Error: "Module not found"**
- Check that `backend/requirements.txt` includes all dependencies
- Verify the build command in `railway.toml` is correct

### Runtime Errors

**Error: "TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD must be set"**
- Set the environment variables in Railway dashboard
- Restart the deployment

**Error: "Connection timed out" (DXLink)**
- This is the SSL certificate issue
- The fix is already applied in `backend/service/market_data.py:79-82`
- If still failing, check Railway logs for details

**Error: "Scanner not initialized yet"**
- The service is still starting up
- Wait a few seconds and retry
- Check logs: `railway logs`

### View Logs

```bash
# View recent logs
railway logs

# Follow logs in real-time
railway logs --follow
```

## Architecture

```
Railway Container
├── Python 3.10
├── backend/
│   ├── service/
│   │   ├── app.py          # FastAPI app
│   │   ├── market_data.py  # DXLink integration
│   │   └── schemas.py      # Request/response models
│   └── requirements.txt
└── Environment Variables
```

## Performance

- **Cold Start:** ~10-15 seconds
- **Memory:** ~200-300 MB
- **CPU:** Minimal (async I/O)

## Cost Estimates

Railway pricing (as of 2024):
- **Starter Plan:** $5/month (500 hours)
- **Hobby Plan:** $10/month (unlimited hours)

For this lightweight service, Hobby plan is recommended.

## Next Steps

1. **Connect Frontend:** Update `TRADING_BOT_API_URL` in Streamlit to point to Railway URL
2. **Add Monitoring:** Set up Railway metrics dashboard
3. **Configure Alerts:** Set up uptime monitoring
4. **SSL/Custom Domain:** Add custom domain in Railway dashboard

## Support

If you encounter issues:
1. Check Railway logs: `railway logs`
2. Review environment variables
3. Test locally first: `cd backend && uvicorn service.app:app --reload`
4. Check GitHub issues: https://github.com/saleemjadallah/optionsbot/issues

---

*Generated with Claude Code*
*Last Updated: November 7, 2025*
