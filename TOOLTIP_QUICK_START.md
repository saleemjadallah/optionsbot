# Quick Start Guide - Trading Term Tooltips

## Installation

### 1. Install Backend Dependencies

```bash
cd backend
pip install openai>=1.3.0 fastapi>=0.104.0 uvicorn>=0.24.0 pydantic>=2.0.0
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

### 2. Verify OpenAI API Key

Make sure your `.env` file in the backend directory has your OpenAI API key:

```bash
cd backend
cat .env | grep OPENAI_API_KEY
```

Should show:
```
OPENAI_API_KEY=sk-proj-tsiBVOK...
```

## Running the System

### Start Backend

```bash
cd backend
python -m uvicorn api.endpoints:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Start Frontend

In a new terminal:

```bash
cd frontend
streamlit run app.py
```

You should see:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

## See It In Action

1. Open your browser to `http://localhost:8501`

2. Navigate through the pages and look for purple **❓** icons next to trading terms:
   - **Dashboard** → Greeks (Delta, Gamma, Theta, Vega)
   - **Portfolio** → Detailed Positions (Greeks Quick Reference)
   - **Risk** → VaR, Max Drawdown, Leverage
   - **Analytics** → Sharpe Ratio, Win Rate, Profit Factor, etc.

3. **Hover over any ❓ icon** to see the definition!

## Example Visual

```
Portfolio Delta ❓     <- Hover here
  ↓
┌─────────────────────────────────────────────┐
│ Delta                                        │
│                                             │
│ Delta measures the rate of change of an    │
│ option's price relative to changes in the  │
│ underlying asset's price. It ranges from   │
│ 0 to 1 for calls and -1 to 0 for puts.    │
│                                             │
│ Delta is crucial for understanding         │
│ directional risk...                        │
└─────────────────────────────────────────────┘
```

## Test Without Backend

The system works even without the backend running! It will use fallback definitions for common terms.

```bash
# Test the feature (works without backend)
python test_tooltip_feature.py
```

You should see:
```
============================================================
✅ ALL TESTS PASSED!
============================================================
```

## Verify Backend API

Test the API directly:

```bash
# Health check
curl http://localhost:8000/api/health

# Get a definition
curl http://localhost:8000/api/term-definition/Delta

# Expected response:
# {
#   "term": "Delta",
#   "definition": "Delta measures the rate of change...",
#   "generated_at": "2025-11-06T...",
#   "cached": false
# }
```

## Troubleshooting

### Issue: "Connection refused" on port 8000

**Solution**: Start the backend server first:
```bash
cd backend
python -m uvicorn api.endpoints:app --reload
```

### Issue: Tooltips not showing

**Solution**:
1. Check browser console for errors (F12)
2. Make sure you're hovering over the ❓ icon, not just near it
3. Clear browser cache (Ctrl+Shift+R)

### Issue: "OpenAI API key not configured"

**Solution**: Add your key to `backend/.env`:
```bash
echo "OPENAI_API_KEY=your-key-here" >> backend/.env
```

### Issue: Import errors

**Solution**: Install missing packages:
```bash
pip install openai fastapi uvicorn pydantic requests
```

## Cost Estimates

- **First 100 terms**: ~$0.015 (one-time cost)
- **With 30-day caching**: Cost spread over month
- **Fallback terms** (30+): $0 forever
- **Annual estimate**: <$0.20 for typical usage

## What's Next?

The system is ready to use! Here are some ideas:

1. **Add more tooltips** to other pages
2. **Customize definitions** in `frontend/utils/term_definitions.py`
3. **Add tooltips to strategy names** (Iron Condor, etc.)
4. **Share feedback** on definition quality

## Support

For detailed documentation, see:
- `TOOLTIP_IMPLEMENTATION_SUMMARY.md` - Complete implementation details
- `frontend/TOOLTIP_FEATURE_README.md` - Technical documentation
- `test_tooltip_feature.py` - Test suite and examples

Enjoy your new educational tooltips! 🎉
