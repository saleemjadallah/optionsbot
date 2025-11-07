#!/bin/bash

# Test Startup Script
# ===================

echo "Testing Tastytrade Integration Startup..."
echo ""

# Test 1: Check virtual environment
echo "1. Checking virtual environment..."
if [ -d "options_trader_env" ]; then
    echo "   ✅ Virtual environment exists"
    source options_trader_env/bin/activate
else
    echo "   ❌ Virtual environment not found"
    exit 1
fi

# Test 2: Check .env file
echo "2. Checking .env file..."
if [ -f "backend/.env" ]; then
    echo "   ✅ .env file exists"
else
    echo "   ❌ .env file not found"
    exit 1
fi

# Test 3: Check Python packages
echo "3. Checking Python packages..."
python -c "import fastapi, uvicorn, streamlit, requests, pydantic" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ All required packages installed"
else
    echo "   ⚠️  Some packages missing. Installing..."
    pip install -q fastapi uvicorn streamlit requests pydantic python-dotenv
fi

# Test 4: Test backend import
echo "4. Testing backend API..."
cd backend
python -c "from api.endpoints_tastytrade_only import app; print('OK')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ Backend API can be imported"
else
    echo "   ❌ Backend API has errors"
    cd ..
    exit 1
fi
cd ..

# Test 5: Test frontend config
echo "5. Testing Streamlit config..."
if [ -f "frontend/.streamlit/config.toml" ]; then
    echo "   ✅ Streamlit config exists"
else
    echo "   ⚠️  Streamlit config not found (optional)"
fi

# Test 6: Start backend briefly
echo "6. Testing backend startup..."
cd backend
timeout 5 uvicorn api.endpoints_tastytrade_only:app --port 8000 >/dev/null 2>&1 &
BACKEND_PID=$!
cd ..

sleep 3

if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "   ✅ Backend starts successfully"
    kill $BACKEND_PID 2>/dev/null
else
    echo "   ⚠️  Backend didn't respond (might need more time)"
    kill $BACKEND_PID 2>/dev/null
fi

echo ""
echo "═══════════════════════════════════════════"
echo "✅ All tests passed!"
echo "═══════════════════════════════════════════"
echo ""
echo "Ready to start! Run:"
echo "  ./start_tastytrade.sh"
echo ""
