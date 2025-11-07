#!/bin/bash

# Start Tastytrade Integration (Lightweight)
# ==========================================

echo "╔══════════════════════════════════════════════════════════╗"
echo "║       Tastytrade Authentication System                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "options_trader_env" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found!${NC}"
    echo "Creating virtual environment..."
    python3 -m venv options_trader_env
    source options_trader_env/bin/activate
    echo "Installing dependencies..."
    pip install fastapi uvicorn requests python-dotenv pydantic streamlit pandas plotly streamlit-option-menu
else
    source options_trader_env/bin/activate
fi

# Check if .env file exists
if [ ! -f "backend/.env" ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "Please ensure backend/.env exists with your Tastytrade credentials"
    exit 1
fi

echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

# Create logs directory
mkdir -p logs

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✓${NC} All services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start FastAPI backend (lightweight version)
echo -e "${BLUE}Starting Tastytrade API Backend...${NC}"
cd backend
uvicorn api.endpoints_tastytrade_only:app --reload --port 8000 > ../logs/tastytrade_backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo -n "Waiting for backend to start"
for i in {1..15}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}✓${NC} Backend API running at http://localhost:8000"
        break
    fi
    echo -n "."
    sleep 1
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo ""
    echo -e "${RED}❌ Backend failed to start. Check logs/tastytrade_backend.log${NC}"
    echo ""
    echo "Recent log output:"
    tail -20 logs/tastytrade_backend.log 2>/dev/null || echo "No log file found"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""

# Start Streamlit frontend
echo -e "${BLUE}Starting Streamlit Frontend...${NC}"
cd frontend
streamlit run app.py --server.port 8501 --server.headless true > ../logs/tastytrade_frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo -n "Waiting for frontend to start"
for i in {1..15}; do
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}✓${NC} Frontend dashboard running at http://localhost:8501"
        break
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo -e "${GREEN}🚀 Tastytrade Integration Ready!${NC}"
echo "═══════════════════════════════════════════════════════"
echo ""
echo -e "${BLUE}Available Services:${NC}"
echo "  📊 Dashboard:  http://localhost:8501"
echo "  🔌 API:        http://localhost:8000"
echo "  📖 API Docs:   http://localhost:8000/docs"
echo ""
echo -e "${BLUE}📋 How to Authenticate:${NC}"
echo "  1. Open http://localhost:8501 in your browser"
echo "  2. Click on the 'Tastytrade' tab in the navigation"
echo "  3. Click 'Connect to Tastytrade' button"
echo "  4. Log in with:"
echo "     Username: tastyslim"
echo "     Password: saleemjadallah1986"
echo "  5. Click 'Authorize' to grant access"
echo "  6. Return to dashboard and click 'Refresh Status'"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo "═══════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}Logs:${NC}"
echo "  Backend:  tail -f logs/tastytrade_backend.log"
echo "  Frontend: tail -f logs/tastytrade_frontend.log"
echo ""

# Keep script running
wait
