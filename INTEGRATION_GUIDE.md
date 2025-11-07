# Options Trading Bot - Frontend Integration Guide
# ================================================

This guide explains how to integrate the Streamlit frontend with the main options trading bot backend.

## 🔗 Architecture Overview

```
┌─────────────────┐    HTTP/WebSocket    ┌──────────────────────┐
│   Streamlit     │ ◄─────────────────► │   Trading Bot API    │
│   Frontend      │                      │   (FastAPI)          │
│   (Port 8501)   │                      │   (Port 8000)        │
└─────────────────┘                      └──────────────────────┘
         │                                          │
         │                                          │
         ▼                                          ▼
┌─────────────────┐                      ┌──────────────────────┐
│     Redis       │                      │    PostgreSQL +      │
│   (Cache/       │                      │    TimescaleDB       │
│   Sessions)     │                      │   (Market Data)      │
└─────────────────┘                      └──────────────────────┘
```

## 📡 Backend API Endpoints

The frontend expects these API endpoints from the trading bot:

### Core Endpoints

```python
# File: backend/api/endpoints.py

from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Options Trading Bot API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://dashboard:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/portfolio/status")
async def get_portfolio_status():
    """Get current portfolio status"""
    return {
        "portfolio_value": 125000,
        "daily_pnl": 850,
        "positions_count": 18,
        "available_cash": 23450,
        "buying_power": 180920
    }

@app.get("/api/positions")
async def get_positions():
    """Get all current positions"""
    return [
        {
            "symbol": "AAPL",
            "strategy": "Model Ensemble",
            "type": "Call Spread",
            "quantity": 10,
            "entry_price": 2.50,
            "current_price": 3.20,
            "pnl": 700,
            "delta": 0.35,
            "gamma": 0.02,
            "theta": -12.5,
            "vega": 45.2,
            "expiration": "2024-04-19"
        }
    ]

@app.get("/api/strategies")
async def get_strategies():
    """Get strategy status and performance"""
    return [
        {
            "name": "Model Ensemble",
            "status": "Active",
            "positions": 8,
            "daily_pnl": 850,
            "success_rate": 72.5,
            "capital_allocated": 50000
        }
    ]

@app.get("/api/models/performance")
async def get_model_performance():
    """Get model ensemble performance"""
    return {
        "weights": {
            "black_scholes": 0.20,
            "merton_jump": 0.25,
            "heston": 0.30,
            "ml_neural": 0.25
        },
        "accuracy": {
            "black_scholes": 0.72,
            "merton_jump": 0.84,
            "heston": 0.78,
            "ml_neural": 0.89
        },
        "predictions": [
            {
                "symbol": "AAPL 175C 04/19",
                "market_price": 3.20,
                "consensus_price": 3.45,
                "edge": 7.8
            }
        ]
    }

@app.get("/api/risk/metrics")
async def get_risk_metrics():
    """Get portfolio risk metrics"""
    return {
        "var_95": 2450,
        "expected_shortfall": 3820,
        "max_drawdown": 8.2,
        "portfolio_delta": 0.15,
        "portfolio_gamma": 0.04,
        "portfolio_theta": -180,
        "portfolio_vega": 150
    }

@app.post("/api/trading/start")
async def start_trading():
    """Start the trading bot"""
    # Start trading logic
    return {"status": "started", "message": "Trading bot started successfully"}

@app.post("/api/trading/stop")
async def stop_trading():
    """Stop the trading bot"""
    # Stop trading logic
    return {"status": "stopped", "message": "Trading bot stopped"}

@app.post("/api/strategies/{strategy_name}/pause")
async def pause_strategy(strategy_name: str):
    """Pause a specific strategy"""
    return {"status": "paused", "strategy": strategy_name}

@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send real-time updates to frontend
            update = {
                "type": "portfolio_update",
                "data": {
                    "portfolio_value": get_current_portfolio_value(),
                    "daily_pnl": get_current_pnl(),
                    "timestamp": datetime.now().isoformat()
                }
            }
            await websocket.send_json(update)
            await asyncio.sleep(1)  # Update every second
            
    except WebSocketDisconnect:
        pass
```

## 🔄 Frontend Backend Connection

### API Client Integration

```python
# File: frontend/utils/api_client.py

import requests
import websocket
import json
import streamlit as st
from typing import Dict, List, Optional
import asyncio

class TradingBotAPI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_portfolio_status(self) -> Dict:
        """Fetch portfolio status from backend"""
        try:
            response = self.session.get(f"{self.base_url}/api/portfolio/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch portfolio status: {e}")
            return self._get_mock_portfolio_status()
    
    def get_positions(self) -> List[Dict]:
        """Fetch current positions"""
        try:
            response = self.session.get(f"{self.base_url}/api/positions")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch positions: {e}")
            return self._get_mock_positions()
    
    def start_trading(self) -> bool:
        """Start the trading bot"""
        try:
            response = self.session.post(f"{self.base_url}/api/trading/start")
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"Failed to start trading: {e}")
            return False
    
    def stop_trading(self) -> bool:
        """Stop the trading bot"""
        try:
            response = self.session.post(f"{self.base_url}/api/trading/stop")
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"Failed to stop trading: {e}")
            return False
    
    def _get_mock_portfolio_status(self) -> Dict:
        """Fallback mock data"""
        return {
            "portfolio_value": 125000,
            "daily_pnl": 850,
            "positions_count": 18,
            "available_cash": 23450
        }
```

### WebSocket Integration

```python
# File: frontend/utils/websocket_client.py

import streamlit as st
import asyncio
import websockets
import json
from typing import Callable

class RealTimeUpdates:
    def __init__(self, websocket_url: str = "ws://localhost:8000/ws/updates"):
        self.websocket_url = websocket_url
        self.connected = False
        
    async def connect_and_listen(self, callback: Callable):
        """Connect to WebSocket and listen for updates"""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                self.connected = True
                st.success("Connected to trading bot")
                
                async for message in websocket:
                    data = json.loads(message)
                    callback(data)
                    
        except Exception as e:
            st.error(f"WebSocket connection failed: {e}")
            self.connected = False

    def update_portfolio(self, data: dict):
        """Update portfolio data in session state"""
        if 'portfolio_updates' not in st.session_state:
            st.session_state.portfolio_updates = []
        
        st.session_state.portfolio_updates.append(data)
        
        # Update current values
        if data['type'] == 'portfolio_update':
            st.session_state.portfolio_value = data['data']['portfolio_value']
            st.session_state.daily_pnl = data['data']['daily_pnl']
```

## 🚀 Deployment Integration

### Docker Compose with Backend

```yaml
# File: docker-compose.full.yml

version: '3.8'

services:
  # Trading Bot Backend
  trading-bot:
    build:
      context: ../backend
      dockerfile: Dockerfile
    container_name: options_trading_bot
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://trader:${DB_PASSWORD}@postgres:5432/options_trader
      - REDIS_URL=redis://redis:6379
      - TASTYWORKS_USERNAME=${TW_USERNAME}
      - TASTYWORKS_PASSWORD=${TW_PASSWORD}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
    networks:
      - trading_network
    restart: unless-stopped

  # Frontend Dashboard
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: options_trading_dashboard
    ports:
      - "8501:8501"
    environment:
      - TRADING_BOT_API_URL=http://trading-bot:8000
      - STREAMLIT_SERVER_HEADLESS=true
    depends_on:
      - trading-bot
      - redis
    networks:
      - trading_network
    restart: unless-stopped

  # Database
  postgres:
    image: timescale/timescaledb:latest-pg14
    container_name: options_trading_db
    environment:
      POSTGRES_DB: options_trader
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - trading_network

  # Redis
  redis:
    image: redis:7-alpine
    container_name: options_trading_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - trading_network

  # Nginx (Production)
  nginx:
    image: nginx:alpine
    container_name: options_trading_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - dashboard
      - trading-bot
    networks:
      - trading_network
    profiles:
      - production

volumes:
  postgres_data:
  redis_data:

networks:
  trading_network:
    driver: bridge
```

## 📊 Real-Time Data Flow

### Implementation in Frontend

```python
# File: frontend/app.py (modified sections)

import asyncio
from utils.api_client import TradingBotAPI
from utils.websocket_client import RealTimeUpdates

class TradingDashboard:
    def __init__(self):
        self.api_client = TradingBotAPI()
        self.websocket_client = RealTimeUpdates()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize with real data from backend"""
        if 'portfolio_data' not in st.session_state:
            # Fetch initial data from backend
            portfolio_status = self.api_client.get_portfolio_status()
            st.session_state.update(portfolio_status)
            
    def render_real_time_updates(self):
        """Handle real-time updates"""
        if st.session_state.get('auto_refresh', True):
            # Setup WebSocket connection
            if not self.websocket_client.connected:
                asyncio.run(
                    self.websocket_client.connect_and_listen(
                        self.websocket_client.update_portfolio
                    )
                )
            
            # Auto-refresh every 30 seconds
            time.sleep(30)
            st.experimental_rerun()
```

## 🔧 Configuration Management

### Environment Variables

```bash
# File: .env

# Backend API
TRADING_BOT_API_URL=http://localhost:8000
TRADING_BOT_WS_URL=ws://localhost:8000/ws/updates

# Database
DATABASE_URL=postgresql://trader:password@localhost:5432/options_trader

# Cache
REDIS_URL=redis://localhost:6379

# Tastyworks
TW_USERNAME=your_username
TW_PASSWORD=your_password
TW_ACCOUNT=your_account

# Security
SECRET_KEY=your_secret_key
JWT_SECRET=your_jwt_secret

# Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO
```

## 🧪 Testing Integration

### API Testing

```python
# File: tests/test_api_integration.py

import pytest
import requests
from unittest.mock import Mock, patch
from frontend.utils.api_client import TradingBotAPI

def test_api_client_connection():
    """Test API client can connect to backend"""
    client = TradingBotAPI("http://localhost:8000")
    
    # Mock successful response
    with patch.object(client.session, 'get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {"portfolio_value": 125000}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = client.get_portfolio_status()
        assert result["portfolio_value"] == 125000

def test_websocket_connection():
    """Test WebSocket connection"""
    # Implementation for WebSocket testing
    pass

def test_trading_controls():
    """Test trading start/stop functionality"""
    client = TradingBotAPI("http://localhost:8000")
    
    with patch.object(client.session, 'post') as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = client.start_trading()
        assert result == True
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Backend API is running and accessible
- [ ] Database is properly configured
- [ ] Redis is running for session management
- [ ] WebSocket endpoints are functional
- [ ] API authentication is working
- [ ] All environment variables are set

### Integration Testing
- [ ] Frontend can connect to backend API
- [ ] Real-time updates are working via WebSocket
- [ ] Trading controls (start/stop) function properly
- [ ] Portfolio data loads correctly
- [ ] Strategy management works end-to-end
- [ ] Risk management integration is functional

### Production Deployment
- [ ] SSL certificates are configured
- [ ] Load balancing is set up (if needed)
- [ ] Monitoring and alerting are active
- [ ] Backup and recovery procedures are tested
- [ ] Security audit is complete

## 🚀 Quick Start Integration

1. **Start Backend**
```bash
cd backend/
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

2. **Start Frontend**
```bash
cd frontend/
./launch.sh
```

3. **Test Integration**
```bash
# Test API connectivity
curl http://localhost:8000/api/portfolio/status

# Test frontend
open http://localhost:8501
```

4. **Full Docker Deployment**
```bash
docker-compose -f docker-compose.full.yml up -d
```

The frontend is now fully integrated with the backend options trading system!
