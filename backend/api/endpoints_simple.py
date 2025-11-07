from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import asyncio
import json
from typing import Dict, List, Optional
from pydantic import BaseModel
import random

app = FastAPI(title="Options Trading Bot API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store WebSocket connections
websocket_connections = set()

# Mock data for demo
bot_state = {
    "is_running": False,
    "start_time": None,
    "positions": [],
    "portfolio_value": 125000,
    "daily_pnl": random.uniform(-500, 1500),
    "available_cash": 23450,
    "buying_power": 180920
}

# Pydantic models
class TradingConfig(BaseModel):
    max_positions: int = 10
    risk_per_trade: float = 0.02
    strategy: str = "iron_condor"

class Position(BaseModel):
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    pnl: float

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Options Trading Bot API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/portfolio/status")
async def get_portfolio_status():
    """Get current portfolio status"""
    return {
        "portfolio_value": bot_state["portfolio_value"] + random.uniform(-100, 100),
        "available_cash": bot_state["available_cash"],
        "buying_power": bot_state["buying_power"],
        "daily_pnl": random.uniform(-500, 1500),
        "total_positions": len(bot_state["positions"]),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/strategies")
async def get_strategies():
    """Get available trading strategies"""
    return {
        "strategies": [
            {
                "name": "Iron Condor",
                "active": True,
                "win_rate": 0.72,
                "avg_return": 0.15,
                "risk_level": "Medium"
            },
            {
                "name": "Put Credit Spread",
                "active": True,
                "win_rate": 0.68,
                "avg_return": 0.12,
                "risk_level": "Low"
            },
            {
                "name": "Straddle",
                "active": False,
                "win_rate": 0.55,
                "avg_return": 0.25,
                "risk_level": "High"
            }
        ]
    }

@app.get("/api/risk/metrics")
async def get_risk_metrics():
    """Get current risk metrics"""
    return {
        "portfolio_var": random.uniform(1000, 3000),
        "portfolio_cvar": random.uniform(1500, 4000),
        "sharpe_ratio": random.uniform(1.2, 2.5),
        "max_drawdown": random.uniform(0.05, 0.15),
        "beta": random.uniform(0.8, 1.2),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/models/performance")
async def get_model_performance():
    """Get ML model performance metrics"""
    return {
        "models": [
            {
                "name": "LSTM Volatility Predictor",
                "accuracy": random.uniform(0.75, 0.85),
                "last_prediction": random.uniform(15, 25),
                "confidence": random.uniform(0.7, 0.95)
            },
            {
                "name": "Random Forest Pricer",
                "accuracy": random.uniform(0.72, 0.82),
                "last_prediction": random.uniform(2.5, 4.5),
                "confidence": random.uniform(0.65, 0.9)
            },
            {
                "name": "XGBoost Direction",
                "accuracy": random.uniform(0.68, 0.78),
                "last_prediction": "BULLISH",
                "confidence": random.uniform(0.6, 0.85)
            }
        ],
        "ensemble_accuracy": random.uniform(0.78, 0.88)
    }

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    # Generate mock positions
    positions = [
        {
            "symbol": "SPY",
            "type": "Iron Condor",
            "expiry": "2024-01-15",
            "strikes": "415/420/450/455",
            "quantity": 5,
            "entry_price": 2.35,
            "current_price": 2.15,
            "pnl": 100,
            "pnl_percent": 8.5
        },
        {
            "symbol": "QQQ",
            "type": "Put Credit Spread",
            "expiry": "2024-01-12",
            "strikes": "360/365",
            "quantity": 10,
            "entry_price": 1.25,
            "current_price": 0.95,
            "pnl": 300,
            "pnl_percent": 24.0
        }
    ]
    return {"positions": positions}

@app.post("/api/bot/start")
async def start_bot():
    """Start the trading bot"""
    bot_state["is_running"] = True
    bot_state["start_time"] = datetime.now().isoformat()
    return {"status": "started", "message": "Trading bot started successfully"}

@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the trading bot"""
    bot_state["is_running"] = False
    return {"status": "stopped", "message": "Trading bot stopped"}

@app.get("/api/bot/status")
async def get_bot_status():
    """Get bot status"""
    return {
        "is_running": bot_state["is_running"],
        "start_time": bot_state["start_time"],
        "uptime_hours": random.uniform(0, 24) if bot_state["is_running"] else 0
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.add(websocket)

    try:
        while True:
            # Send periodic updates
            update = {
                "type": "portfolio_update",
                "data": {
                    "portfolio_value": bot_state["portfolio_value"] + random.uniform(-100, 100),
                    "daily_pnl": random.uniform(-500, 1500),
                    "timestamp": datetime.now().isoformat()
                }
            }
            await websocket.send_json(update)
            await asyncio.sleep(5)  # Send updates every 5 seconds

    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

@app.get("/api/alerts")
async def get_alerts():
    """Get system alerts"""
    return {
        "alerts": [
            {
                "level": "info",
                "message": "Market opened",
                "timestamp": datetime.now().isoformat()
            },
            {
                "level": "warning",
                "message": "High volatility detected in SPY",
                "timestamp": datetime.now().isoformat()
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)