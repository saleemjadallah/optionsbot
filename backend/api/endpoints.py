from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import asyncio
import json
from typing import Dict, List, Optional
import os
import sys

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config
from models.ensemble_model import EnsemblePricer
from strategies.strategy_manager import StrategyManager
from risk.portfolio_manager import PortfolioManager
from execution.order_executor import OrderExecutor
from monitoring.logger import OptionsLogger
from api.term_definitions_endpoint import router as term_definitions_router
from api.tastytrade_endpoints import router as tastytrade_router

app = FastAPI(title="Options Trading Bot API")

# Include routers
app.include_router(term_definitions_router)
app.include_router(tastytrade_router)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://dashboard:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = Config()
logger = OptionsLogger()
ensemble_pricer = None
strategy_manager = None
portfolio_manager = None
order_executor = None

# Store WebSocket connections
websocket_connections = set()

# Trading bot state
bot_state = {
    "is_running": False,
    "start_time": None,
    "positions": [],
    "portfolio_value": 125000,
    "daily_pnl": 0,
    "available_cash": 23450,
    "buying_power": 180920
}

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global ensemble_pricer, strategy_manager, portfolio_manager, order_executor

    try:
        # Initialize components
        ensemble_pricer = EnsemblePricer(config)
        strategy_manager = StrategyManager(config)
        portfolio_manager = PortfolioManager(config)
        order_executor = OrderExecutor(config)

        logger.info("API components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API components: {e}")

@app.get("/api/portfolio/status")
async def get_portfolio_status():
    """Get current portfolio status"""
    try:
        if portfolio_manager:
            # Get real portfolio data
            positions = portfolio_manager.get_positions()
            metrics = portfolio_manager.calculate_portfolio_metrics()

            return {
                "portfolio_value": metrics.get("total_value", bot_state["portfolio_value"]),
                "daily_pnl": metrics.get("daily_pnl", bot_state["daily_pnl"]),
                "positions_count": len(positions),
                "available_cash": metrics.get("available_cash", bot_state["available_cash"]),
                "buying_power": metrics.get("buying_power", bot_state["buying_power"])
            }
        else:
            # Return mock data if components not initialized
            return bot_state
    except Exception as e:
        logger.error(f"Error getting portfolio status: {e}")
        return bot_state

@app.get("/api/positions")
async def get_positions():
    """Get all current positions"""
    try:
        if portfolio_manager:
            positions = portfolio_manager.get_positions()
            return [
                {
                    "symbol": pos.get("symbol", ""),
                    "strategy": pos.get("strategy", "Unknown"),
                    "type": pos.get("option_type", ""),
                    "quantity": pos.get("quantity", 0),
                    "entry_price": pos.get("entry_price", 0),
                    "current_price": pos.get("current_price", 0),
                    "pnl": pos.get("pnl", 0),
                    "delta": pos.get("delta", 0),
                    "gamma": pos.get("gamma", 0),
                    "theta": pos.get("theta", 0),
                    "vega": pos.get("vega", 0),
                    "expiration": pos.get("expiration", "")
                }
                for pos in positions
            ]
        else:
            # Return mock data
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
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return []

@app.get("/api/strategies")
async def get_strategies():
    """Get strategy status and performance"""
    try:
        if strategy_manager:
            strategies = strategy_manager.get_all_strategies()
            return [
                {
                    "name": strategy.name,
                    "status": "Active" if strategy.is_active else "Inactive",
                    "positions": len(strategy.positions),
                    "daily_pnl": strategy.daily_pnl,
                    "success_rate": strategy.success_rate,
                    "capital_allocated": strategy.allocated_capital
                }
                for strategy in strategies
            ]
        else:
            # Return mock data
            return [
                {
                    "name": "Model Ensemble",
                    "status": "Active",
                    "positions": 8,
                    "daily_pnl": 850,
                    "success_rate": 72.5,
                    "capital_allocated": 50000
                },
                {
                    "name": "Volatility Arbitrage",
                    "status": "Active",
                    "positions": 5,
                    "daily_pnl": 420,
                    "success_rate": 68.3,
                    "capital_allocated": 30000
                },
                {
                    "name": "Delta Neutral",
                    "status": "Paused",
                    "positions": 3,
                    "daily_pnl": -120,
                    "success_rate": 65.0,
                    "capital_allocated": 20000
                }
            ]
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        return []

@app.get("/api/models/performance")
async def get_model_performance():
    """Get model ensemble performance"""
    try:
        if ensemble_pricer:
            performance = ensemble_pricer.get_performance_metrics()
            predictions = ensemble_pricer.get_recent_predictions()

            return {
                "weights": performance.get("weights", {}),
                "accuracy": performance.get("accuracy", {}),
                "predictions": predictions
            }
        else:
            # Return mock data
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
                    },
                    {
                        "symbol": "TSLA 250P 04/19",
                        "market_price": 5.10,
                        "consensus_price": 4.85,
                        "edge": -4.9
                    }
                ]
            }
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        return {}

@app.get("/api/risk/metrics")
async def get_risk_metrics():
    """Get portfolio risk metrics"""
    try:
        if portfolio_manager:
            risk_metrics = portfolio_manager.calculate_risk_metrics()
            return risk_metrics
        else:
            # Return mock data
            return {
                "var_95": 2450,
                "expected_shortfall": 3820,
                "max_drawdown": 8.2,
                "portfolio_delta": 0.15,
                "portfolio_gamma": 0.04,
                "portfolio_theta": -180,
                "portfolio_vega": 150,
                "sharpe_ratio": 1.45,
                "kelly_fraction": 0.23
            }
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        return {}

@app.post("/api/trading/start")
async def start_trading():
    """Start the trading bot"""
    global bot_state

    try:
        if bot_state["is_running"]:
            return {"status": "already_running", "message": "Trading bot is already running"}

        bot_state["is_running"] = True
        bot_state["start_time"] = datetime.now().isoformat()

        # Start background task for trading
        asyncio.create_task(trading_loop())

        logger.info("Trading bot started")
        return {"status": "started", "message": "Trading bot started successfully"}
    except Exception as e:
        logger.error(f"Failed to start trading bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trading/stop")
async def stop_trading():
    """Stop the trading bot"""
    global bot_state

    try:
        if not bot_state["is_running"]:
            return {"status": "not_running", "message": "Trading bot is not running"}

        bot_state["is_running"] = False
        bot_state["start_time"] = None

        logger.info("Trading bot stopped")
        return {"status": "stopped", "message": "Trading bot stopped"}
    except Exception as e:
        logger.error(f"Failed to stop trading bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_name}/pause")
async def pause_strategy(strategy_name: str):
    """Pause a specific strategy"""
    try:
        if strategy_manager:
            success = strategy_manager.pause_strategy(strategy_name)
            if success:
                return {"status": "paused", "strategy": strategy_name}
            else:
                raise HTTPException(status_code=404, detail=f"Strategy {strategy_name} not found")
        else:
            return {"status": "paused", "strategy": strategy_name}
    except Exception as e:
        logger.error(f"Failed to pause strategy {strategy_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_name}/resume")
async def resume_strategy(strategy_name: str):
    """Resume a specific strategy"""
    try:
        if strategy_manager:
            success = strategy_manager.resume_strategy(strategy_name)
            if success:
                return {"status": "resumed", "strategy": strategy_name}
            else:
                raise HTTPException(status_code=404, detail=f"Strategy {strategy_name} not found")
        else:
            return {"status": "resumed", "strategy": strategy_name}
    except Exception as e:
        logger.error(f"Failed to resume strategy {strategy_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    websocket_connections.add(websocket)

    try:
        while True:
            # Send real-time updates to frontend
            update = await get_portfolio_update()
            await websocket.send_json(update)
            await asyncio.sleep(1)  # Update every second

    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

async def get_portfolio_update():
    """Get current portfolio update for WebSocket"""
    try:
        portfolio_status = await get_portfolio_status()
        return {
            "type": "portfolio_update",
            "data": {
                "portfolio_value": portfolio_status["portfolio_value"],
                "daily_pnl": portfolio_status["daily_pnl"],
                "positions_count": portfolio_status["positions_count"],
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting portfolio update: {e}")
        return {
            "type": "error",
            "data": {"message": str(e)}
        }

async def trading_loop():
    """Main trading loop that runs in background"""
    global bot_state

    while bot_state["is_running"]:
        try:
            # Execute trading logic
            if strategy_manager and ensemble_pricer:
                # Get market data
                # Analyze opportunities
                # Execute trades
                pass

            # Broadcast updates to all WebSocket connections
            update = await get_portfolio_update()
            for websocket in websocket_connections:
                try:
                    await websocket.send_json(update)
                except:
                    pass

            await asyncio.sleep(5)  # Run every 5 seconds

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await asyncio.sleep(10)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bot_running": bot_state["is_running"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)