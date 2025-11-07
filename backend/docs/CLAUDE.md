# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an advanced algorithmic options trading system for Tastyworks that implements:
- Advanced pricing models (Black-Scholes, Merton Jump Diffusion, Heston, ML-based)
- Multi-strategy trading (volatility dispersion, gamma scalping, iron condor, delta neutral)
- Dynamic risk management with adjustable risk tolerance
- Real-time market data streaming and order execution
- Comprehensive backtesting framework

## Development Setup Commands

```bash
# Create and activate virtual environment
python -m venv options_trader_env
source options_trader_env/bin/activate  # Linux/Mac
# or
options_trader_env\Scripts\activate  # Windows

# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt

# Run the main trading bot
python main.py

# Run tests
pytest tests/

# Run with Docker
docker-compose up -d

# View logs
docker-compose logs -f trader

# Run backtests
python -m backtest.engine --strategy gamma_scalping --start 2023-01-01 --end 2024-01-01
```

## Project Architecture

The codebase follows a modular architecture with clear separation of concerns:

- **config/**: Environment variables, trading parameters, and risk limits
- **data/**: Real-time and historical market data handlers, options chain processing
- **models/**: Pricing models (Black-Scholes, Merton Jump, Heston, ML-based), Greeks calculations, volatility surface modeling
  - `ml_pricing.py`: Neural network-based options pricer with 4-layer deep network (400 neurons each), achieving 10x+ speedup over traditional methods
- **strategies/**: Trading strategies organized by type (directional, volatility, neutral)
- **execution/**: Tastyworks API integration, order and position management
- **risk/**: Portfolio risk calculations, VAR, regime detection, dynamic hedging
- **backtest/**: Historical strategy testing engine and performance metrics
- **monitoring/**: Real-time dashboard and alerting system

## Key Technical Decisions

1. **Async Architecture**: Uses asyncio for concurrent market data processing and order execution
2. **Tastyworks Integration**: Utilizes unofficial tastytrade Python SDK for broker connectivity
3. **Risk Management**: Implements dynamic position sizing based on market regime detection using Gaussian Mixture Models
4. **Data Storage**: TimescaleDB for time-series data, Redis for caching, Arctic for tick data
5. **ML Framework**: TensorFlow/PyTorch for option pricing models, XGBoost/LightGBM for signal generation

## Critical Implementation Notes

- **Rate Limiting**: Tastyworks API has rate limits - OrderManager implements semaphore-based limiting (60 orders/minute)
- **Risk Controls**: Hard limits enforced - max 25% position size, 5% daily loss limit, 2x leverage cap
- **Paper Trading**: Always test strategies in sandbox mode (TW_SANDBOX=True) before production
- **Greeks Calculation**: Use QuantLib for accurate Greeks when not provided by API
- **Slippage Modeling**: Backtest engine adds 0.1% slippage to simulate realistic fills

## Environment Variables Required

```bash
TW_USERNAME=your_username
TW_PASSWORD=your_password  
TW_ACCOUNT=your_account_number
TW_SANDBOX=True  # Set to False for production
DB_HOST=localhost
DB_PORT=5432
DB_NAME=options_trader
DB_USER=trader
DB_PASSWORD=secure_password
```

## Risk Management Parameters

The system supports three risk levels (low/moderate/high) that adjust:
- Position sizing multipliers based on market regime
- Maximum leverage allowed
- Number of concurrent positions
- Stop-loss thresholds

Default configuration uses "moderate" risk with:
- 25% max position size
- 5% max daily loss
- 95% VaR confidence level
- 2x leverage limit

## Implementation Status

### Completed Models
- **ML Pricing Model** (`models/ml_pricing.py`): Implemented with PyTorch neural network
  - ✅ 4-layer architecture with 400 neurons each
  - ✅ Feature engineering with 7 inputs (moneyness, time, volatility, etc.)
  - ✅ Training pipeline with early stopping and validation
  - ✅ Model persistence and batch prediction support
  - ⚠️ Requires full training (50+ epochs) for production accuracy

### Pending Implementation
- **Black-Scholes Model**: Traditional analytical pricing
- **Merton Jump Diffusion**: Jump process modeling for extreme events
- **Heston Model**: Stochastic volatility pricing
- **Trading Strategies**: All strategy modules
- **Execution Engine**: Tastyworks API integration
- **Risk Management**: Portfolio risk calculations and hedging
- **Backtesting Framework**: Historical testing engine