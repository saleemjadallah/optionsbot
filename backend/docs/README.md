# Options Trading Bot

An advanced algorithmic options trading system for Tastyworks that implements sophisticated pricing models, multi-strategy trading, and dynamic risk management.

## Features

- **Advanced Pricing Models**: Black-Scholes, Merton Jump Diffusion, Heston, ML-based pricing
- **Multiple Trading Strategies**: 
  - Volatility strategies (Dispersion, Gamma Scalping, Vol Arbitrage)
  - Neutral strategies (Iron Condor, Delta Neutral)
  - Directional strategies with ML signals
- **Dynamic Risk Management**: Adjustable risk tolerance with regime detection
- **Real-time Integration**: Tastyworks API for live trading
- **Comprehensive Backtesting**: Historical strategy testing with realistic slippage modeling

## Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL (for TimescaleDB)
- Redis (for caching)
- Tastyworks account (paper trading available)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/options-trader.git
cd options-trader
```

2. Create virtual environment:
```bash
python -m venv options_trader_env
source options_trader_env/bin/activate  # Linux/Mac
# or
options_trader_env\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your credentials and settings
```

5. Run the bot:
```bash
python main.py
```

## Project Structure

```
options_trader/
├── config/           # Configuration and parameters
├── data/            # Market data handlers
├── models/          # Pricing models and Greeks
├── strategies/      # Trading strategies
├── execution/       # Order execution and API
├── risk/           # Risk management
├── backtest/       # Backtesting framework
├── monitoring/     # Dashboard and alerts
└── tests/          # Unit and integration tests
```

## Configuration

The bot uses environment variables for configuration. Key settings:

- `TW_SANDBOX`: Set to `True` for paper trading
- `DEFAULT_RISK_LEVEL`: Choose from `low`, `moderate`, `high`
- `MAX_POSITION_SIZE`: Maximum position as % of portfolio
- `MAX_DAILY_LOSS`: Daily loss limit as % of capital

See `.env.example` for all available options.

## Risk Management

The system implements multiple layers of risk control:

- Position-level limits (size, Greeks exposure)
- Portfolio-level limits (concentration, correlation)
- Dynamic position sizing based on market regime
- Circuit breakers for daily/intraday losses
- Real-time margin monitoring

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black .  # Format code
flake8   # Lint code
mypy .   # Type checking
```

## Docker Deployment

```bash
docker-compose up -d
```

Access the dashboard at `http://localhost:8050`

## Safety Notice

⚠️ **WARNING**: This bot trades real money when `TW_SANDBOX=False`. Always:
- Test strategies thoroughly in sandbox mode
- Start with minimal capital
- Monitor continuously during initial deployment
- Set appropriate risk limits
- Never risk more than you can afford to lose

## License

[Your License Here]

## Support

For issues and questions, please open a GitHub issue.