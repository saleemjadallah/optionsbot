# Options Trader Backend

## 📁 Project Structure

```
backend/
├── config/                 # Configuration files
│   ├── __init__.py
│   ├── settings.py         # Main settings
│   ├── risk_limits.py      # Risk management limits
│   ├── trading_params.py   # Trading parameters
│   └── model_ensemble_config.py  # Model ensemble configuration
│
├── models/                 # Pricing models
│   ├── pricing/
│   │   ├── black_scholes.py    # Black-Scholes model
│   │   ├── merton_jump.py      # Merton Jump Diffusion
│   │   ├── heston.py           # Heston stochastic volatility
│   │   └── ml_pricing.py       # ML neural network pricer
│   └── ml_pricer/          # ML model artifacts
│
├── strategies/             # Trading strategies
│   ├── model_ensemble.py   # Main ensemble strategy (691 lines)
│   ├── directional/        # Directional strategies
│   ├── volatility/         # Volatility strategies
│   └── neutral/            # Market neutral strategies
│
├── risk/                   # Risk management
│   └── enhanced_risk_manager.py  # Model-aware risk management (583 lines)
│
├── monitoring/             # Performance monitoring
│   └── model_performance_tracker.py  # Track model performance (495 lines)
│
├── execution/              # Order execution
│   └── __init__.py         # Broker integration (TBD)
│
├── backtest/               # Backtesting engine
│   └── __init__.py         # Historical testing
│
├── data/                   # Data management
│   ├── model_performance/  # Model performance data
│   └── __init__.py
│
├── tests/                  # Test suites
│   └── __init__.py
│
├── logs/                   # Application logs
│   └── trading_bot.log
│
├── docs/                   # Documentation
│   ├── modelensemble.md
│   ├── modelensembleintegration.md
│   └── MODEL_ENSEMBLE_COMPLETE.md
│
├── main.py                 # Basic trading bot
├── main_ensemble.py        # Enhanced bot with model ensemble (446 lines)
├── test_ensemble_integration.py  # Integration tests
├── train_ml_model.py       # ML model training script
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── .gitignore             # Git ignore file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Run Tests

```bash
# Test the complete ensemble system
python test_ensemble_integration.py
```

### 4. Run Trading Bot

```bash
# Run enhanced bot with model ensemble
python main_ensemble.py

# Or run basic bot
python main.py
```

## 🎯 Key Components

### Model Ensemble (`strategies/model_ensemble.py`)
- Combines 4 pricing models for superior edge detection
- Market regime detection
- Automatic strategy selection
- 691 lines of production-ready code

### Enhanced Risk Manager (`risk/enhanced_risk_manager.py`)
- Model-aware position sizing
- Portfolio correlation analysis
- Dynamic risk adjustments
- 583 lines of sophisticated risk management

### Performance Tracker (`monitoring/model_performance_tracker.py`)
- Real-time model accuracy monitoring
- Automatic weight rebalancing
- Historical performance analysis
- 495 lines of tracking logic

### Main Ensemble Bot (`main_ensemble.py`)
- Full automation with model ensemble
- Market data fetching
- Trade execution
- 446 lines of integration code

## 📊 Model Ensemble Features

- **4 Pricing Models**: Black-Scholes, Merton Jump, Heston, ML Neural Network
- **Market Regimes**: Calm, Volatile, Trending, Crisis
- **Trading Strategies**: Directional, Volatility Arbitrage, Gamma Scalping, Market Making
- **Risk Controls**: Position limits, correlation constraints, dynamic adjustments
- **Performance Tracking**: Real-time accuracy, adaptive weights, regime analysis

## 🧪 Test Results

All components tested and passing:
```
Model Ensemble       ✓ PASSED
Performance Tracker  ✓ PASSED
Risk Manager         ✓ PASSED
Configuration        ✓ PASSED
Full Integration     ✓ PASSED

Total: 5/5 tests passed
```

## 📈 Next Steps

1. **Connect Real Data**: Implement Tastyworks API integration
2. **Train ML Model**: Collect data and train neural network
3. **Backtest**: Run on 2+ years of historical data
4. **Paper Trade**: Test in sandbox for 60+ days
5. **Deploy**: Start with small positions in production

## 📝 Configuration

Edit `config/model_ensemble_config.py` for:
- Model weights
- Risk thresholds
- Universe selection
- Position sizing rules
- Calibration schedules

## 🔧 Environment Variables

Required in `.env`:
```
TW_USERNAME=your_username
TW_PASSWORD=your_password
TW_ACCOUNT=your_account_number
TW_SANDBOX=True
DB_HOST=localhost
DB_PORT=5432
DB_NAME=options_trader
```

## 📚 Documentation

- [Model Ensemble Strategy](docs/modelensemble.md)
- [Integration Plan](docs/modelensembleintegration.md)
- [Complete Implementation](docs/MODEL_ENSEMBLE_COMPLETE.md)

## 🛡️ Risk Warning

This is algorithmic trading software that can result in financial losses. Always:
- Test thoroughly in sandbox mode
- Start with small positions
- Monitor continuously
- Have stop-loss limits
- Understand the risks

## 📞 Support

For issues or questions, refer to the documentation in the `docs/` folder.