# Advanced Options Trading Bot - Project Instructions

## Project Overview
Build a sophisticated algorithmic options trading system for Tastyworks that implements advanced pricing models, multi-strategy trading, and dynamic risk management with adjustable risk tolerance parameters.

---

## Phase 1: Development Environment Setup (Week 1)

### 1.1 Core Dependencies Installation

```bash
# Create virtual environment
python -m venv options_trader_env
source options_trader_env/bin/activate  # Linux/Mac
# or
options_trader_env\Scripts\activate  # Windows

# Core libraries
pip install numpy pandas scipy scikit-learn
pip install asyncio aiohttp websockets
pip install redis celery

# Options-specific libraries
pip install QuantLib-Python py_vollib
pip install tastytrade  # Unofficial Tastyworks SDK

# Data and visualization
pip install plotly dash streamlit
pip install arctic  # MongoDB-based tick database

# Machine learning
pip install tensorflow torch xgboost lightgbm
pip install optuna  # Hyperparameter optimization

# Database
pip install sqlalchemy psycopg2-binary
pip install timescaledb-toolkit
```

### 1.2 Project Structure

```
options_trader/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Environment variables
│   ├── trading_params.py    # Strategy parameters
│   └── risk_limits.py       # Risk management configs
├── data/
│   ├── __init__.py
│   ├── market_data.py       # Real-time data handlers
│   ├── historical.py        # Historical data management
│   └── options_chain.py     # Options chain processing
├── models/
│   ├── __init__.py
│   ├── pricing/
│   │   ├── black_scholes.py
│   │   ├── merton_jump.py
│   │   ├── heston.py
│   │   └── ml_pricing.py
│   ├── greeks.py            # Greeks calculations
│   └── volatility/
│       ├── surface.py       # Vol surface modeling
│       └── smile.py         # Volatility smile
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── directional/
│   │   ├── momentum.py
│   │   └── ml_signals.py
│   ├── volatility/
│   │   ├── dispersion.py
│   │   ├── vol_arb.py
│   │   └── gamma_scalping.py
│   └── neutral/
│       ├── iron_condor.py
│       └── delta_neutral.py
├── execution/
│   ├── __init__.py
│   ├── order_manager.py
│   ├── position_manager.py
│   └── tastyworks_api.py
├── risk/
│   ├── __init__.py
│   ├── portfolio_risk.py
│   ├── var_calculator.py
│   ├── regime_detection.py
│   └── dynamic_hedging.py
├── backtest/
│   ├── __init__.py
│   ├── engine.py
│   └── metrics.py
├── monitoring/
│   ├── __init__.py
│   ├── dashboard.py
│   └── alerts.py
├── tests/
│   └── ...
├── notebooks/           # Research notebooks
├── logs/
├── requirements.txt
├── docker-compose.yml
└── main.py
```

### 1.3 Configuration Files

**config/settings.py:**
```python
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TastyworksConfig:
    username: str = os.getenv('TW_USERNAME')
    password: str = os.getenv('TW_PASSWORD')
    account_number: str = os.getenv('TW_ACCOUNT')
    is_sandbox: bool = os.getenv('TW_SANDBOX', 'True') == 'True'
    
@dataclass
class DatabaseConfig:
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', 5432))
    name: str = os.getenv('DB_NAME', 'options_trader')
    user: str = os.getenv('DB_USER')
    password: str = os.getenv('DB_PASSWORD')
    
@dataclass
class RiskConfig:
    max_position_size: float = 0.25  # 25% of portfolio
    max_daily_loss: float = 0.05     # 5% daily loss limit
    default_risk_level: str = 'moderate'  # low/moderate/high
    var_confidence: float = 0.95
    leverage_limit: float = 2.0
```

---

## Phase 2: Advanced Pricing Models Implementation (Weeks 2-3)

### 2.1 Merton Jump Diffusion Model

**models/pricing/merton_jump.py:**
```python
import numpy as np
from scipy.stats import norm
from typing import Tuple

class MertonJumpDiffusion:
    def __init__(self, max_iterations: int = 40):
        self.max_iterations = max_iterations
    
    def price(self, S: float, K: float, T: float, r: float, 
              sigma: float, lam: float, mu_j: float, 
              sigma_j: float, option_type: str = 'call') -> float:
        """
        Price option using Merton Jump Diffusion model
        
        Parameters:
        - S: Current stock price
        - K: Strike price
        - T: Time to maturity
        - r: Risk-free rate
        - sigma: Volatility of diffusion
        - lam: Jump intensity (average number of jumps per year)
        - mu_j: Mean of jump size
        - sigma_j: Standard deviation of jump size
        """
        # Compensated drift
        drift = r - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
        
        price = 0
        for n in range(self.max_iterations):
            # Poisson probability
            prob = np.exp(-lam * T) * (lam * T)**n / np.math.factorial(n)
            
            # Adjusted parameters for n jumps
            r_n = drift + n * mu_j / T
            sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
            
            # Black-Scholes price with adjusted parameters
            bs_price = self._black_scholes(S, K, T, r_n, sigma_n, option_type)
            price += prob * bs_price
            
            # Early termination if contribution is negligible
            if prob * bs_price < 1e-10:
                break
        
        return price
    
    def _black_scholes(self, S, K, T, r, sigma, option_type):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
```

### 2.2 Machine Learning Pricing Model

**models/pricing/ml_pricing.py:**
```python
import torch
import torch.nn as nn
import numpy as np

class OptionPricingNN(nn.Module):
    def __init__(self, input_dim=7, hidden_dims=[400, 400, 400, 400]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
class MLPricer:
    def __init__(self, model_path: str = None):
        self.model = OptionPricingNN()
        if model_path:
            self.load_model(model_path)
        self.scaler = None  # Will be fitted during training
        
    def preprocess_features(self, S, K, T, r, sigma, option_type):
        """Convert raw inputs to model features"""
        moneyness = np.log(S / K)
        features = np.array([
            moneyness,
            T,
            r,
            sigma,
            r * T,  # Discount factor component
            sigma * np.sqrt(T),  # Total volatility
            1 if option_type == 'call' else 0
        ])
        return features
    
    def predict_price(self, S, K, T, r, sigma, option_type='call'):
        features = self.preprocess_features(S, K, T, r, sigma, option_type)
        
        if self.scaler:
            features = self.scaler.transform(features.reshape(1, -1))
        
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            price = self.model(features_tensor).item()
        
        return price * S  # Denormalize
```

---

## Phase 3: Trading Strategies Implementation (Weeks 4-5)

### 3.1 Volatility Dispersion Trading

**strategies/volatility/dispersion.py:**
```python
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class DispersionStrategy:
    def __init__(self, index_symbol: str, components: List[str], 
                 capital: float, risk_level: str = 'moderate'):
        self.index_symbol = index_symbol
        self.components = components
        self.capital = capital
        self.risk_multiplier = {'low': 0.5, 'moderate': 1.0, 'high': 1.5}[risk_level]
        
    def calculate_weights(self, market_caps: Dict[str, float]) -> Dict[str, float]:
        """Calculate vega-neutral weights for dispersion trade"""
        total_cap = sum(market_caps.values())
        weights = {symbol: cap/total_cap for symbol, cap in market_caps.items()}
        
        # Select top components representing 70-80% of index
        sorted_components = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        selected = []
        cumulative_weight = 0
        
        for symbol, weight in sorted_components:
            selected.append(symbol)
            cumulative_weight += weight
            if cumulative_weight >= 0.75:
                break
        
        return {s: weights[s] for s in selected}
    
    def generate_signals(self, correlations: pd.DataFrame, 
                        implied_vols: Dict[str, float]) -> List[Dict]:
        """Generate trading signals based on correlation and volatility"""
        signals = []
        avg_correlation = correlations.mean().mean()
        
        if avg_correlation < 0.5:  # Low correlation environment
            # Calculate expected dispersion profit
            index_iv = implied_vols[self.index_symbol]
            weighted_component_iv = sum(
                implied_vols[s] * self.weights.get(s, 0) 
                for s in self.components
            )
            
            dispersion_spread = weighted_component_iv - index_iv
            
            if dispersion_spread > 0.05:  # 5% volatility spread threshold
                position_size = self.capital * 0.1 * self.risk_multiplier
                
                # Short index straddle
                signals.append({
                    'symbol': self.index_symbol,
                    'strategy': 'short_straddle',
                    'size': -position_size,
                    'reason': f'Dispersion trade: correlation={avg_correlation:.2f}'
                })
                
                # Long component straddles
                for symbol in self.selected_components:
                    component_size = position_size * self.weights[symbol]
                    signals.append({
                        'symbol': symbol,
                        'strategy': 'long_straddle',
                        'size': component_size,
                        'reason': 'Dispersion trade component'
                    })
        
        return signals
```

### 3.2 Gamma Scalping Strategy

**strategies/volatility/gamma_scalping.py:**
```python
class GammaScalpingStrategy:
    def __init__(self, rebalance_threshold: float = 0.1, 
                 min_profit_threshold: float = 50):
        self.rebalance_threshold = rebalance_threshold
        self.min_profit_threshold = min_profit_threshold
        self.positions = {}
        
    def calculate_hedge_quantity(self, position: Dict) -> float:
        """Calculate shares needed to maintain delta neutrality"""
        portfolio_delta = sum(
            pos['quantity'] * pos['delta'] 
            for pos in self.positions.values()
        )
        
        # Number of shares to buy/sell
        hedge_shares = -portfolio_delta * 100  # Options are 100-share contracts
        return hedge_shares
    
    def should_rebalance(self, current_price: float, last_hedge_price: float,
                        gamma: float) -> Tuple[bool, float]:
        """Determine if rebalancing is profitable"""
        price_move = current_price - last_hedge_price
        
        # Expected profit from gamma
        gamma_pnl = 0.5 * gamma * (price_move ** 2) * 100
        
        # Estimated transaction costs
        transaction_cost = abs(self.calculate_hedge_quantity(None)) * 0.01
        
        if abs(price_move) > self.rebalance_threshold:
            if gamma_pnl - transaction_cost > self.min_profit_threshold:
                return True, gamma_pnl - transaction_cost
        
        return False, 0
    
    def execute_scalp(self, current_price: float) -> Dict:
        """Execute gamma scalping trade"""
        hedge_qty = self.calculate_hedge_quantity(self.positions)
        
        if abs(hedge_qty) < 1:
            return None
        
        return {
            'action': 'buy' if hedge_qty > 0 else 'sell',
            'quantity': abs(hedge_qty),
            'instrument': 'stock',
            'price': current_price,
            'reason': 'gamma_scalp_rebalance'
        }
```

---

## Phase 4: Tastyworks Integration (Week 6)

### 4.1 API Connection Manager

**execution/tastyworks_api.py:**
```python
import asyncio
from typing import Optional, List, Dict
from tastytrade import Session, Account, DXLinkStreamer
from tastytrade.order import NewOrder, OrderAction, OrderTimeInForce, OrderType
from tastytrade.instruments import Equity, Option
import logging

class TastyworksClient:
    def __init__(self, username: str, password: str, account_number: str, 
                 is_sandbox: bool = True):
        self.username = username
        self.password = password
        self.account_number = account_number
        self.is_sandbox = is_sandbox
        self.session: Optional[Session] = None
        self.account: Optional[Account] = None
        self.streamer: Optional[DXLinkStreamer] = None
        
    async def connect(self):
        """Establish connection to Tastyworks"""
        try:
            self.session = Session(self.username, self.password, 
                                 is_test=self.is_sandbox)
            
            accounts = await self.session.get_accounts()
            self.account = next(
                (a for a in accounts if a.account_number == self.account_number),
                None
            )
            
            if not self.account:
                raise ValueError(f"Account {self.account_number} not found")
            
            # Initialize streamer for real-time data
            self.streamer = await DXLinkStreamer.create(self.session)
            
            logging.info(f"Connected to Tastyworks {'Sandbox' if self.is_sandbox else 'Production'}")
            
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            raise
    
    async def get_option_chain(self, symbol: str, expiration: str = None) -> pd.DataFrame:
        """Fetch option chain with Greeks"""
        option_chain = await self.session.get_option_chain(symbol)
        
        if expiration:
            option_chain = option_chain[option_chain['expiration'] == expiration]
        
        # Add Greeks calculation if not provided
        for idx, row in option_chain.iterrows():
            if pd.isna(row['delta']):
                # Calculate Greeks using QuantLib
                option_chain.loc[idx, 'delta'] = self._calculate_delta(row)
                option_chain.loc[idx, 'gamma'] = self._calculate_gamma(row)
                option_chain.loc[idx, 'theta'] = self._calculate_theta(row)
                option_chain.loc[idx, 'vega'] = self._calculate_vega(row)
        
        return option_chain
    
    async def place_order(self, order_config: Dict) -> str:
        """Place order with comprehensive error handling"""
        try:
            if order_config['order_type'] == 'option_spread':
                order = self._build_spread_order(order_config)
            else:
                order = self._build_single_order(order_config)
            
            response = await self.account.place_order(order, dry_run=False)
            
            logging.info(f"Order placed: {response.order.id}")
            return response.order.id
            
        except Exception as e:
            logging.error(f"Order failed: {e}")
            raise
    
    def _build_spread_order(self, config: Dict) -> NewOrder:
        """Build multi-leg option order"""
        legs = []
        
        for leg in config['legs']:
            option = Option(
                ticker=leg['symbol'],
                expiration=leg['expiration'],
                strike=leg['strike'],
                option_type=leg['option_type']
            )
            
            legs.append({
                'instrument': option,
                'action': OrderAction[leg['action'].upper()],
                'quantity': leg['quantity']
            })
        
        return NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.LIMIT,
            legs=legs,
            price=config.get('limit_price'),
            gtc_date=config.get('gtc_date')
        )
    
    async def stream_quotes(self, symbols: List[str], callback):
        """Stream real-time quotes with automatic reconnection"""
        async def quote_handler(data):
            await callback(data)
        
        await self.streamer.subscribe_quotes(symbols, quote_handler)
        
        # Heartbeat to maintain connection
        while True:
            await asyncio.sleep(60)
            await self.streamer.heartbeat()
```

### 4.2 Order Management System

**execution/order_manager.py:**
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import asyncio
from collections import deque

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    id: str
    symbol: str
    quantity: int
    order_type: str
    limit_price: Optional[float]
    status: OrderStatus
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    
class OrderManager:
    def __init__(self, tastyworks_client, max_orders_per_minute: int = 60):
        self.client = tastyworks_client
        self.orders = {}
        self.order_queue = deque()
        self.rate_limiter = asyncio.Semaphore(max_orders_per_minute)
        
    async def submit_order(self, order_request: Dict) -> Order:
        """Submit order with rate limiting and retry logic"""
        async with self.rate_limiter:
            try:
                # Check risk limits before submission
                if not self._check_risk_limits(order_request):
                    raise ValueError("Order exceeds risk limits")
                
                order_id = await self.client.place_order(order_request)
                
                order = Order(
                    id=order_id,
                    symbol=order_request['symbol'],
                    quantity=order_request['quantity'],
                    order_type=order_request['order_type'],
                    limit_price=order_request.get('limit_price'),
                    status=OrderStatus.SUBMITTED
                )
                
                self.orders[order_id] = order
                return order
                
            except Exception as e:
                logging.error(f"Order submission failed: {e}")
                
                # Retry logic for transient failures
                if "rate limit" in str(e).lower():
                    await asyncio.sleep(5)
                    return await self.submit_order(order_request)
                raise
    
    def _check_risk_limits(self, order_request: Dict) -> bool:
        """Validate order against risk parameters"""
        # Implementation depends on risk management rules
        return True
```

---

## Phase 5: Risk Management System (Week 7)

### 5.1 Dynamic Risk Manager

**risk/portfolio_risk.py:**
```python
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple

class DynamicRiskManager:
    def __init__(self, base_risk_level: str = 'moderate'):
        self.base_risk_level = base_risk_level
        self.regime_model = GaussianMixture(n_components=3)
        self.current_regime = 'normal'
        self.risk_multipliers = {
            'low': {'low': 1.5, 'normal': 1.0, 'high': 0.3},
            'moderate': {'low': 2.0, 'normal': 1.0, 'high': 0.5},
            'high': {'low': 2.5, 'normal': 1.5, 'high': 0.7}
        }
        
    def detect_market_regime(self, returns: np.ndarray, 
                            volatility: np.ndarray) -> str:
        """Detect current market regime using HMM"""
        features = np.column_stack([returns, volatility])
        
        # Fit or update regime model
        if not hasattr(self.regime_model, 'means_'):
            self.regime_model.fit(features)
        
        # Predict current regime
        regime = self.regime_model.predict(features[-1].reshape(1, -1))[0]
        
        regime_names = ['low', 'normal', 'high']
        return regime_names[regime]
    
    def calculate_var(self, portfolio_values: np.ndarray, 
                     confidence: float = 0.95) -> Dict:
        """Calculate Value at Risk using multiple methods"""
        # Historical VaR
        historical_var = np.percentile(portfolio_values, (1 - confidence) * 100)
        
        # Parametric VaR
        mean = np.mean(portfolio_values)
        std = np.std(portfolio_values)
        z_score = norm.ppf(1 - confidence)
        parametric_var = mean + z_score * std
        
        # Monte Carlo VaR
        simulations = np.random.normal(mean, std, 10000)
        monte_carlo_var = np.percentile(simulations, (1 - confidence) * 100)
        
        # Conditional VaR (Expected Shortfall)
        threshold = np.percentile(portfolio_values, (1 - confidence) * 100)
        cvar = np.mean(portfolio_values[portfolio_values <= threshold])
        
        return {
            'historical_var': historical_var,
            'parametric_var': parametric_var,
            'monte_carlo_var': monte_carlo_var,
            'cvar': cvar,
            'confidence': confidence
        }
    
    def adjust_position_sizes(self, base_sizes: Dict[str, float]) -> Dict[str, float]:
        """Adjust position sizes based on current regime"""
        regime = self.detect_market_regime(self.recent_returns, self.recent_volatility)
        multiplier = self.risk_multipliers[self.base_risk_level][regime]
        
        adjusted_sizes = {}
        for symbol, size in base_sizes.items():
            adjusted_sizes[symbol] = size * multiplier
            
            # Apply maximum position limits
            max_size = self.capital * 0.25  # 25% max position
            adjusted_sizes[symbol] = min(adjusted_sizes[symbol], max_size)
        
        return adjusted_sizes
    
    def calculate_kelly_fraction(self, win_probability: float, 
                                win_amount: float, loss_amount: float) -> float:
        """Calculate Kelly Criterion for options with bounded losses"""
        if loss_amount == 0:
            return 0
        
        # Full Kelly
        kelly = (win_probability * win_amount - 
                (1 - win_probability) * loss_amount) / win_amount
        
        # Quarter Kelly for reduced variance
        return kelly * 0.25
```

### 5.2 Real-time Monitoring

**monitoring/dashboard.py:**
```python
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd

class TradingDashboard:
    def __init__(self, portfolio_manager, risk_manager):
        self.portfolio = portfolio_manager
        self.risk = risk_manager
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Options Trading Dashboard'),
            
            # Risk Metrics
            html.Div([
                html.H3('Risk Metrics'),
                html.Div(id='risk-metrics'),
                dcc.Interval(id='risk-update', interval=5000)  # Update every 5 seconds
            ]),
            
            # Portfolio Greeks
            html.Div([
                html.H3('Portfolio Greeks'),
                dcc.Graph(id='greeks-chart'),
                dcc.Interval(id='greeks-update', interval=10000)
            ]),
            
            # P&L Chart
            html.Div([
                html.H3('Real-time P&L'),
                dcc.Graph(id='pnl-chart'),
                dcc.Interval(id='pnl-update', interval=1000)
            ]),
            
            # Position Monitor
            html.Div([
                html.H3('Active Positions'),
                html.Div(id='positions-table'),
                dcc.Interval(id='positions-update', interval=5000)
            ]),
            
            # Risk Level Control
            html.Div([
                html.H3('Risk Level'),
                dcc.Dropdown(
                    id='risk-level',
                    options=[
                        {'label': 'Low', 'value': 'low'},
                        {'label': 'Moderate', 'value': 'moderate'},
                        {'label': 'High', 'value': 'high'}
                    ],
                    value='moderate'
                )
            ])
        ])
    
    def setup_callbacks(self):
        @self.app.callback(
            Output('risk-metrics', 'children'),
            Input('risk-update', 'n_intervals')
        )
        def update_risk_metrics(n):
            var_metrics = self.risk.calculate_var(self.portfolio.get_values())
            
            return html.Div([
                html.P(f"VaR (95%): ${var_metrics['historical_var']:,.2f}"),
                html.P(f"CVaR: ${var_metrics['cvar']:,.2f}"),
                html.P(f"Current Regime: {self.risk.current_regime}"),
                html.P(f"Leverage: {self.portfolio.get_leverage():.2f}x")
            ])
        
        @self.app.callback(
            Output('greeks-chart', 'figure'),
            Input('greeks-update', 'n_intervals')
        )
        def update_greeks_chart(n):
            greeks = self.portfolio.get_portfolio_greeks()
            
            fig = go.Figure(data=[
                go.Bar(name='Values', x=list(greeks.keys()), y=list(greeks.values()))
            ])
            
            fig.update_layout(title='Portfolio Greeks')
            return fig
```

---

## Phase 6: Backtesting Framework (Week 8)

### 6.1 Options Backtesting Engine

**backtest/engine.py:**
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Callable
from dataclasses import dataclass

@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    
class OptionsBacktester:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []
        
    def run_backtest(self, data: pd.DataFrame, strategy: Callable,
                    start_date: str, end_date: str) -> BacktestResult:
        """Run backtest on historical options data"""
        
        # Filter data for backtest period
        mask = (data.index >= start_date) & (data.index <= end_date)
        backtest_data = data[mask]
        
        for timestamp, row in backtest_data.iterrows():
            # Update existing positions
            self._update_positions(row)
            
            # Generate signals
            signals = strategy(row, self.positions, self.capital)
            
            # Execute trades
            for signal in signals:
                self._execute_trade(signal, row)
            
            # Record equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self._calculate_equity(row)
            })
        
        return self._calculate_metrics()
    
    def _execute_trade(self, signal: Dict, market_data: pd.Series):
        """Execute trade with realistic fills"""
        
        # Add slippage
        if signal['action'] == 'buy':
            fill_price = market_data['ask'] * 1.001  # 0.1% slippage
        else:
            fill_price = market_data['bid'] * 0.999
        
        # Calculate position size
        contracts = int(signal['size'] / (fill_price * 100))
        
        # Update positions
        position_id = f"{signal['symbol']}_{signal['strike']}_{signal['expiry']}"
        
        if position_id in self.positions:
            self.positions[position_id]['quantity'] += contracts
        else:
            self.positions[position_id] = {
                'symbol': signal['symbol'],
                'strike': signal['strike'],
                'expiry': signal['expiry'],
                'quantity': contracts,
                'entry_price': fill_price,
                'option_type': signal['option_type']
            }
        
        # Record trade
        self.trade_history.append({
            'timestamp': market_data.name,
            'symbol': signal['symbol'],
            'action': signal['action'],
            'quantity': contracts,
            'price': fill_price,
            'commission': contracts * 0.65  # Typical options commission
        })
    
    def _calculate_metrics(self) -> BacktestResult:
        """Calculate backtest performance metrics"""
        equity_df = pd.DataFrame(self.equity_curve)
        returns = equity_df['equity'].pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital - 1)
        
        # Sharpe ratio (assuming 252 trading days)
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate and profit factor
        trades_df = pd.DataFrame(self.trade_history)
        if len(trades_df) > 0:
            profitable_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = len(profitable_trades) / len(trades_df)
            
            gross_profit = profitable_trades['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        else:
            win_rate = 0
            profit_factor = 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df)
        )
```

---

## Phase 7: Production Deployment (Week 9)

### 7.1 Docker Configuration

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: options_trader
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  trader:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      - TW_USERNAME=${TW_USERNAME}
      - TW_PASSWORD=${TW_PASSWORD}
      - TW_ACCOUNT=${TW_ACCOUNT}
      - DB_HOST=postgres
      - REDIS_HOST=redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
  
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8050:8050"
    depends_on:
      - trader
      - postgres
    environment:
      - DB_HOST=postgres
  
  monitoring:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
```

### 7.2 Main Application Entry Point

**main.py:**
```python
import asyncio
import logging
import signal
import sys
from typing import Dict, List

from config.settings import TastyworksConfig, DatabaseConfig, RiskConfig
from execution.tastyworks_api import TastyworksClient
from execution.order_manager import OrderManager
from risk.portfolio_risk import DynamicRiskManager
from strategies.volatility.dispersion import DispersionStrategy
from strategies.volatility.gamma_scalping import GammaScalpingStrategy
from monitoring.dashboard import TradingDashboard

class OptionsTradingBot:
    def __init__(self):
        self.config = self._load_config()
        self.client = None
        self.order_manager = None
        self.risk_manager = None
        self.strategies = []
        self.running = False
        
    def _load_config(self) -> Dict:
        return {
            'tastyworks': TastyworksConfig(),
            'database': DatabaseConfig(),
            'risk': RiskConfig()
        }
    
    async def initialize(self):
        """Initialize all components"""
        # Connect to Tastyworks
        self.client = TastyworksClient(
            self.config['tastyworks'].username,
            self.config['tastyworks'].password,
            self.config['tastyworks'].account_number,
            self.config['tastyworks'].is_sandbox
        )
        await self.client.connect()
        
        # Initialize managers
        self.order_manager = OrderManager(self.client)
        self.risk_manager = DynamicRiskManager(
            self.config['risk'].default_risk_level
        )
        
        # Initialize strategies
        self.strategies = [
            DispersionStrategy('SPY', ['AAPL', 'MSFT', 'GOOGL'], 100000),
            GammaScalpingStrategy()
        ]
        
        logging.info("Bot initialized successfully")
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        
        while self.running:
            try:
                # Fetch market data
                market_data = await self._fetch_market_data()
                
                # Update risk manager
                self.risk_manager.update(market_data)
                
                # Generate signals from all strategies
                all_signals = []
                for strategy in self.strategies:
                    signals = await strategy.generate_signals(market_data)
                    all_signals.extend(signals)
                
                # Filter signals through risk management
                approved_signals = self.risk_manager.filter_signals(all_signals)
                
                # Execute approved trades
                for signal in approved_signals:
                    await self.order_manager.submit_order(signal)
                
                # Sleep before next iteration
                await asyncio.sleep(1)  # 1-second loop for real-time trading
                
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def shutdown(self):
        """Graceful shutdown"""
        logging.info("Shutting down bot...")
        self.running = False
        
        # Close all positions if configured
        if self.config['risk'].close_on_shutdown:
            await self._close_all_positions()
        
        # Disconnect from services
        if self.client:
            await self.client.disconnect()
        
        logging.info("Shutdown complete")
    
    async def _fetch_market_data(self) -> Dict:
        """Fetch current market data"""
        # Implementation depends on data requirements
        pass
    
    async def _close_all_positions(self):
        """Close all open positions"""
        # Implementation for emergency shutdown
        pass

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    asyncio.create_task(bot.shutdown())
    sys.exit(0)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    # Handle shutdown signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run bot
    bot = OptionsTradingBot()
    
    async def main():
        await bot.initialize()
        await bot.run()
    
    asyncio.run(main())
```

---

## Testing Strategy

### Unit Tests Structure
```
tests/
├── test_pricing/
│   ├── test_black_scholes.py
│   ├── test_merton_jump.py
│   └── test_ml_pricing.py
├── test_strategies/
│   ├── test_dispersion.py
│   └── test_gamma_scalping.py
├── test_risk/
│   └── test_var_calculation.py
└── test_integration/
    └── test_tastyworks_api.py
```

### Performance Benchmarks
- Pricing models: < 1ms per option
- Greeks calculation: < 10ms for entire chain
- Order execution: < 100ms latency
- Risk calculations: < 500ms for full portfolio
- Data processing: 10,000 options/second

---

## Deployment Checklist

### Pre-Production
- [ ] Complete unit test coverage (>80%)
- [ ] Backtest all strategies on 2+ years of data
- [ ] Paper trade for minimum 30 days
- [ ] Stress test with 10x normal volume
- [ ] Implement circuit breakers and kill switches
- [ ] Set up monitoring and alerting
- [ ] Document all configuration parameters
- [ ] Create runbooks for common issues

### Production Launch
- [ ] Start with minimum capital (5% of target)
- [ ] Enable single strategy at a time
- [ ] Monitor for first 48 hours continuously
- [ ] Daily performance reviews for first month
- [ ] Gradual capital increase (weekly 20% increments)
- [ ] Full automation after 3 months stable operation

### Risk Controls
- [ ] Maximum daily loss: 5% of capital
- [ ] Maximum position size: 25% of portfolio
- [ ] Correlation limits: No more than 3 correlated positions
- [ ] Leverage limit: 2x maximum
- [ ] Mandatory hedging for positions > $10,000
- [ ] Automatic shutdown on technical failures

---

## Monthly Development Timeline

**Month 1:** Core infrastructure, pricing models, basic strategies
**Month 2:** Tastyworks integration, risk management, backtesting
**Month 3:** Paper trading, optimization, monitoring tools

## Next Steps

1. Set up development environment with all dependencies
2. Implement core pricing models with test coverage
3. Build Tastyworks connection layer
4. Develop first strategy (recommend starting with gamma scalping)
5. Create backtesting framework
6. Begin paper trading
7. Iterate based on results

This project will create a sophisticated, production-ready options trading system. Start with Phase 1 and progress systematically through each phase, ensuring thorough testing at each stage.