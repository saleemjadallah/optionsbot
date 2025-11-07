"""
Trading parameters and strategy configurations
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderAction(Enum):
    """Order action enumeration"""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_OPEN = "sell_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_CLOSE = "sell_to_close"


class StrategyType(Enum):
    """Strategy type enumeration"""
    # Directional strategies
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    
    # Volatility strategies
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    GAMMA_SCALPING = "gamma_scalping"
    VOLATILITY_ARBITRAGE = "volatility_arbitrage"
    DISPERSION = "dispersion"
    
    # Neutral strategies
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    DELTA_NEUTRAL = "delta_neutral"
    
    # Advanced strategies
    RATIO_SPREAD = "ratio_spread"
    BACKSPREAD = "backspread"
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"


@dataclass
class StrategyParameters:
    """Base parameters for all strategies"""
    strategy_type: StrategyType
    enabled: bool = True
    allocation_percent: float = 0.1  # 10% of capital
    max_positions: int = 5
    min_dte: int = 7  # Minimum days to expiration
    max_dte: int = 45  # Maximum days to expiration
    min_iv_rank: float = 0.3  # Minimum IV rank (0-1)
    max_iv_rank: float = 0.7  # Maximum IV rank (0-1)
    profit_target: float = 0.5  # 50% of max profit
    stop_loss: float = 2.0  # 2x credit received
    
    def validate(self) -> bool:
        """Validate strategy parameters"""
        if not 0 < self.allocation_percent <= 1:
            raise ValueError(f"Invalid allocation percent: {self.allocation_percent}")
        if self.min_dte < 0 or self.max_dte < self.min_dte:
            raise ValueError(f"Invalid DTE range: {self.min_dte}-{self.max_dte}")
        if not 0 <= self.min_iv_rank <= 1 or not 0 <= self.max_iv_rank <= 1:
            raise ValueError(f"Invalid IV rank range")
        return True


@dataclass
class IronCondorParameters(StrategyParameters):
    """Iron Condor specific parameters"""
    strategy_type: StrategyType = field(default=StrategyType.IRON_CONDOR)
    short_delta: float = 0.15  # Delta for short strikes
    wing_width: int = 5  # Width between short and long strikes
    min_credit: float = 0.35  # Minimum credit as % of width
    max_loss_per_trade: float = 500  # Maximum loss per trade
    
    # Market conditions
    preferred_iv_environment: str = "high"  # high, normal, low
    avoid_earnings: bool = True
    earnings_buffer_days: int = 7
    
    # Entry filters
    min_roc: float = 0.10  # Minimum return on capital
    max_bid_ask_spread: float = 0.10  # Maximum spread as $ amount
    
    # Management rules
    manage_at_percent: float = 0.25  # Manage at 25% of max profit
    roll_tested_side: bool = True
    roll_when_delta: float = 0.30  # Roll when short delta exceeds this


@dataclass
class GammaScalpingParameters(StrategyParameters):
    """Gamma Scalping specific parameters"""
    strategy_type: StrategyType = field(default=StrategyType.GAMMA_SCALPING)
    initial_delta: float = 0.50  # ATM options
    min_gamma: float = 0.01  # Minimum gamma for entry
    rebalance_threshold: float = 0.10  # Rebalance when delta moves by 10%
    min_profit_per_scalp: float = 50  # Minimum profit to trigger scalp
    max_scalps_per_day: int = 10
    
    # Volatility parameters
    target_iv_percentile: float = 0.75  # Enter when IV is in 75th percentile
    min_realized_iv_premium: float = 0.05  # RV must be 5% below IV
    
    # Position sizing
    contracts_per_position: int = 10
    max_underlying_shares: int = 1000
    
    # Exit conditions
    exit_on_iv_collapse: bool = True
    iv_collapse_threshold: float = 0.20  # Exit if IV drops 20%
    time_stop_days: int = 5  # Exit after 5 days regardless


@dataclass
class DispersionParameters(StrategyParameters):
    """Dispersion Trading specific parameters"""
    strategy_type: StrategyType = field(default=StrategyType.DISPERSION)
    index_symbol: str = "SPY"
    component_symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "NVDA", "TSLA", "JPM", "V", "JNJ"
    ])
    min_correlation: float = 0.30  # Minimum average correlation
    max_correlation: float = 0.60  # Maximum average correlation
    min_dispersion_spread: float = 0.05  # 5% vol spread required
    
    # Position construction
    weight_by_market_cap: bool = True
    max_component_weight: float = 0.20  # No component > 20%
    min_components: int = 5  # Minimum components for trade
    
    # Hedging parameters
    vega_neutral: bool = True
    delta_hedge_frequency: str = "daily"  # daily, continuous, threshold
    hedge_threshold: float = 0.05  # Hedge when portfolio delta > 5%


@dataclass
class MarketRegimeParameters:
    """Market regime detection parameters"""
    lookback_period: int = 60  # Days for regime detection
    vol_regimes: Dict[str, tuple] = field(default_factory=lambda: {
        "low": (0, 15),
        "normal": (15, 25),
        "high": (25, 40),
        "extreme": (40, 100)
    })
    trend_threshold: float = 0.02  # 2% move defines trend
    
    # Regime-specific adjustments
    regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "low": 1.5,      # Increase size in low vol
        "normal": 1.0,   # Normal sizing
        "high": 0.7,     # Reduce size in high vol
        "extreme": 0.3   # Minimal size in extreme vol
    })
    
    # Correlation regimes
    correlation_buckets: Dict[str, tuple] = field(default_factory=lambda: {
        "decorrelated": (0, 0.3),
        "normal": (0.3, 0.6),
        "correlated": (0.6, 0.8),
        "crisis": (0.8, 1.0)
    })


@dataclass
class ExecutionParameters:
    """Order execution parameters"""
    use_limit_orders: bool = True
    limit_order_offset: float = 0.01  # $0.01 better than mid
    max_order_attempts: int = 3
    order_timeout_seconds: int = 30
    
    # Smart routing
    use_smart_routing: bool = True
    preferred_exchange: Optional[str] = None
    
    # Slippage estimates
    expected_slippage: Dict[str, float] = field(default_factory=lambda: {
        "market": 0.02,  # 2 cents for market orders
        "limit": 0.01,   # 1 cent for limit orders
        "stop": 0.03     # 3 cents for stop orders
    })
    
    # Fill improvement
    use_midpoint_orders: bool = True
    use_iceberg_orders: bool = False
    iceberg_display_size: int = 10


@dataclass
class Greeks:
    """Option Greeks container"""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    @property
    def speed(self) -> float:
        """Rate of change of gamma (3rd derivative)"""
        # Would need to be calculated from gamma changes
        return 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho
        }


@dataclass
class OptionContract:
    """Option contract details"""
    symbol: str
    strike: float
    expiration: str
    option_type: OptionType
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    greeks: Optional[Greeks] = None
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.ask - self.bid if self.bid and self.ask else 0.0
    
    @property
    def spread_percent(self) -> float:
        """Calculate spread as percentage of mid price"""
        if self.mid_price > 0:
            return self.spread / self.mid_price
        return 0.0