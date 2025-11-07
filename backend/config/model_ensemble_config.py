"""
Model Ensemble Configuration
============================

Configuration for the model ensemble strategy including model weights,
calibration schedules, performance tracking, and universe management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class CalibrationFrequency(Enum):
    """Model calibration frequency options"""
    NEVER = "never"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class ModelType(Enum):
    """Available pricing models"""
    BLACK_SCHOLES = "black_scholes"
    MERTON_JUMP = "merton_jump"
    HESTON = "heston"
    ML_NEURAL = "ml_neural"


@dataclass
class ModelWeights:
    """Model weight configuration"""
    black_scholes: float = 0.20
    merton_jump: float = 0.25
    heston: float = 0.30
    ml_neural: float = 0.25

    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.black_scholes + self.merton_jump + self.heston + self.ml_neural
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Model weights must sum to 1.0, got {total}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'black_scholes': self.black_scholes,
            'merton_jump': self.merton_jump,
            'heston': self.heston,
            'ml_neural': self.ml_neural
        }


@dataclass
class RegimeWeights:
    """Model weights for different market regimes"""
    calm: ModelWeights = field(default_factory=lambda: ModelWeights(
        black_scholes=0.30,
        merton_jump=0.15,
        heston=0.25,
        ml_neural=0.30
    ))
    volatile: ModelWeights = field(default_factory=lambda: ModelWeights(
        black_scholes=0.15,
        merton_jump=0.30,
        heston=0.35,
        ml_neural=0.20
    ))
    trending: ModelWeights = field(default_factory=lambda: ModelWeights(
        black_scholes=0.20,
        merton_jump=0.20,
        heston=0.25,
        ml_neural=0.35
    ))
    crisis: ModelWeights = field(default_factory=lambda: ModelWeights(
        black_scholes=0.10,
        merton_jump=0.40,
        heston=0.35,
        ml_neural=0.15
    ))


@dataclass
class CalibrationSchedule:
    """Calibration schedule for each model"""
    black_scholes: CalibrationFrequency = CalibrationFrequency.NEVER
    merton_jump: CalibrationFrequency = CalibrationFrequency.WEEKLY
    heston: CalibrationFrequency = CalibrationFrequency.DAILY
    ml_neural: CalibrationFrequency = CalibrationFrequency.MONTHLY

    def get_days(self, model: str) -> Optional[int]:
        """Get calibration frequency in days"""
        freq_map = {
            CalibrationFrequency.NEVER: None,
            CalibrationFrequency.DAILY: 1,
            CalibrationFrequency.WEEKLY: 7,
            CalibrationFrequency.MONTHLY: 30,
            CalibrationFrequency.QUARTERLY: 90
        }
        freq = getattr(self, model, CalibrationFrequency.NEVER)
        return freq_map[freq]


@dataclass
class PerformanceTracking:
    """Performance tracking configuration"""
    lookback_days: int = 30  # Days to look back for performance
    min_trades_for_rebalance: int = 20  # Minimum trades before rebalancing weights
    weight_rebalance_threshold: float = 0.10  # 10% weight change triggers rebalance
    performance_update_frequency: int = 10  # Update metrics every N iterations
    track_top_n_predictions: int = 50  # Track top N predictions for analysis
    save_predictions_every: int = 100  # Save predictions to disk every N
    smoothing_factor: float = 0.7  # Weight smoothing (0.7 = keep 70% of old weight)


@dataclass
class UniverseConfig:
    """Trading universe configuration"""
    # Universe selection criteria
    universe_size: int = 50
    min_option_volume: int = 1000
    max_bid_ask_spread_pct: float = 0.05
    min_market_cap: float = 10e9  # $10B
    min_daily_stock_volume: float = 1e6  # 1M shares

    # Default universe
    default_symbols: List[str] = field(default_factory=lambda: [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
        # Financial
        'JPM', 'BAC', 'WFC', 'GS', 'MS',
        # Healthcare
        'JNJ', 'PFE', 'ABBV', 'MRK', 'UNH',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'DIA'
    ])

    # Sector allocations
    max_sector_concentration: float = 0.40  # Max 40% in one sector
    target_sector_weights: Dict[str, float] = field(default_factory=lambda: {
        'technology': 0.30,
        'financial': 0.20,
        'healthcare': 0.15,
        'consumer': 0.15,
        'industrial': 0.10,
        'energy': 0.05,
        'etf': 0.05
    })


@dataclass
class PredictionFilters:
    """Filters for ensemble predictions"""
    min_edge_magnitude: float = 0.02  # 2% minimum edge
    min_confidence_score: float = 0.60  # 60% minimum confidence
    max_model_disagreement: float = 0.30  # 30% max disagreement for directional
    min_model_agreement_count: int = 2  # At least 2 models must agree

    # Strategy-specific filters
    directional_min_edge: float = 0.05  # 5% for directional trades
    volatility_min_disagreement: float = 0.15  # 15% disagreement for vol arb
    gamma_min_gamma: float = 0.01  # Minimum gamma for gamma scalping


@dataclass
class PositionSizing:
    """Position sizing configuration"""
    max_position_pct: float = 0.10  # 10% max per position
    max_strategy_allocation: float = 0.30  # 30% max per strategy type
    max_portfolio_utilization: float = 0.80  # 80% max capital utilization
    min_position_pct: float = 0.01  # 1% minimum position
    max_positions_per_strategy: int = 5  # Max positions per strategy
    max_total_positions: int = 20  # Max total positions

    # Risk-based adjustments
    confidence_size_adjustment: bool = True  # Adjust size based on confidence
    disagreement_size_penalty: float = 0.30  # Reduce size by 30% for high disagreement
    regime_size_adjustments: Dict[str, float] = field(default_factory=lambda: {
        'calm': 1.2,
        'volatile': 0.8,
        'trending': 1.0,
        'crisis': 0.5
    })


@dataclass
class ExecutionConfig:
    """Trade execution configuration"""
    # Entry rules
    limit_price_buffer: float = 0.02  # 2% above fair value
    order_time_limit: str = "1_day"  # Cancel unfilled orders after 1 day
    min_volume_threshold: int = 10  # Minimum volume for entry

    # Exit rules
    profit_target_pct: float = 0.70  # Take 70% of expected edge
    stop_loss_pct: float = 0.25  # 25% stop loss
    time_decay_exit_dte: int = 50  # Exit at 50 days to expiry
    iv_change_exit_threshold: float = 0.15  # Exit if IV changes >15%

    # Slippage and costs
    expected_slippage_pct: float = 0.001  # 0.1% slippage
    commission_per_contract: float = 0.65  # $0.65 per contract


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    rebalance_frequency: str = "daily"
    performance_report_frequency: str = "hourly"
    alert_on_large_disagreement: bool = True
    disagreement_alert_threshold: float = 0.40  # Alert if >40% disagreement

    # Greek monitoring
    monitor_greeks: bool = True
    delta_neutral_threshold: float = 0.10  # Maintain delta within ±10
    gamma_scalping_threshold: float = 0.01  # Minimum gamma for scalping
    vega_limit: float = 0.20  # Maximum vega exposure


@dataclass
class ModelEnsembleConfig:
    """Complete model ensemble configuration"""
    # Model configuration
    initial_weights: ModelWeights = field(default_factory=ModelWeights)
    regime_weights: RegimeWeights = field(default_factory=RegimeWeights)
    calibration_schedule: CalibrationSchedule = field(default_factory=CalibrationSchedule)

    # Performance tracking
    performance_tracking: PerformanceTracking = field(default_factory=PerformanceTracking)

    # Universe management
    universe: UniverseConfig = field(default_factory=UniverseConfig)

    # Prediction filtering
    prediction_filters: PredictionFilters = field(default_factory=PredictionFilters)

    # Position sizing
    position_sizing: PositionSizing = field(default_factory=PositionSizing)

    # Execution
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Feature flags
    use_adaptive_weights: bool = True  # Dynamically adjust weights based on performance
    use_regime_detection: bool = True  # Use market regime detection
    use_ml_model: bool = True  # Use ML pricing model if available
    use_correlation_filters: bool = True  # Filter based on position correlations

    def validate(self) -> bool:
        """Validate configuration"""
        # Validate model weights
        self.initial_weights.__post_init__()
        self.regime_weights.calm.__post_init__()
        self.regime_weights.volatile.__post_init__()
        self.regime_weights.trending.__post_init__()
        self.regime_weights.crisis.__post_init__()

        # Validate thresholds
        assert 0 < self.prediction_filters.min_edge_magnitude < 1
        assert 0 < self.prediction_filters.min_confidence_score < 1
        assert 0 < self.position_sizing.max_position_pct < 1
        assert 0 < self.position_sizing.max_portfolio_utilization < 1

        return True


# Pre-configured templates for different trading styles
def get_conservative_config() -> ModelEnsembleConfig:
    """Conservative configuration with lower risk"""
    config = ModelEnsembleConfig()

    # More weight to Black-Scholes in calm markets
    config.initial_weights = ModelWeights(
        black_scholes=0.35,
        merton_jump=0.20,
        heston=0.25,
        ml_neural=0.20
    )

    # Stricter filters
    config.prediction_filters.min_edge_magnitude = 0.03  # 3% minimum
    config.prediction_filters.min_confidence_score = 0.70  # 70% confidence

    # Smaller positions
    config.position_sizing.max_position_pct = 0.05  # 5% max
    config.position_sizing.max_portfolio_utilization = 0.60  # 60% max

    return config


def get_moderate_config() -> ModelEnsembleConfig:
    """Moderate configuration (default)"""
    return ModelEnsembleConfig()


def get_aggressive_config() -> ModelEnsembleConfig:
    """Aggressive configuration with higher risk tolerance"""
    config = ModelEnsembleConfig()

    # More weight to jump and ML models
    config.initial_weights = ModelWeights(
        black_scholes=0.15,
        merton_jump=0.30,
        heston=0.30,
        ml_neural=0.25
    )

    # Looser filters
    config.prediction_filters.min_edge_magnitude = 0.015  # 1.5% minimum
    config.prediction_filters.min_confidence_score = 0.50  # 50% confidence

    # Larger positions
    config.position_sizing.max_position_pct = 0.15  # 15% max
    config.position_sizing.max_portfolio_utilization = 0.90  # 90% max

    # More positions
    config.position_sizing.max_total_positions = 30

    return config


# Example usage
if __name__ == "__main__":
    # Create and validate configuration
    config = get_moderate_config()

    if config.validate():
        print("✓ Model ensemble configuration valid")
        print(f"Initial weights: {config.initial_weights.to_dict()}")
        print(f"Universe size: {config.universe.universe_size}")
        print(f"Min edge: {config.prediction_filters.min_edge_magnitude:.1%}")
        print(f"Max position: {config.position_sizing.max_position_pct:.1%}")