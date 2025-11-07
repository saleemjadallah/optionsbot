"""
Risk management limits and thresholds
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    AGGRESSIVE = "aggressive"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PositionLimits:
    """Position-level risk limits"""
    max_contracts_per_position: int = 100
    max_position_value: float = 50000  # $50,000 max per position
    max_position_delta: float = 100  # Maximum delta exposure
    max_position_gamma: float = 50  # Maximum gamma exposure
    max_position_vega: float = 500  # Maximum vega exposure
    max_position_theta: float = -500  # Maximum theta (note: negative)
    
    # Percentage limits
    max_percent_of_portfolio: float = 0.25  # 25% of total portfolio
    max_percent_of_daily_volume: float = 0.05  # 5% of average daily volume
    
    # Loss limits
    max_loss_per_position: float = 2000  # $2,000 max loss
    stop_loss_percent: float = 0.50  # 50% of premium paid
    
    def validate_position(self, position: Dict) -> tuple[bool, Optional[str]]:
        """Validate a position against limits"""
        if position.get('contracts', 0) > self.max_contracts_per_position:
            return False, f"Exceeds max contracts: {position['contracts']} > {self.max_contracts_per_position}"
        
        if position.get('value', 0) > self.max_position_value:
            return False, f"Exceeds max value: ${position['value']} > ${self.max_position_value}"
        
        if abs(position.get('delta', 0)) > self.max_position_delta:
            return False, f"Exceeds max delta: {position['delta']} > {self.max_position_delta}"
        
        return True, None


@dataclass
class PortfolioLimits:
    """Portfolio-level risk limits"""
    max_open_positions: int = 20
    max_portfolio_value: float = 500000  # $500,000 max deployed
    max_leverage: float = 2.0  # 2x leverage maximum
    max_margin_usage: float = 0.75  # 75% of available margin
    
    # Greeks limits (portfolio-wide)
    max_portfolio_delta: float = 500
    max_portfolio_gamma: float = 200
    max_portfolio_vega: float = 2000
    max_portfolio_theta: float = -2000
    
    # Concentration limits
    max_single_underlying_exposure: float = 0.30  # 30% in one underlying
    max_sector_exposure: float = 0.40  # 40% in one sector
    max_correlated_positions: int = 5  # Max positions with >0.7 correlation
    
    # Daily limits
    max_daily_trades: int = 50
    max_daily_contracts: int = 500
    max_daily_commission: float = 500  # $500 in commissions
    
    def check_portfolio_health(self, portfolio: Dict) -> Dict[str, bool]:
        """Check portfolio against all limits"""
        return {
            'positions_ok': portfolio['open_positions'] <= self.max_open_positions,
            'value_ok': portfolio['total_value'] <= self.max_portfolio_value,
            'leverage_ok': portfolio['leverage'] <= self.max_leverage,
            'margin_ok': portfolio['margin_usage'] <= self.max_margin_usage,
            'delta_ok': abs(portfolio['total_delta']) <= self.max_portfolio_delta,
            'gamma_ok': abs(portfolio['total_gamma']) <= self.max_portfolio_gamma,
            'vega_ok': abs(portfolio['total_vega']) <= self.max_portfolio_vega,
            'theta_ok': portfolio['total_theta'] >= -abs(self.max_portfolio_theta)
        }


@dataclass
class DrawdownLimits:
    """Drawdown and loss limits"""
    max_daily_drawdown: float = 0.05  # 5% daily loss limit
    max_weekly_drawdown: float = 0.10  # 10% weekly loss limit
    max_monthly_drawdown: float = 0.15  # 15% monthly loss limit
    max_total_drawdown: float = 0.25  # 25% maximum drawdown
    
    # Consecutive loss limits
    max_consecutive_losses: int = 5
    max_consecutive_loss_amount: float = 5000  # $5,000
    
    # Recovery requirements
    drawdown_recovery_period: int = 5  # Days to wait after max drawdown
    reduce_size_after_drawdown: float = 0.5  # Reduce size by 50% after drawdown
    
    # Circuit breakers
    daily_loss_circuit_breaker: float = 0.03  # Stop at 3% daily loss
    intraday_loss_circuit_breaker: float = 0.02  # Stop at 2% intraday loss
    
    def calculate_current_drawdown(self, peak_value: float, current_value: float) -> float:
        """Calculate current drawdown percentage"""
        if peak_value <= 0:
            return 0.0
        return (peak_value - current_value) / peak_value


@dataclass
class VolatilityLimits:
    """Volatility-based risk limits"""
    max_iv_for_selling: float = 0.50  # Don't sell options above 50% IV
    min_iv_for_selling: float = 0.15  # Don't sell options below 15% IV
    max_iv_for_buying: float = 0.80  # Don't buy options above 80% IV
    
    # IV rank limits
    min_iv_rank_for_selling: float = 0.30  # Sell when IV rank > 30%
    max_iv_rank_for_buying: float = 0.70  # Buy when IV rank < 70%
    
    # Realized vs Implied
    min_rv_iv_spread: float = 0.05  # Need 5% spread for vol arb
    
    # Volatility regime limits
    vol_regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "low": 1.5,      # Increase size in low vol
        "normal": 1.0,   # Normal sizing
        "high": 0.7,     # Reduce size in high vol
        "extreme": 0.3   # Minimal size in extreme vol
    })
    
    # VIX-based limits
    max_vix_for_normal_trading: float = 30
    vix_panic_threshold: float = 40
    vix_complacency_threshold: float = 12


@dataclass
class TimeDecayLimits:
    """Time decay and expiration limits"""
    min_dte_for_selling: int = 20  # Don't sell options < 20 DTE
    max_dte_for_selling: int = 60  # Don't sell options > 60 DTE
    min_dte_for_buying: int = 7   # Don't buy options < 7 DTE
    max_dte_for_buying: int = 90  # Don't buy options > 90 DTE
    
    # Gamma risk limits
    max_gamma_exposure_near_expiry: float = 100  # Reduce gamma < 7 DTE
    close_positions_dte: int = 2  # Close all positions at 2 DTE
    
    # Weekend risk
    reduce_size_before_weekend: float = 0.5  # Reduce by 50% on Friday
    avoid_weekend_gap_risk: bool = True


@dataclass
class CorrelationLimits:
    """Correlation and diversification limits"""
    max_portfolio_correlation: float = 0.60  # Max average correlation
    max_position_correlation: float = 0.80  # Max correlation between two positions
    min_diversification_ratio: float = 0.30  # Minimum diversification
    
    # Sector limits
    max_positions_per_sector: int = 5
    sector_weights: Dict[str, float] = field(default_factory=lambda: {
        "technology": 0.30,
        "financials": 0.20,
        "healthcare": 0.20,
        "consumer": 0.15,
        "industrials": 0.15
    })
    
    # Asset class limits
    equity_options_weight: float = 0.70
    index_options_weight: float = 0.20
    etf_options_weight: float = 0.10


@dataclass
class MarginRequirements:
    """Margin and capital requirements"""
    min_account_value: float = 25000  # PDT rule
    min_excess_liquidity: float = 10000  # Minimum cash buffer
    max_margin_usage: float = 0.75  # 75% max margin usage
    
    # Strategy-specific margin
    margin_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "cash_secured_put": 1.0,
        "covered_call": 0.0,  # No additional margin
        "naked_put": 0.20,    # 20% of strike
        "naked_call": 0.30,   # 30% of underlying
        "spread": 0.0,        # Max loss is margin
        "iron_condor": 0.0    # Max loss is margin
    })
    
    # Stress test multipliers
    stress_test_multiplier: float = 2.0  # 2x normal margin for stress
    overnight_margin_multiplier: float = 1.5  # 50% more margin overnight


@dataclass
class RiskAlerts:
    """Risk alert thresholds and messages"""
    alerts: List[Dict] = field(default_factory=lambda: [
        {
            "metric": "daily_loss",
            "threshold": 0.02,
            "level": AlertLevel.WARNING,
            "message": "Daily loss exceeds 2%"
        },
        {
            "metric": "daily_loss",
            "threshold": 0.05,
            "level": AlertLevel.CRITICAL,
            "message": "Daily loss exceeds 5% - stopping trading"
        },
        {
            "metric": "margin_usage",
            "threshold": 0.70,
            "level": AlertLevel.WARNING,
            "message": "Margin usage above 70%"
        },
        {
            "metric": "margin_usage",
            "threshold": 0.90,
            "level": AlertLevel.EMERGENCY,
            "message": "Margin usage above 90% - reduce positions"
        },
        {
            "metric": "portfolio_delta",
            "threshold": 300,
            "level": AlertLevel.WARNING,
            "message": "Portfolio delta exposure high"
        },
        {
            "metric": "vix",
            "threshold": 30,
            "level": AlertLevel.WARNING,
            "message": "VIX above 30 - high volatility environment"
        }
    ])
    
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check metrics against alert thresholds"""
        triggered_alerts = []
        for alert in self.alerts:
            metric_value = metrics.get(alert['metric'], 0)
            if metric_value >= alert['threshold']:
                triggered_alerts.append({
                    'level': alert['level'],
                    'message': alert['message'],
                    'value': metric_value,
                    'threshold': alert['threshold']
                })
        return triggered_alerts


@dataclass
class ComprehensiveRiskLimits:
    """Aggregate all risk limits"""
    position_limits: PositionLimits = field(default_factory=PositionLimits)
    portfolio_limits: PortfolioLimits = field(default_factory=PortfolioLimits)
    drawdown_limits: DrawdownLimits = field(default_factory=DrawdownLimits)
    volatility_limits: VolatilityLimits = field(default_factory=VolatilityLimits)
    time_decay_limits: TimeDecayLimits = field(default_factory=TimeDecayLimits)
    correlation_limits: CorrelationLimits = field(default_factory=CorrelationLimits)
    margin_requirements: MarginRequirements = field(default_factory=MarginRequirements)
    risk_alerts: RiskAlerts = field(default_factory=RiskAlerts)
    
    def get_risk_level_multiplier(self, risk_level: RiskLevel) -> float:
        """Get position size multiplier based on risk level"""
        multipliers = {
            RiskLevel.LOW: 0.5,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.HIGH: 1.5,
            RiskLevel.AGGRESSIVE: 2.0
        }
        return multipliers.get(risk_level, 1.0)
    
    def adjust_limits_for_risk_level(self, risk_level: RiskLevel):
        """Adjust all limits based on risk level"""
        multiplier = self.get_risk_level_multiplier(risk_level)
        
        # Adjust position limits
        self.position_limits.max_contracts_per_position = int(
            self.position_limits.max_contracts_per_position * multiplier
        )
        self.position_limits.max_position_value *= multiplier
        
        # Adjust portfolio limits
        self.portfolio_limits.max_portfolio_value *= multiplier
        self.portfolio_limits.max_leverage *= multiplier
        
        # Adjust drawdown limits (inverse relationship)
        self.drawdown_limits.max_daily_drawdown *= (2 - multiplier)
        self.drawdown_limits.max_total_drawdown *= (2 - multiplier)