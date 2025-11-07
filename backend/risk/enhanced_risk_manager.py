"""
Enhanced Risk Management for Model Ensemble
============================================

Advanced risk management system that understands model ensemble predictions and
provides intelligent position sizing and risk controls based on:
- Model consensus and disagreement
- Market regime detection
- Portfolio correlations
- Dynamic risk adjustments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from config.risk_limits import RiskLevel, AlertLevel, PositionLimits
from strategies.model_ensemble import MarketRegime, StrategyType

@dataclass
class PortfolioRiskMetrics:
    """Current portfolio risk metrics"""
    total_value: float
    total_delta: float
    total_gamma: float
    total_vega: float
    total_theta: float
    var_95: float  # 95% Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    correlation_matrix: np.ndarray
    concentration_risk: float

@dataclass
class PositionRiskScore:
    """Risk score for a single position"""
    model_risk: float  # Risk from model disagreement
    market_risk: float  # Risk from market conditions
    portfolio_risk: float  # Risk from portfolio correlation
    liquidity_risk: float  # Risk from liquidity constraints
    total_risk: float  # Combined risk score
    risk_adjusted_size: float  # Adjusted position size

class EnhancedRiskManager:
    """
    Extended risk manager that understands model ensemble predictions
    """

    def __init__(self, base_risk_level: str = 'moderate'):
        self.base_risk_level = base_risk_level
        self.position_limits = self._get_position_limits(base_risk_level)

        # Model-specific risk parameters
        self.model_correlations = {}
        self.model_performance_history = {}
        self.calibration_timestamps = {}

        # Portfolio tracking
        self.current_positions = {}
        self.position_correlations = {}
        self.historical_returns = []

        # Risk thresholds
        self.max_portfolio_risk = self._get_max_portfolio_risk(base_risk_level)
        self.max_model_disagreement = 0.3  # 30% disagreement
        self.min_model_confidence = 0.5  # 50% minimum confidence
        self.max_correlation = 0.7  # 70% max correlation between positions

        # Dynamic adjustments
        self.volatility_multiplier = 1.0
        self.regime_adjustments = {
            MarketRegime.CALM: 1.2,
            MarketRegime.TRENDING: 1.0,
            MarketRegime.VOLATILE: 0.8,
            MarketRegime.CRISIS: 0.5
        }

        # Setup logging
        self.logger = logging.getLogger('EnhancedRiskManager')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _get_position_limits(self, risk_level: str) -> PositionLimits:
        """Get position limits based on risk level"""

        limits_config = {
            'low': PositionLimits(
                max_contracts_per_position=50,
                max_position_value=25000,
                max_percent_of_portfolio=0.15,
                stop_loss_percent=0.30
            ),
            'moderate': PositionLimits(
                max_contracts_per_position=100,
                max_position_value=50000,
                max_percent_of_portfolio=0.25,
                stop_loss_percent=0.50
            ),
            'high': PositionLimits(
                max_contracts_per_position=200,
                max_position_value=100000,
                max_percent_of_portfolio=0.35,
                stop_loss_percent=0.70
            )
        }

        return limits_config.get(risk_level, limits_config['moderate'])

    def _get_max_portfolio_risk(self, risk_level: str) -> float:
        """Get maximum portfolio risk based on risk level"""

        risk_limits = {
            'low': 0.02,  # 2% daily VaR
            'moderate': 0.05,  # 5% daily VaR
            'high': 0.10  # 10% daily VaR
        }

        return risk_limits.get(risk_level, 0.05)

    async def evaluate_ensemble_recommendations(self, recommendations: List[Dict],
                                              current_positions: Dict) -> List[Dict]:
        """
        Enhanced evaluation considering model consensus and disagreement
        """

        self.current_positions = current_positions
        approved_trades = []

        # Calculate current portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(current_positions)

        for rec in recommendations:
            # Traditional risk checks
            if not self._passes_basic_risk_checks(rec, current_positions):
                continue

            # Model-specific risk evaluation
            risk_score = self._calculate_position_risk_score(rec, portfolio_metrics)

            if risk_score.total_risk > 0.8:  # High risk threshold
                self.logger.warning(f"High risk for {rec['symbol']}: {risk_score.total_risk:.2f}")
                continue

            # Portfolio correlation analysis
            if self._would_exceed_correlation_limits(rec, current_positions):
                self.logger.info(f"Correlation limit exceeded for {rec['symbol']}")
                continue

            # Model disagreement analysis
            disagreement = rec['market_analysis']['model_disagreement']
            confidence = rec['market_analysis']['confidence']

            # Adjust position size based on risk score
            adjusted_size = risk_score.risk_adjusted_size

            if adjusted_size < self._get_min_position_size():
                continue

            # Update recommendation with risk adjustments
            rec['position_sizing']['risk_adjusted_size'] = adjusted_size
            rec['risk_assessment'] = {
                'model_risk': risk_score.model_risk,
                'market_risk': risk_score.market_risk,
                'portfolio_risk': risk_score.portfolio_risk,
                'liquidity_risk': risk_score.liquidity_risk,
                'total_risk': risk_score.total_risk,
                'risk_level': self._categorize_risk_level(risk_score.total_risk)
            }

            approved_trades.append(rec)

        # Final portfolio-level validation
        approved_trades = self._final_portfolio_validation(
            approved_trades, current_positions, portfolio_metrics
        )

        self.logger.info(f"Approved {len(approved_trades)} of {len(recommendations)} recommendations")
        return approved_trades

    def _calculate_position_risk_score(self, recommendation: Dict,
                                      portfolio_metrics: PortfolioRiskMetrics) -> PositionRiskScore:
        """Calculate comprehensive risk score for a position"""

        # Model risk from disagreement and confidence
        model_risk = self._calculate_model_risk(recommendation)

        # Market risk from current regime and volatility
        market_risk = self._calculate_market_risk(recommendation)

        # Portfolio risk from correlation and concentration
        portfolio_risk = self._calculate_portfolio_risk(recommendation, portfolio_metrics)

        # Liquidity risk from volume and spread
        liquidity_risk = self._calculate_liquidity_risk(recommendation)

        # Combined risk score (weighted average)
        total_risk = (
            model_risk * 0.3 +
            market_risk * 0.3 +
            portfolio_risk * 0.25 +
            liquidity_risk * 0.15
        )

        # Calculate risk-adjusted position size
        base_size = recommendation['position_sizing']['recommended_size']
        risk_multiplier = max(0.2, 1.0 - total_risk)
        risk_adjusted_size = base_size * risk_multiplier

        return PositionRiskScore(
            model_risk=model_risk,
            market_risk=market_risk,
            portfolio_risk=portfolio_risk,
            liquidity_risk=liquidity_risk,
            total_risk=total_risk,
            risk_adjusted_size=risk_adjusted_size
        )

    def _calculate_model_risk(self, recommendation: Dict) -> float:
        """Calculate risk from model behavior"""

        # Base risk from confidence
        base_risk = 1 - recommendation['market_analysis']['confidence']

        # Model disagreement risk
        disagreement = recommendation['market_analysis']['model_disagreement']
        strategy = recommendation['strategy_type']

        if strategy == 'vol_arb':
            # High disagreement is good for vol arb
            disagreement_risk = max(0, 0.3 - disagreement)
        else:
            # High disagreement is bad for directional strategies
            disagreement_risk = disagreement * 0.7

        # Historical model performance risk
        best_model = recommendation['market_analysis']['best_model']
        model_perf = self.model_performance_history.get(best_model, 0.5)
        performance_risk = 1 - model_perf

        # Calibration freshness risk
        if best_model in ['heston', 'merton_jump']:
            calibration_age = self._get_calibration_age(best_model)
            calibration_risk = min(0.5, calibration_age / 30)  # Max 50% risk at 30 days
        else:
            calibration_risk = 0

        # Combined model risk
        model_risk = (
            base_risk * 0.4 +
            disagreement_risk * 0.3 +
            performance_risk * 0.2 +
            calibration_risk * 0.1
        )

        return min(1.0, model_risk)

    def _calculate_market_risk(self, recommendation: Dict) -> float:
        """Calculate risk from market conditions"""

        # Get current market metrics (simplified)
        current_vol = 0.25  # Would get from market data
        historical_vol = 0.20
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0

        # Base market risk from volatility
        vol_risk = min(1.0, (vol_ratio - 1.0) * 0.5) if vol_ratio > 1 else 0

        # Time decay risk for options
        # Would calculate from actual expiry
        time_to_expiry = 30  # days
        if time_to_expiry < 7:
            time_risk = 0.8
        elif time_to_expiry < 30:
            time_risk = 0.3
        else:
            time_risk = 0.1

        # Strategy-specific market risk
        strategy = recommendation['strategy_type']
        if strategy == 'directional':
            strategy_risk = vol_risk * 1.5  # More sensitive to volatility
        elif strategy == 'vol_arb':
            strategy_risk = 0.3  # Less sensitive to market direction
        else:
            strategy_risk = 0.5

        # Combined market risk
        market_risk = (
            vol_risk * 0.4 +
            time_risk * 0.3 +
            strategy_risk * 0.3
        )

        return min(1.0, market_risk)

    def _calculate_portfolio_risk(self, recommendation: Dict,
                                 portfolio_metrics: PortfolioRiskMetrics) -> float:
        """Calculate risk from portfolio perspective"""

        # Concentration risk
        position_size = recommendation['position_sizing']['recommended_size']
        portfolio_value = portfolio_metrics.total_value if portfolio_metrics.total_value > 0 else 100000
        concentration = position_size / portfolio_value
        concentration_risk = min(1.0, concentration * 4)  # Risk increases rapidly above 25%

        # Correlation risk (simplified - would calculate actual correlations)
        symbol = recommendation['symbol']
        avg_correlation = self._calculate_average_correlation(symbol)
        correlation_risk = avg_correlation

        # Greeks concentration risk
        # Would calculate actual Greek exposures
        greek_risk = 0.3  # Placeholder

        # VaR contribution
        var_contribution = position_size * 0.15  # Simplified
        var_ratio = var_contribution / (portfolio_value * self.max_portfolio_risk)
        var_risk = min(1.0, var_ratio)

        # Combined portfolio risk
        portfolio_risk = (
            concentration_risk * 0.3 +
            correlation_risk * 0.3 +
            greek_risk * 0.2 +
            var_risk * 0.2
        )

        return min(1.0, portfolio_risk)

    def _calculate_liquidity_risk(self, recommendation: Dict) -> float:
        """Calculate liquidity risk"""

        # Would get actual market data
        bid_ask_spread = 0.05  # 5% spread
        daily_volume = 1000  # contracts
        position_size = recommendation['position_sizing'].get('contracts', 10)

        # Spread risk
        if bid_ask_spread < 0.02:
            spread_risk = 0.1
        elif bid_ask_spread < 0.05:
            spread_risk = 0.3
        elif bid_ask_spread < 0.10:
            spread_risk = 0.6
        else:
            spread_risk = 0.9

        # Volume risk
        volume_ratio = position_size / daily_volume if daily_volume > 0 else 1.0
        if volume_ratio < 0.01:
            volume_risk = 0.1
        elif volume_ratio < 0.05:
            volume_risk = 0.3
        elif volume_ratio < 0.10:
            volume_risk = 0.6
        else:
            volume_risk = 0.9

        # Combined liquidity risk
        liquidity_risk = (spread_risk * 0.5 + volume_risk * 0.5)

        return min(1.0, liquidity_risk)

    def _passes_basic_risk_checks(self, recommendation: Dict,
                                 current_positions: Dict) -> bool:
        """Perform basic risk limit checks"""

        # Check position limits
        position_value = recommendation['position_sizing']['recommended_size']
        if position_value > self.position_limits.max_position_value:
            return False

        # Check portfolio concentration
        total_portfolio = sum(p.get('value', 0) for p in current_positions.values())
        if total_portfolio > 0:
            concentration = position_value / total_portfolio
            if concentration > self.position_limits.max_percent_of_portfolio:
                return False

        # Check max positions
        if len(current_positions) >= 20:  # Max 20 positions
            return False

        return True

    def _would_exceed_correlation_limits(self, recommendation: Dict,
                                        current_positions: Dict) -> bool:
        """Check if adding position would exceed correlation limits"""

        symbol = recommendation['symbol']

        # Calculate correlations with existing positions
        high_correlations = 0
        for pos_symbol in current_positions:
            correlation = self._get_correlation(symbol, pos_symbol)
            if correlation > self.max_correlation:
                high_correlations += 1

        # Don't allow more than 3 highly correlated positions
        return high_correlations > 3

    def _final_portfolio_validation(self, approved_trades: List[Dict],
                                   current_positions: Dict,
                                   portfolio_metrics: PortfolioRiskMetrics) -> List[Dict]:
        """Final portfolio-level risk validation"""

        # Calculate pro-forma portfolio metrics
        proforma_metrics = self._calculate_proforma_metrics(
            approved_trades, current_positions, portfolio_metrics
        )

        # Check if portfolio VaR would exceed limits
        if proforma_metrics['var_95'] > self.max_portfolio_risk:
            # Remove lowest confidence trades until within limits
            approved_trades.sort(key=lambda x: x['market_analysis']['confidence'], reverse=True)

            while approved_trades and proforma_metrics['var_95'] > self.max_portfolio_risk:
                approved_trades.pop()
                proforma_metrics = self._calculate_proforma_metrics(
                    approved_trades, current_positions, portfolio_metrics
                )

        # Ensure minimum diversification
        if len(approved_trades) > 10:
            # Keep only top 10 to maintain focus
            approved_trades = approved_trades[:10]

        return approved_trades

    def _calculate_portfolio_metrics(self, positions: Dict) -> PortfolioRiskMetrics:
        """Calculate current portfolio risk metrics"""

        # Simplified implementation - would calculate actual metrics
        total_value = sum(p.get('value', 0) for p in positions.values())

        return PortfolioRiskMetrics(
            total_value=total_value,
            total_delta=sum(p.get('delta', 0) for p in positions.values()),
            total_gamma=sum(p.get('gamma', 0) for p in positions.values()),
            total_vega=sum(p.get('vega', 0) for p in positions.values()),
            total_theta=sum(p.get('theta', 0) for p in positions.values()),
            var_95=total_value * 0.05,  # Simplified 5% VaR
            max_drawdown=0.1,
            sharpe_ratio=1.5,
            correlation_matrix=np.eye(len(positions)),
            concentration_risk=0.3
        )

    def _calculate_proforma_metrics(self, new_trades: List[Dict],
                                   current_positions: Dict,
                                   current_metrics: PortfolioRiskMetrics) -> Dict:
        """Calculate pro-forma portfolio metrics with new trades"""

        # Add new trade values to current metrics
        new_value = sum(t['position_sizing']['risk_adjusted_size'] for t in new_trades)
        total_value = current_metrics.total_value + new_value

        # Simplified VaR calculation
        # Would use actual portfolio VaR methodology
        position_vars = []
        for trade in new_trades:
            position_var = trade['position_sizing']['risk_adjusted_size'] * 0.15
            position_vars.append(position_var)

        # Assume some correlation between positions
        correlation_factor = 0.5
        new_var = np.sqrt(sum(v**2 for v in position_vars)) * correlation_factor
        total_var = np.sqrt(current_metrics.var_95**2 + new_var**2)

        return {
            'total_value': total_value,
            'var_95': total_var / total_value if total_value > 0 else 0
        }

    def _get_calibration_age(self, model_name: str) -> float:
        """Get days since last calibration"""

        if model_name in self.calibration_timestamps:
            last_calibration = self.calibration_timestamps[model_name]
            age = (datetime.now() - last_calibration).days
            return age
        return 30  # Default to old if unknown

    def _calculate_average_correlation(self, symbol: str) -> float:
        """Calculate average correlation with existing positions"""

        if not self.current_positions:
            return 0

        correlations = []
        for pos_symbol in self.current_positions:
            corr = self._get_correlation(symbol, pos_symbol)
            correlations.append(corr)

        return np.mean(correlations) if correlations else 0

    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""

        # Simplified - would use actual correlation matrix
        if symbol1 == symbol2:
            return 1.0

        # Sector correlations (simplified)
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
        financials = ['JPM', 'BAC', 'GS', 'WFC']

        if symbol1 in tech_stocks and symbol2 in tech_stocks:
            return 0.7
        elif symbol1 in financials and symbol2 in financials:
            return 0.6
        else:
            return 0.3

    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk score into levels"""

        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.5:
            return "moderate"
        elif risk_score < 0.7:
            return "high"
        else:
            return "very_high"

    def _get_min_position_size(self) -> float:
        """Get minimum position size"""

        min_sizes = {
            'low': 1000,
            'moderate': 500,
            'high': 250
        }

        return min_sizes.get(self.base_risk_level, 500)

    def update_model_performance(self, model_name: str, performance: float):
        """Update model performance history"""

        self.model_performance_history[model_name] = performance
        self.logger.info(f"Updated {model_name} performance: {performance:.2%}")

    def update_calibration_timestamp(self, model_name: str):
        """Update calibration timestamp for a model"""

        self.calibration_timestamps[model_name] = datetime.now()
        self.logger.info(f"Updated {model_name} calibration timestamp")

    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""

        portfolio_metrics = self._calculate_portfolio_metrics(self.current_positions)

        report = {
            'timestamp': datetime.now().isoformat(),
            'risk_level': self.base_risk_level,
            'portfolio_metrics': {
                'total_value': portfolio_metrics.total_value,
                'var_95': portfolio_metrics.var_95,
                'max_drawdown': portfolio_metrics.max_drawdown,
                'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                'concentration_risk': portfolio_metrics.concentration_risk
            },
            'greek_exposures': {
                'delta': portfolio_metrics.total_delta,
                'gamma': portfolio_metrics.total_gamma,
                'vega': portfolio_metrics.total_vega,
                'theta': portfolio_metrics.total_theta
            },
            'position_count': len(self.current_positions),
            'model_performance': self.model_performance_history,
            'risk_limits': {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_position_value': self.position_limits.max_position_value,
                'max_correlation': self.max_correlation
            }
        }

        return report


# Example usage
async def example_risk_management():
    """Example of using enhanced risk manager"""

    risk_manager = EnhancedRiskManager(base_risk_level='moderate')

    # Sample recommendations from model ensemble
    recommendations = [
        {
            'symbol': 'AAPL',
            'strategy_type': 'directional',
            'market_analysis': {
                'confidence': 0.8,
                'model_disagreement': 0.1,
                'best_model': 'heston'
            },
            'position_sizing': {
                'recommended_size': 10000,
                'contracts': 10
            }
        },
        {
            'symbol': 'TSLA',
            'strategy_type': 'vol_arb',
            'market_analysis': {
                'confidence': 0.6,
                'model_disagreement': 0.3,
                'best_model': 'merton_jump'
            },
            'position_sizing': {
                'recommended_size': 15000,
                'contracts': 15
            }
        }
    ]

    # Current positions
    current_positions = {
        'MSFT': {'value': 20000, 'delta': 50, 'gamma': 10, 'vega': 100, 'theta': -50}
    }

    # Evaluate recommendations
    approved = await risk_manager.evaluate_ensemble_recommendations(
        recommendations, current_positions
    )

    print("Risk Management Results:")
    print("=" * 50)
    print(f"Approved {len(approved)} of {len(recommendations)} trades")

    for trade in approved:
        print(f"\n{trade['symbol']}:")
        print(f"  Original Size: ${trade['position_sizing']['recommended_size']:,.0f}")
        print(f"  Risk-Adjusted Size: ${trade['position_sizing']['risk_adjusted_size']:,.0f}")
        print(f"  Risk Assessment: {trade['risk_assessment']}")

    # Generate risk report
    report = risk_manager.generate_risk_report()
    print(f"\nPortfolio Risk Report:")
    print(f"  VaR (95%): ${report['portfolio_metrics']['var_95']:,.0f}")
    print(f"  Greeks: {report['greek_exposures']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_risk_management())