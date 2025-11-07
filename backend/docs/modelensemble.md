"""
Model Ensemble Strategy for Options Trading
==========================================

This strategy uses all four pricing models (Black-Scholes, Merton Jump Diffusion, 
Heston, ML Neural Network) working together to:

1. Identify mispriced options through model consensus/disagreement
2. Select optimal underlying assets based on model fit quality
3. Dynamically choose the best trading strategy based on market conditions
4. Optimize position sizing using model confidence metrics

The ensemble approach provides superior edge detection and risk management
by leveraging the strengths of each pricing model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Import our pricing models
from models.pricing.black_scholes import BlackScholesModel
from models.pricing.merton_jump import MertonJumpDiffusion  
from models.pricing.heston import HestonModel
from models.pricing.ml_pricing import MLOptionsPricer

class MarketRegime(Enum):
    CALM = "calm"
    VOLATILE = "volatile"
    TRENDING = "trending"
    CRISIS = "crisis"

class StrategyType(Enum):
    VOLATILITY_ARBITRAGE = "vol_arb"
    GAMMA_SCALPING = "gamma_scalp"
    DISPERSION = "dispersion"
    DIRECTIONAL = "directional"
    MARKET_MAKING = "market_making"

@dataclass
class ModelPrediction:
    """Individual model prediction with confidence metrics"""
    model_name: str
    predicted_price: float
    confidence: float
    implied_vol: float
    greeks: Dict[str, float]
    fit_quality: float  # How well model fits recent data
    calibration_error: float

@dataclass
class EnsemblePrediction:
    """Combined prediction from all models"""
    symbol: str
    strike: float
    expiry: str
    option_type: str
    market_price: float
    consensus_price: float
    price_std: float
    model_disagreement: float
    best_model: str
    edge_magnitude: float
    confidence_score: float
    recommended_strategy: StrategyType

class ModelEnsemble:
    """
    Orchestrates multiple pricing models to provide superior predictions
    and trading signals through model consensus and disagreement analysis.
    """
    
    def __init__(self, risk_level: str = 'moderate'):
        self.risk_level = risk_level
        self.models = self._initialize_models()
        self.model_weights = {
            'black_scholes': 0.2,
            'merton_jump': 0.25,
            'heston': 0.3,
            'ml_neural': 0.25
        }
        self.performance_tracker = {}
        self.calibration_schedule = {}
        
        # Setup logging
        self.logger = logging.getLogger('ModelEnsemble')
        
    def _initialize_models(self) -> Dict:
        """Initialize all four pricing models"""
        return {
            'black_scholes': BlackScholesModel(),
            'merton_jump': MertonJumpDiffusion(max_iterations=40),
            'heston': HestonModel(),
            'ml_neural': MLOptionsPricer()
        }
    
    async def analyze_universe(self, option_chains: Dict[str, pd.DataFrame],
                              market_data: Dict[str, pd.DataFrame]) -> List[EnsemblePrediction]:
        """
        Analyze entire options universe using all models to identify
        best opportunities across all underlyings and strategies.
        """
        self.logger.info(f"Analyzing {len(option_chains)} underlying assets...")
        
        all_predictions = []
        
        # Process each underlying in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = []
            
            for symbol, chain in option_chains.items():
                if symbol in market_data:
                    task = executor.submit(
                        self._analyze_single_underlying,
                        symbol, chain, market_data[symbol]
                    )
                    tasks.append(task)
            
            # Collect results
            for task in tasks:
                symbol_predictions = task.result()
                all_predictions.extend(symbol_predictions)
        
        # Rank all predictions by edge magnitude and confidence
        ranked_predictions = self._rank_predictions(all_predictions)
        
        self.logger.info(f"Found {len(ranked_predictions)} trading opportunities")
        return ranked_predictions
    
    def _analyze_single_underlying(self, symbol: str, option_chain: pd.DataFrame,
                                 market_data: pd.DataFrame) -> List[EnsemblePrediction]:
        """Analyze all options for a single underlying asset"""
        
        # First, determine which models work best for this underlying
        model_fitness = self._evaluate_model_fitness(symbol, option_chain, market_data)
        
        # Detect current market regime for this asset
        regime = self._detect_market_regime(market_data)
        
        predictions = []
        
        # Analyze each option in the chain
        for _, option in option_chain.iterrows():
            try:
                # Get predictions from all models
                model_predictions = self._get_model_predictions(
                    symbol, option, market_data, regime
                )
                
                # Create ensemble prediction
                ensemble_pred = self._create_ensemble_prediction(
                    symbol, option, model_predictions, model_fitness, regime
                )
                
                if ensemble_pred and ensemble_pred.edge_magnitude > 0.02:  # 2% edge minimum
                    predictions.append(ensemble_pred)
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze {symbol} option: {e}")
                continue
        
        return predictions
    
    def _get_model_predictions(self, symbol: str, option: pd.Series,
                             market_data: pd.DataFrame, regime: MarketRegime) -> List[ModelPrediction]:
        """Get predictions from all four pricing models"""
        
        # Extract option parameters
        S = market_data['close'].iloc[-1]
        K = option['strike']
        T = self._calculate_time_to_expiry(option['expiration'])
        r = 0.05  # Risk-free rate (would be dynamic in production)
        market_price = (option['bid'] + option['ask']) / 2
        option_type = option['option_type']
        
        predictions = []
        
        # Black-Scholes prediction
        try:
            bs_iv = self._calculate_implied_volatility(S, K, T, r, market_price, option_type)
            bs_price = self.models['black_scholes'].price(S, K, T, r, bs_iv, option_type)
            bs_greeks = self.models['black_scholes'].calculate_greeks(S, K, T, r, bs_iv)
            
            predictions.append(ModelPrediction(
                model_name='black_scholes',
                predicted_price=bs_price,
                confidence=0.7,  # BS has moderate confidence in all regimes
                implied_vol=bs_iv,
                greeks=bs_greeks,
                fit_quality=self._calculate_bs_fit_quality(symbol, market_data),
                calibration_error=0.0  # BS doesn't require calibration
            ))
        except Exception as e:
            self.logger.warning(f"Black-Scholes failed for {symbol}: {e}")
        
        # Merton Jump Diffusion prediction
        try:
            # Use calibrated or default parameters
            merton_params = self._get_merton_parameters(symbol, regime)
            merton_price = self.models['merton_jump'].price_option(
                S, K, T, r, merton_params, option_type
            )
            
            # Estimate Greeks via finite differences
            merton_greeks = self._calculate_merton_greeks(
                S, K, T, r, merton_params, option_type
            )
            
            # Merton confidence is higher during volatile periods
            merton_confidence = 0.9 if regime in [MarketRegime.VOLATILE, MarketRegime.CRISIS] else 0.6
            
            predictions.append(ModelPrediction(
                model_name='merton_jump',
                predicted_price=merton_price,
                confidence=merton_confidence,
                implied_vol=bs_iv,  # Use BS IV as proxy
                greeks=merton_greeks,
                fit_quality=self._calculate_merton_fit_quality(symbol, market_data),
                calibration_error=self._get_merton_calibration_error(symbol)
            ))
        except Exception as e:
            self.logger.warning(f"Merton Jump Diffusion failed for {symbol}: {e}")
        
        # Heston prediction
        try:
            heston_params = self._get_heston_parameters(symbol, regime)
            heston_price = self.models['heston'].price_option(
                S, K, T, r, heston_params, option_type
            )
            heston_greeks = self.models['heston'].calculate_greeks(
                S, K, T, r, heston_params, option_type
            )
            
            # Heston confidence varies with volatility clustering
            vol_clustering = self._measure_volatility_clustering(market_data)
            heston_confidence = min(0.95, 0.6 + vol_clustering * 0.4)
            
            predictions.append(ModelPrediction(
                model_name='heston',
                predicted_price=heston_price,
                confidence=heston_confidence,
                implied_vol=np.sqrt(heston_params.v0),  # Current vol
                greeks=heston_greeks,
                fit_quality=self._calculate_heston_fit_quality(symbol, market_data),
                calibration_error=self._get_heston_calibration_error(symbol)
            ))
        except Exception as e:
            self.logger.warning(f"Heston failed for {symbol}: {e}")
        
        # ML Neural Network prediction
        try:
            ml_price = self.models['ml_neural'].predict_price(
                S, K, T, r, bs_iv, option_type
            )
            
            # ML Greeks via finite differences
            ml_greeks = self._calculate_ml_greeks(S, K, T, r, bs_iv, option_type)
            
            # ML confidence based on training data similarity
            ml_confidence = self._calculate_ml_confidence(S, K, T, bs_iv)
            
            predictions.append(ModelPrediction(
                model_name='ml_neural',
                predicted_price=ml_price,
                confidence=ml_confidence,
                implied_vol=bs_iv,
                greeks=ml_greeks,
                fit_quality=self._calculate_ml_fit_quality(symbol),
                calibration_error=0.0  # ML doesn't have traditional calibration error
            ))
        except Exception as e:
            self.logger.warning(f"ML Neural Network failed for {symbol}: {e}")
        
        return predictions
    
    def _create_ensemble_prediction(self, symbol: str, option: pd.Series,
                                  model_predictions: List[ModelPrediction],
                                  model_fitness: Dict[str, float],
                                  regime: MarketRegime) -> Optional[EnsemblePrediction]:
        """Create consensus prediction from all models"""
        
        if len(model_predictions) < 2:
            return None
        
        # Calculate weighted consensus
        total_weight = 0
        consensus_price = 0
        prices = []
        
        for pred in model_predictions:
            # Dynamic weight based on model fitness and confidence
            base_weight = self.model_weights[pred.model_name]
            fitness_weight = model_fitness.get(pred.model_name, 0.5)
            confidence_weight = pred.confidence
            
            # Regime-specific adjustments
            regime_weight = self._get_regime_weight(pred.model_name, regime)
            
            final_weight = base_weight * fitness_weight * confidence_weight * regime_weight
            
            consensus_price += pred.predicted_price * final_weight
            total_weight += final_weight
            prices.append(pred.predicted_price)
        
        if total_weight == 0:
            return None
        
        consensus_price /= total_weight
        
        # Calculate metrics
        market_price = (option['bid'] + option['ask']) / 2
        price_std = np.std(prices)
        model_disagreement = price_std / consensus_price if consensus_price > 0 else 1.0
        
        # Edge calculation (positive = underpriced by market)
        edge_magnitude = (consensus_price - market_price) / market_price
        
        # Overall confidence (higher when models agree and individual confidence high)
        confidence_weights = [p.confidence for p in model_predictions]
        avg_confidence = np.mean(confidence_weights)
        disagreement_penalty = max(0, 1 - model_disagreement * 2)  # Penalty for disagreement
        confidence_score = avg_confidence * disagreement_penalty
        
        # Determine best performing model for this prediction
        best_model = max(model_predictions, key=lambda p: p.confidence).model_name
        
        # Recommend strategy based on edge characteristics and market conditions
        recommended_strategy = self._recommend_strategy(
            edge_magnitude, model_disagreement, regime, model_predictions
        )
        
        return EnsemblePrediction(
            symbol=symbol,
            strike=option['strike'],
            expiry=option['expiration'],
            option_type=option['option_type'],
            market_price=market_price,
            consensus_price=consensus_price,
            price_std=price_std,
            model_disagreement=model_disagreement,
            best_model=best_model,
            edge_magnitude=edge_magnitude,
            confidence_score=confidence_score,
            recommended_strategy=recommended_strategy
        )
    
    def _recommend_strategy(self, edge_magnitude: float, model_disagreement: float,
                          regime: MarketRegime, predictions: List[ModelPrediction]) -> StrategyType:
        """
        Recommend optimal strategy based on ensemble analysis:
        - Large edge + low disagreement = Directional play
        - High disagreement = Volatility arbitrage 
        - Volatile regime + good gamma = Gamma scalping
        - Multiple assets with edge = Dispersion
        """
        
        abs_edge = abs(edge_magnitude)
        
        # Strong directional signal
        if abs_edge > 0.05 and model_disagreement < 0.1:
            return StrategyType.DIRECTIONAL
        
        # High model disagreement suggests volatility mispricing
        elif model_disagreement > 0.15:
            return StrategyType.VOLATILITY_ARBITRAGE
        
        # Volatile regime with good gamma opportunities
        elif regime in [MarketRegime.VOLATILE, MarketRegime.CRISIS]:
            # Check if we have good gamma exposure potential
            avg_gamma = np.mean([abs(p.greeks.get('gamma', 0)) for p in predictions])
            if avg_gamma > 0.01:
                return StrategyType.GAMMA_SCALPING
            else:
                return StrategyType.VOLATILITY_ARBITRAGE
        
        # Calm markets good for market making
        elif regime == MarketRegime.CALM and abs_edge < 0.03:
            return StrategyType.MARKET_MAKING
        
        # Default to volatility arbitrage
        else:
            return StrategyType.VOLATILITY_ARBITRAGE
    
    def _rank_predictions(self, predictions: List[EnsemblePrediction]) -> List[EnsemblePrediction]:
        """
        Rank predictions by attractiveness considering:
        - Edge magnitude
        - Confidence score  
        - Model disagreement (can be good or bad depending on strategy)
        - Risk-adjusted returns
        """
        
        def score_prediction(pred: EnsemblePrediction) -> float:
            # Base score from edge magnitude
            edge_score = abs(pred.edge_magnitude) * 10
            
            # Confidence multiplier
            confidence_multiplier = pred.confidence_score
            
            # Strategy-specific scoring
            if pred.recommended_strategy == StrategyType.DIRECTIONAL:
                # Want high edge, high confidence, low disagreement
                strategy_score = edge_score * confidence_multiplier * (1 - pred.model_disagreement)
                
            elif pred.recommended_strategy == StrategyType.VOLATILITY_ARBITRAGE:
                # Want high disagreement (different models see different values)
                strategy_score = edge_score * confidence_multiplier * (1 + pred.model_disagreement)
                
            elif pred.recommended_strategy == StrategyType.GAMMA_SCALPING:
                # Focus on options with good gamma and edge
                strategy_score = edge_score * confidence_multiplier * 1.2  # Bonus for gamma opportunities
                
            else:
                strategy_score = edge_score * confidence_multiplier
            
            # Apply risk adjustment based on user risk level
            risk_adjustments = {
                'low': 0.7,
                'moderate': 1.0, 
                'high': 1.3
            }
            
            return strategy_score * risk_adjustments[self.risk_level]
        
        # Sort by score descending
        return sorted(predictions, key=score_prediction, reverse=True)
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime for regime-specific model weighting"""
        
        returns = market_data['close'].pct_change().dropna()
        
        # Calculate regime indicators
        recent_vol = returns.tail(20).std() * np.sqrt(252)
        vol_regime = recent_vol > 0.25
        
        trend_strength = abs(returns.tail(20).mean()) / returns.tail(20).std()
        trending = trend_strength > 0.2
        
        crisis_indicators = (returns.tail(5) < -0.05).sum()  # Multiple large down days
        crisis = crisis_indicators >= 2
        
        if crisis:
            return MarketRegime.CRISIS
        elif vol_regime:
            return MarketRegime.VOLATILE
        elif trending:
            return MarketRegime.TRENDING
        else:
            return MarketRegime.CALM
    
    def _get_regime_weight(self, model_name: str, regime: MarketRegime) -> float:
        """Adjust model weights based on market regime"""
        
        regime_weights = {
            MarketRegime.CALM: {
                'black_scholes': 1.2,    # BS works well in calm markets
                'merton_jump': 0.8,      # Jump models less important
                'heston': 1.0,
                'ml_neural': 1.1         # ML good at pattern recognition
            },
            MarketRegime.VOLATILE: {
                'black_scholes': 0.7,    # BS underperforms in volatile markets
                'merton_jump': 1.3,      # Jump models shine
                'heston': 1.2,           # Stochastic vol important
                'ml_neural': 0.9
            },
            MarketRegime.TRENDING: {
                'black_scholes': 0.9,
                'merton_jump': 1.1,
                'heston': 1.0,
                'ml_neural': 1.2         # ML good at trend recognition
            },
            MarketRegime.CRISIS: {
                'black_scholes': 0.5,    # BS breaks down in crisis
                'merton_jump': 1.5,      # Jumps very important
                'heston': 1.1,
                'ml_neural': 0.8         # ML may not have crisis training data
            }
        }
        
        return regime_weights[regime].get(model_name, 1.0)
    
    def generate_portfolio_recommendations(self, predictions: List[EnsemblePrediction],
                                         portfolio_capital: float) -> List[Dict]:
        """
        Generate final trading recommendations by considering:
        - Portfolio diversification
        - Risk limits
        - Capital allocation
        - Strategy complementarity
        """
        
        recommendations = []
        allocated_capital = 0
        max_capital_per_trade = portfolio_capital * 0.1  # 10% max per trade
        
        # Group by strategy type for balanced allocation
        strategy_groups = {}
        for pred in predictions[:50]:  # Top 50 opportunities
            strategy = pred.recommended_strategy
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(pred)
        
        # Allocate capital across strategies
        for strategy_type, strategy_preds in strategy_groups.items():
            if allocated_capital >= portfolio_capital * 0.8:  # 80% max utilization
                break
            
            strategy_capital = min(
                portfolio_capital * 0.3,  # Max 30% per strategy type
                portfolio_capital * 0.8 - allocated_capital
            )
            
            # Select best predictions within this strategy
            for pred in strategy_preds[:5]:  # Max 5 positions per strategy
                if strategy_capital <= 0:
                    break
                
                position_size = min(
                    max_capital_per_trade,
                    strategy_capital * (pred.confidence_score / 1.0),
                    strategy_capital * 0.4  # Max 40% of strategy allocation
                )
                
                if position_size < portfolio_capital * 0.01:  # Min 1% position
                    continue
                
                recommendation = self._create_trade_recommendation(pred, position_size)
                recommendations.append(recommendation)
                
                allocated_capital += position_size
                strategy_capital -= position_size
        
        self.logger.info(f"Generated {len(recommendations)} portfolio recommendations")
        self.logger.info(f"Total allocated capital: ${allocated_capital:,.2f} ({allocated_capital/portfolio_capital:.1%})")
        
        return recommendations
    
    def _create_trade_recommendation(self, prediction: EnsemblePrediction, 
                                   position_size: float) -> Dict:
        """Create specific trade recommendation with entry/exit rules"""
        
        return {
            'symbol': prediction.symbol,
            'strategy_type': prediction.recommended_strategy.value,
            'option_details': {
                'strike': prediction.strike,
                'expiry': prediction.expiry,
                'option_type': prediction.option_type
            },
            'market_analysis': {
                'market_price': prediction.market_price,
                'fair_value': prediction.consensus_price,
                'edge_magnitude': prediction.edge_magnitude,
                'confidence': prediction.confidence_score,
                'model_disagreement': prediction.model_disagreement,
                'best_model': prediction.best_model
            },
            'position_sizing': {
                'recommended_size': position_size,
                'max_risk': position_size * 0.5,  # Max 50% loss
                'contracts': int(position_size / (prediction.market_price * 100))
            },
            'entry_rules': {
                'limit_price': prediction.market_price * 1.02,  # 2% above fair value
                'time_limit': '1_day',  # Cancel if not filled in 1 day
                'volume_threshold': 10  # Minimum volume requirement
            },
            'exit_rules': {
                'profit_target': abs(prediction.edge_magnitude) * 0.7,  # 70% of expected edge
                'stop_loss': -0.25,  # 25% loss
                'time_decay_exit': '50_dte',  # Exit at 50 days to expiry
                'volatility_exit': {
                    'iv_change_threshold': 0.15,  # Exit if IV changes >15%
                    'direction': 'any'
                }
            },
            'risk_metrics': {
                'max_drawdown': position_size * 0.3,
                'var_contribution': position_size * 0.15,
                'correlation_limit': 0.6  # Max correlation with other positions
            },
            'monitoring': {
                'rebalance_frequency': 'daily',
                'greek_limits': {
                    'delta': 0.1,  # Maintain delta neutral
                    'gamma': None,  # No gamma limits for scalping
                    'vega': 0.2    # Moderate vega exposure
                }
            }
        }
    
    # Helper methods for model calibration and fitness evaluation
    def _evaluate_model_fitness(self, symbol: str, option_chain: pd.DataFrame,
                               market_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate how well each model fits recent market data"""
        # Implementation would analyze recent pricing errors for each model
        # Return fitness scores 0-1 for each model
        return {
            'black_scholes': 0.7,
            'merton_jump': 0.8,
            'heston': 0.75,
            'ml_neural': 0.85
        }
    
    def _calculate_time_to_expiry(self, expiration_date: str) -> float:
        """Calculate time to expiry in years"""
        # Implementation would parse date and calculate years
        return 0.25  # Placeholder: 3 months
    
    def _calculate_implied_volatility(self, S: float, K: float, T: float, 
                                    r: float, price: float, option_type: str) -> float:
        """Calculate implied volatility from market price"""
        # Implementation would use numerical methods to solve for IV
        return 0.25  # Placeholder: 25% volatility
    
    # Additional helper methods would be implemented for:
    # - Model parameter retrieval/calibration
    # - Greeks calculations
    # - Fit quality metrics
    # - Calibration error tracking
    # - ML confidence scoring
    # - Volatility clustering measurement

# Example usage showing how all models work together
async def example_model_ensemble_usage():
    """Example of how to use the model ensemble for trading"""
    
    # Initialize ensemble
    ensemble = ModelEnsemble(risk_level='moderate')
    
    # Sample data (in production would come from market data feeds)
    option_chains = {
        'AAPL': pd.DataFrame({
            'strike': [150, 155, 160, 165, 170],
            'expiration': ['2024-03-15'] * 5,
            'option_type': ['call'] * 5,
            'bid': [5.20, 2.30, 0.90, 0.25, 0.05],
            'ask': [5.40, 2.50, 1.10, 0.35, 0.15]
        }),
        'TSLA': pd.DataFrame({
            'strike': [200, 210, 220, 230, 240], 
            'expiration': ['2024-03-15'] * 5,
            'option_type': ['call'] * 5,
            'bid': [8.50, 4.20, 1.80, 0.60, 0.15],
            'ask': [8.80, 4.50, 2.00, 0.80, 0.25]
        })
    }
    
    market_data = {
        'AAPL': pd.DataFrame({
            'close': [158.50] * 20,  # Would be historical prices
            'volume': [50000000] * 20
        }),
        'TSLA': pd.DataFrame({
            'close': [215.75] * 20,
            'volume': [25000000] * 20
        })
    }
    
    # Analyze entire universe
    predictions = await ensemble.analyze_universe(option_chains, market_data)
    
    # Generate portfolio recommendations
    portfolio_capital = 100000  # $100k portfolio
    recommendations = ensemble.generate_portfolio_recommendations(predictions, portfolio_capital)
    
    # Print top recommendations
    print("Top Trading Recommendations:")
    print("=" * 50)
    
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec['symbol']} - {rec['strategy_type']}")
        print(f"   Strike: {rec['option_details']['strike']}")
        print(f"   Edge: {rec['market_analysis']['edge_magnitude']:.1%}")
        print(f"   Confidence: {rec['market_analysis']['confidence']:.2f}")
        print(f"   Position Size: ${rec['position_sizing']['recommended_size']:,.0f}")
        print(f"   Best Model: {rec['market_analysis']['best_model']}")
        print()

if __name__ == "__main__":
    asyncio.run(example_model_ensemble_usage())

    