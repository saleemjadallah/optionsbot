"""
Model Performance Tracking System
==================================

Track and analyze performance of individual pricing models to:
- Monitor prediction accuracy over time
- Automatically rebalance model weights based on performance
- Identify which models work best in different market conditions
- Provide detailed performance analytics for each model
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import asyncio
from collections import defaultdict

from strategies.model_ensemble import EnsemblePrediction, MarketRegime

@dataclass
class PredictionRecord:
    """Record of a single prediction for tracking"""
    prediction_id: str
    timestamp: datetime
    symbol: str
    strike: float
    expiry: str
    option_type: str
    market_price: float
    consensus_price: float
    model_predictions: Dict[str, float]
    market_regime: str
    actual_outcome: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    accuracy_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelMetrics:
    """Performance metrics for a single model"""
    model_name: str
    total_predictions: int
    accurate_predictions: int
    accuracy_rate: float
    mean_absolute_error: float
    mean_squared_error: float
    directional_accuracy: float
    regime_performance: Dict[str, float]
    recent_performance: float
    calibration_quality: float

class ModelPerformanceTracker:
    """
    Track and analyze performance of individual pricing models
    """

    def __init__(self, data_dir: str = "data/model_performance"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.performance_db = {}  # In-memory database
        self.prediction_history = {}
        self.model_weights = {
            'black_scholes': 0.25,
            'merton_jump': 0.25,
            'heston': 0.25,
            'ml_neural': 0.25
        }
        self.rebalance_threshold = 0.1  # 10% performance difference triggers reweight
        self.min_predictions_for_rebalance = 20

        # Performance tracking windows
        self.short_window = 7  # days
        self.medium_window = 30  # days
        self.long_window = 90  # days

        # Setup logging
        self.logger = logging.getLogger('ModelPerformanceTracker')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Load historical data if exists
        self._load_historical_data()

    async def track_prediction(self, prediction: EnsemblePrediction,
                              model_predictions: Dict[str, float],
                              market_regime: MarketRegime):
        """Track a new prediction for later evaluation"""

        prediction_id = f"{prediction.symbol}_{prediction.strike}_{prediction.expiry}_{datetime.now().timestamp()}"

        record = PredictionRecord(
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            symbol=prediction.symbol,
            strike=prediction.strike,
            expiry=prediction.expiry,
            option_type=prediction.option_type,
            market_price=prediction.market_price,
            consensus_price=prediction.consensus_price,
            model_predictions=model_predictions,
            market_regime=market_regime.value,
            actual_outcome=None,
            outcome_timestamp=None,
            accuracy_scores={}
        )

        self.prediction_history[prediction_id] = record

        # Save to disk periodically
        if len(self.prediction_history) % 100 == 0:
            self._save_predictions()

    async def update_outcome(self, symbol: str, strike: float, expiry: str,
                           actual_price: float):
        """Update actual outcome for a prediction"""

        # Find matching predictions
        for pred_id, record in self.prediction_history.items():
            if (record.symbol == symbol and
                record.strike == strike and
                record.expiry == expiry and
                record.actual_outcome is None):

                record.actual_outcome = actual_price
                record.outcome_timestamp = datetime.now()

                # Calculate accuracy scores for each model
                for model_name, predicted_price in record.model_predictions.items():
                    error = abs(predicted_price - actual_price) / actual_price if actual_price > 0 else 1.0
                    accuracy = max(0, 1 - error)
                    record.accuracy_scores[model_name] = accuracy

                self.logger.info(f"Updated outcome for {symbol} {strike} {expiry}: {actual_price}")

    async def update_performance_metrics(self) -> Dict[str, ModelMetrics]:
        """Calculate current performance metrics for all models"""

        metrics = {}

        for model_name in ['black_scholes', 'merton_jump', 'heston', 'ml_neural']:
            metrics[model_name] = self._calculate_model_metrics(model_name)

        # Check if rebalancing is needed
        if self._should_rebalance_weights(metrics):
            new_weights = self._calculate_new_weights(metrics)
            await self._update_model_weights(new_weights)

        return metrics

    def _calculate_model_metrics(self, model_name: str) -> ModelMetrics:
        """Calculate detailed metrics for a specific model"""

        # Filter predictions with outcomes
        completed_predictions = [
            record for record in self.prediction_history.values()
            if record.actual_outcome is not None and model_name in record.model_predictions
        ]

        if not completed_predictions:
            return ModelMetrics(
                model_name=model_name,
                total_predictions=0,
                accurate_predictions=0,
                accuracy_rate=0.5,
                mean_absolute_error=0,
                mean_squared_error=0,
                directional_accuracy=0.5,
                regime_performance={},
                recent_performance=0.5,
                calibration_quality=0.5
            )

        # Calculate metrics
        errors = []
        directional_correct = 0
        regime_accuracy = defaultdict(list)

        for record in completed_predictions:
            predicted = record.model_predictions[model_name]
            actual = record.actual_outcome
            market = record.market_price

            # Absolute and squared errors
            error = abs(predicted - actual)
            errors.append(error)

            # Directional accuracy
            predicted_direction = predicted > market
            actual_direction = actual > market
            if predicted_direction == actual_direction:
                directional_correct += 1

            # Regime-specific accuracy
            if model_name in record.accuracy_scores:
                regime_accuracy[record.market_regime].append(record.accuracy_scores[model_name])

        # Recent performance (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_predictions = [
            record for record in completed_predictions
            if record.outcome_timestamp and record.outcome_timestamp > recent_cutoff
        ]

        recent_performance = 0.5
        if recent_predictions:
            recent_accuracies = [
                record.accuracy_scores.get(model_name, 0.5)
                for record in recent_predictions
                if model_name in record.accuracy_scores
            ]
            if recent_accuracies:
                recent_performance = np.mean(recent_accuracies)

        # Aggregate regime performance
        regime_performance = {}
        for regime, accuracies in regime_accuracy.items():
            if accuracies:
                regime_performance[regime] = np.mean(accuracies)

        # Overall accuracy
        accurate_predictions = sum(
            1 for record in completed_predictions
            if model_name in record.accuracy_scores and record.accuracy_scores[model_name] > 0.7
        )

        return ModelMetrics(
            model_name=model_name,
            total_predictions=len(completed_predictions),
            accurate_predictions=accurate_predictions,
            accuracy_rate=accurate_predictions / len(completed_predictions) if completed_predictions else 0.5,
            mean_absolute_error=np.mean(errors) if errors else 0,
            mean_squared_error=np.mean([e**2 for e in errors]) if errors else 0,
            directional_accuracy=directional_correct / len(completed_predictions) if completed_predictions else 0.5,
            regime_performance=regime_performance,
            recent_performance=recent_performance,
            calibration_quality=self._assess_calibration_quality(model_name)
        )

    def _should_rebalance_weights(self, metrics: Dict[str, ModelMetrics]) -> bool:
        """Determine if model weights should be rebalanced"""

        # Need minimum predictions
        total_predictions = sum(m.total_predictions for m in metrics.values())
        if total_predictions < self.min_predictions_for_rebalance * len(metrics):
            return False

        # Calculate performance-based weights
        performance_scores = {
            name: self._calculate_composite_score(metric)
            for name, metric in metrics.items()
        }

        # Normalize to weights
        total_score = sum(performance_scores.values())
        if total_score == 0:
            return False

        performance_weights = {
            name: score / total_score
            for name, score in performance_scores.items()
        }

        # Check if any weight has changed significantly
        for model in self.model_weights:
            current_weight = self.model_weights[model]
            new_weight = performance_weights.get(model, 0.25)
            if abs(current_weight - new_weight) > self.rebalance_threshold:
                return True

        return False

    def _calculate_composite_score(self, metrics: ModelMetrics) -> float:
        """Calculate composite performance score for a model"""

        if metrics.total_predictions == 0:
            return 0.25  # Default score

        # Weighted combination of different metrics
        score = (
            metrics.accuracy_rate * 0.3 +
            metrics.directional_accuracy * 0.2 +
            metrics.recent_performance * 0.3 +
            metrics.calibration_quality * 0.1 +
            (1 - min(1, metrics.mean_absolute_error / 10)) * 0.1  # Lower error is better
        )

        return max(0.1, min(1.0, score))

    def _calculate_new_weights(self, metrics: Dict[str, ModelMetrics]) -> Dict[str, float]:
        """Calculate new model weights based on performance"""

        # Calculate composite scores
        scores = {
            name: self._calculate_composite_score(metric)
            for name, metric in metrics.items()
        }

        # Apply smoothing to prevent drastic changes
        smoothing_factor = 0.7  # Keep 70% of old weight
        new_weights = {}

        total_score = sum(scores.values())
        if total_score == 0:
            return self.model_weights

        for model in self.model_weights:
            performance_weight = scores.get(model, 0.25) / total_score
            current_weight = self.model_weights[model]

            # Smooth the transition
            new_weight = smoothing_factor * current_weight + (1 - smoothing_factor) * performance_weight

            # Enforce bounds [0.1, 0.5]
            new_weights[model] = max(0.1, min(0.5, new_weight))

        # Normalize to sum to 1.0
        total_weight = sum(new_weights.values())
        for model in new_weights:
            new_weights[model] /= total_weight

        return new_weights

    async def _update_model_weights(self, new_weights: Dict[str, float]):
        """Update model weights and log the change"""

        old_weights = self.model_weights.copy()
        self.model_weights = new_weights

        self.logger.info("Model weights rebalanced:")
        for model in new_weights:
            old_w = old_weights[model]
            new_w = new_weights[model]
            change = (new_w - old_w) / old_w * 100 if old_w > 0 else 0
            self.logger.info(f"  {model}: {old_w:.3f} -> {new_w:.3f} ({change:+.1f}%)")

        # Save weights to file
        self._save_weights()

    def _assess_calibration_quality(self, model_name: str) -> float:
        """Assess calibration quality for models that require it"""

        if model_name == 'black_scholes':
            return 1.0  # No calibration needed
        elif model_name == 'ml_neural':
            return 0.9  # Assume good training
        else:
            # For Merton and Heston, check calibration recency and error
            # Simplified implementation
            return 0.8

    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """Get current model rankings by performance"""

        scores = {}
        for model_name in self.model_weights:
            metrics = self._calculate_model_metrics(model_name)
            scores[model_name] = self._calculate_composite_score(metrics)

        # Sort by score descending
        rankings = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return rankings

    def get_regime_specific_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get model weights optimized for specific market regime"""

        weights = self.model_weights.copy()

        # Adjust based on historical regime performance
        for model_name in weights:
            metrics = self._calculate_model_metrics(model_name)
            regime_perf = metrics.regime_performance.get(regime.value, 0.5)

            # Boost weight if model performs well in this regime
            if regime_perf > 0.7:
                weights[model_name] *= 1.2
            elif regime_perf < 0.3:
                weights[model_name] *= 0.8

        # Normalize
        total = sum(weights.values())
        if total > 0:
            for model in weights:
                weights[model] /= total

        return weights

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_weights': self.model_weights,
            'model_metrics': {},
            'rankings': self.get_model_rankings(),
            'total_predictions': len(self.prediction_history),
            'predictions_with_outcomes': sum(
                1 for r in self.prediction_history.values()
                if r.actual_outcome is not None
            )
        }

        # Add detailed metrics for each model
        for model_name in self.model_weights:
            metrics = self._calculate_model_metrics(model_name)
            report['model_metrics'][model_name] = {
                'accuracy_rate': metrics.accuracy_rate,
                'directional_accuracy': metrics.directional_accuracy,
                'mean_absolute_error': metrics.mean_absolute_error,
                'recent_performance': metrics.recent_performance,
                'regime_performance': metrics.regime_performance,
                'total_predictions': metrics.total_predictions
            }

        return report

    def _save_predictions(self):
        """Save predictions to disk"""

        file_path = self.data_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"

        # Convert to serializable format
        data = []
        for record in self.prediction_history.values():
            data.append({
                'prediction_id': record.prediction_id,
                'timestamp': record.timestamp.isoformat(),
                'symbol': record.symbol,
                'strike': record.strike,
                'expiry': record.expiry,
                'option_type': record.option_type,
                'market_price': record.market_price,
                'consensus_price': record.consensus_price,
                'model_predictions': record.model_predictions,
                'market_regime': record.market_regime,
                'actual_outcome': record.actual_outcome,
                'outcome_timestamp': record.outcome_timestamp.isoformat() if record.outcome_timestamp else None,
                'accuracy_scores': record.accuracy_scores
            })

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_weights(self):
        """Save model weights to disk"""

        file_path = self.data_dir / "model_weights.json"

        data = {
            'timestamp': datetime.now().isoformat(),
            'weights': self.model_weights
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_historical_data(self):
        """Load historical predictions and weights from disk"""

        # Load weights
        weights_file = self.data_dir / "model_weights.json"
        if weights_file.exists():
            with open(weights_file, 'r') as f:
                data = json.load(f)
                self.model_weights = data['weights']
                self.logger.info(f"Loaded model weights from {weights_file}")

        # Load recent predictions
        prediction_files = sorted(self.data_dir.glob("predictions_*.json"))
        if prediction_files:
            # Load last file
            with open(prediction_files[-1], 'r') as f:
                data = json.load(f)

                for item in data:
                    record = PredictionRecord(
                        prediction_id=item['prediction_id'],
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        symbol=item['symbol'],
                        strike=item['strike'],
                        expiry=item['expiry'],
                        option_type=item['option_type'],
                        market_price=item['market_price'],
                        consensus_price=item['consensus_price'],
                        model_predictions=item['model_predictions'],
                        market_regime=item['market_regime'],
                        actual_outcome=item['actual_outcome'],
                        outcome_timestamp=datetime.fromisoformat(item['outcome_timestamp']) if item['outcome_timestamp'] else None,
                        accuracy_scores=item['accuracy_scores']
                    )
                    self.prediction_history[record.prediction_id] = record

                self.logger.info(f"Loaded {len(data)} predictions from {prediction_files[-1]}")


# Example usage
async def example_performance_tracking():
    """Example of using the performance tracker"""

    tracker = ModelPerformanceTracker()

    # Create sample prediction
    from strategies.model_ensemble import EnsemblePrediction, StrategyType

    prediction = EnsemblePrediction(
        symbol='AAPL',
        strike=160,
        expiry='2024-03-15',
        option_type='call',
        market_price=5.0,
        consensus_price=5.5,
        price_std=0.3,
        model_disagreement=0.1,
        best_model='heston',
        edge_magnitude=0.1,
        confidence_score=0.8,
        recommended_strategy=StrategyType.DIRECTIONAL
    )

    model_predictions = {
        'black_scholes': 5.2,
        'merton_jump': 5.6,
        'heston': 5.5,
        'ml_neural': 5.7
    }

    # Track prediction
    await tracker.track_prediction(prediction, model_predictions, MarketRegime.VOLATILE)

    # Simulate outcome
    await tracker.update_outcome('AAPL', 160, '2024-03-15', 5.4)

    # Update metrics
    metrics = await tracker.update_performance_metrics()

    # Generate report
    report = tracker.generate_performance_report()

    print("Performance Report:")
    print("=" * 50)
    print(f"Model Rankings: {report['rankings']}")
    print(f"Current Weights: {report['model_weights']}")

    for model, metrics in report['model_metrics'].items():
        print(f"\n{model}:")
        print(f"  Accuracy: {metrics['accuracy_rate']:.2%}")
        print(f"  Directional: {metrics['directional_accuracy']:.2%}")
        print(f"  Recent Performance: {metrics['recent_performance']:.2%}")

if __name__ == "__main__":
    asyncio.run(example_performance_tracking())