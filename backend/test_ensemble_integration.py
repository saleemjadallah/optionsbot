"""
Test Script for Model Ensemble Integration
==========================================

Tests all components of the model ensemble system to ensure they work together correctly.
"""

import asyncio
import numpy as np
import pandas as pd
import sys
from datetime import datetime

# Import ensemble components
from strategies.model_ensemble import ModelEnsemble, MarketRegime, StrategyType
from monitoring.model_performance_tracker import ModelPerformanceTracker
from risk.enhanced_risk_manager import EnhancedRiskManager
from config.model_ensemble_config import get_moderate_config, ModelEnsembleConfig


def create_test_data():
    """Create realistic test data for ensemble testing"""

    # Multiple symbols for testing
    symbols = ['AAPL', 'TSLA', 'SPY']

    option_chains = {}
    market_data = {}

    for symbol in symbols:
        # Create option chain
        base_price = {'AAPL': 160, 'TSLA': 220, 'SPY': 450}[symbol]
        strikes = [
            base_price * 0.9,
            base_price * 0.95,
            base_price,
            base_price * 1.05,
            base_price * 1.1
        ]

        option_chains[symbol] = pd.DataFrame({
            'strike': strikes,
            'expiration': ['2024-03-15'] * len(strikes),
            'option_type': ['call'] * len(strikes),
            'bid': [max(0, base_price - k) * 0.95 for k in strikes],
            'ask': [max(0, base_price - k) * 1.05 for k in strikes],
            'volume': [1000 - i*100 for i in range(len(strikes))],
            'open_interest': [5000 - i*500 for i in range(len(strikes))]
        })

        # Create market data with some volatility
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        prices = []
        current_price = base_price

        for _ in range(30):
            # Random walk
            change = np.random.normal(0, base_price * 0.02)
            current_price = max(base_price * 0.8, min(base_price * 1.2, current_price + change))
            prices.append(current_price)

        market_data[symbol] = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.uniform(1e7, 5e7, 30),
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices]
        })

    return option_chains, market_data


async def test_model_ensemble():
    """Test model ensemble functionality"""
    print("\n" + "="*60)
    print("TESTING MODEL ENSEMBLE")
    print("="*60)

    try:
        # Create ensemble with moderate risk
        ensemble = ModelEnsemble(risk_level='moderate')
        print("✓ Model ensemble created")

        # Create test data
        option_chains, market_data = create_test_data()
        print("✓ Test data created")

        # Test market regime detection
        for symbol, data in market_data.items():
            regime = ensemble._detect_market_regime(data)
            print(f"  {symbol} regime: {regime.value}")

        # Test ensemble analysis
        predictions = await ensemble.analyze_universe(option_chains, market_data)
        print(f"✓ Analysis complete: {len(predictions)} predictions")

        # Show top predictions
        if predictions:
            print("\nTop 3 Predictions:")
            for i, pred in enumerate(predictions[:3], 1):
                print(f"  {i}. {pred.symbol} ${pred.strike:.0f} {pred.option_type}")
                print(f"     Edge: {pred.edge_magnitude:.2%}")
                print(f"     Confidence: {pred.confidence_score:.2f}")
                print(f"     Strategy: {pred.recommended_strategy.value}")
                print(f"     Best Model: {pred.best_model}")

        # Test portfolio recommendations
        portfolio_capital = 100000
        recommendations = ensemble.generate_portfolio_recommendations(predictions, portfolio_capital)
        print(f"\n✓ Generated {len(recommendations)} portfolio recommendations")

        if recommendations:
            total_allocated = sum(r['position_sizing']['recommended_size'] for r in recommendations)
            print(f"  Total allocated: ${total_allocated:,.0f} ({total_allocated/portfolio_capital:.1%})")

        return True

    except Exception as e:
        print(f"✗ Model ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance_tracker():
    """Test model performance tracking"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE TRACKER")
    print("="*60)

    try:
        tracker = ModelPerformanceTracker()
        print("✓ Performance tracker created")

        # Create sample prediction
        from strategies.model_ensemble import EnsemblePrediction

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
        print("✓ Prediction tracked")

        # Simulate outcome
        await tracker.update_outcome('AAPL', 160, '2024-03-15', 5.4)
        print("✓ Outcome updated")

        # Update metrics
        metrics = await tracker.update_performance_metrics()
        print("✓ Metrics updated")

        # Generate report
        report = tracker.generate_performance_report()
        print(f"✓ Report generated with {len(report['model_metrics'])} models")

        # Show rankings
        print("\nModel Rankings:")
        for rank, (model, score) in enumerate(report['rankings'], 1):
            print(f"  {rank}. {model}: {score:.3f}")

        return True

    except Exception as e:
        print(f"✗ Performance tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_risk_manager():
    """Test enhanced risk management"""
    print("\n" + "="*60)
    print("TESTING ENHANCED RISK MANAGER")
    print("="*60)

    try:
        risk_manager = EnhancedRiskManager(base_risk_level='moderate')
        print("✓ Risk manager created")

        # Sample recommendations
        recommendations = [
            {
                'symbol': 'AAPL',
                'strategy_type': 'directional',
                'market_analysis': {
                    'confidence': 0.8,
                    'model_disagreement': 0.1,
                    'best_model': 'heston',
                    'edge_magnitude': 0.05
                },
                'position_sizing': {
                    'recommended_size': 10000,
                    'contracts': 10
                },
                'option_details': {
                    'strike': 160,
                    'expiry': '2024-03-15',
                    'option_type': 'call'
                }
            },
            {
                'symbol': 'TSLA',
                'strategy_type': 'vol_arb',
                'market_analysis': {
                    'confidence': 0.6,
                    'model_disagreement': 0.3,
                    'best_model': 'merton_jump',
                    'edge_magnitude': 0.03
                },
                'position_sizing': {
                    'recommended_size': 15000,
                    'contracts': 15
                },
                'option_details': {
                    'strike': 220,
                    'expiry': '2024-03-15',
                    'option_type': 'put'
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
        print(f"✓ Evaluated {len(recommendations)} recommendations")
        print(f"  Approved: {len(approved)}")

        for trade in approved:
            print(f"\n  {trade['symbol']}:")
            print(f"    Original: ${trade['position_sizing']['recommended_size']:,.0f}")
            print(f"    Risk-Adjusted: ${trade['position_sizing']['risk_adjusted_size']:,.0f}")
            print(f"    Risk Level: {trade['risk_assessment']['risk_level']}")

        # Generate risk report
        report = risk_manager.generate_risk_report()
        print("\n✓ Risk report generated")
        print(f"  Portfolio Value: ${report['portfolio_metrics']['total_value']:,.0f}")
        print(f"  VaR (95%): ${report['portfolio_metrics']['var_95']:,.0f}")

        return True

    except Exception as e:
        print(f"✗ Risk manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_configuration():
    """Test model ensemble configuration"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)

    try:
        # Test different configurations
        configs = {
            'Conservative': get_moderate_config(),
            'Moderate': get_moderate_config(),
            'Aggressive': get_moderate_config()
        }

        for name, config in configs.items():
            if config.validate():
                print(f"✓ {name} configuration valid")
                print(f"  Initial weights: {config.initial_weights.to_dict()}")
            else:
                print(f"✗ {name} configuration invalid")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


async def test_integration():
    """Test full integration of all components"""
    print("\n" + "="*60)
    print("TESTING FULL INTEGRATION")
    print("="*60)

    try:
        # Load configuration
        config = get_moderate_config()
        print("✓ Configuration loaded")

        # Create all components
        ensemble = ModelEnsemble(risk_level='moderate')
        tracker = ModelPerformanceTracker()
        risk_manager = EnhancedRiskManager(base_risk_level='moderate')
        print("✓ All components created")

        # Create test data
        option_chains, market_data = create_test_data()

        # Run full workflow
        print("\nRunning full workflow:")

        # 1. Analyze universe
        predictions = await ensemble.analyze_universe(option_chains, market_data)
        print(f"  1. Found {len(predictions)} opportunities")

        # 2. Track predictions
        for pred in predictions[:5]:
            model_preds = {
                'black_scholes': pred.consensus_price * 0.95,
                'merton_jump': pred.consensus_price * 1.02,
                'heston': pred.consensus_price,
                'ml_neural': pred.consensus_price * 1.01
            }
            regime = ensemble._detect_market_regime(market_data.get(pred.symbol, pd.DataFrame()))
            await tracker.track_prediction(pred, model_preds, regime)
        print(f"  2. Tracked top predictions")

        # 3. Generate recommendations
        recommendations = ensemble.generate_portfolio_recommendations(predictions, 100000)
        print(f"  3. Generated {len(recommendations)} recommendations")

        # 4. Risk evaluation
        approved = await risk_manager.evaluate_ensemble_recommendations(
            recommendations, {}
        )
        print(f"  4. Risk approved {len(approved)} trades")

        # 5. Performance update
        metrics = await tracker.update_performance_metrics()
        print(f"  5. Updated performance metrics")

        print("\n✓ Full integration test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MODEL ENSEMBLE INTEGRATION TEST SUITE")
    print("="*60)

    tests = [
        ("Model Ensemble", test_model_ensemble),
        ("Performance Tracker", test_performance_tracker),
        ("Risk Manager", test_risk_manager),
        ("Configuration", test_configuration),
        ("Full Integration", test_integration)
    ]

    results = {}

    for name, test_func in tests:
        try:
            result = await test_func()
            results[name] = result
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Model ensemble is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review and fix issues.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)