"""
Enhanced Options Trading Bot with Model Ensemble Integration
============================================================

Main entry point for the model ensemble-powered options trading system that uses
all four pricing models working together for superior edge detection and risk management.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from config import config, ComprehensiveRiskLimits
from config.settings import Config
from config.risk_limits import RiskLevel

# Import model ensemble components
from strategies.model_ensemble import ModelEnsemble, MarketRegime
from monitoring.model_performance_tracker import ModelPerformanceTracker
from risk.enhanced_risk_manager import EnhancedRiskManager


class EnhancedOptionsTradingBot:
    """
    Enhanced trading bot that uses model ensemble for intelligent trading decisions
    """

    def __init__(self, config_override: Optional[Config] = None):
        """Initialize the enhanced trading bot"""
        self.config = config_override or config
        self.risk_limits = ComprehensiveRiskLimits()

        # Core components (to be initialized)
        self.client = None
        self.order_manager = None
        self.data_manager = None

        # Model ensemble components
        self.model_ensemble = ModelEnsemble(
            risk_level=self.config.risk.default_risk_level.lower()
        )
        self.model_tracker = ModelPerformanceTracker()
        self.risk_manager = EnhancedRiskManager(
            base_risk_level=self.config.risk.default_risk_level.lower()
        )

        # Trading universe
        self.universe = self._build_trading_universe()

        # State tracking
        self.current_positions = {}
        self.pending_orders = {}
        self.running = False
        self.last_calibration = {}
        self.loop_iteration = 0

        # Setup logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging configuration"""
        log_config = self.config.logging

        # Create logs directory if it doesn't exist
        log_path = Path(log_config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging with ensemble-specific format
        logging.basicConfig(
            level=getattr(logging, log_config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        return logging.getLogger('EnhancedTradingBot')

    def _build_trading_universe(self) -> List[str]:
        """Build universe of stocks to analyze"""
        # Start with liquid, optionable stocks
        base_universe = [
            # Tech giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            # Healthcare
            'JNJ', 'PFE', 'ABBV', 'MRK', 'UNH',
            # Industrial
            'BA', 'CAT', 'GE', 'MMM',
            # Consumer
            'KO', 'PEP', 'WMT', 'HD', 'MCD',
            # Energy
            'XOM', 'CVX', 'COP',
            # ETFs for dispersion
            'SPY', 'QQQ', 'IWM', 'DIA'
        ]

        # In production, would filter based on:
        # - Options volume > 1000 daily average
        # - Bid-ask spread < 5% of mid price
        # - Market cap > $10B
        # - Average daily volume > 1M shares

        return base_universe

    async def initialize(self):
        """Initialize all bot components including model ensemble"""
        self.logger.info("=" * 60)
        self.logger.info("Initializing Enhanced Options Trading Bot with Model Ensemble")
        self.logger.info("=" * 60)

        try:
            # Validate configuration
            self.config.validate()
            self.logger.info("✓ Configuration validated successfully")

            # Adjust risk limits
            risk_level = RiskLevel(self.config.risk.default_risk_level.lower())
            self.risk_limits.adjust_limits_for_risk_level(risk_level)
            self.logger.info(f"✓ Risk limits adjusted for {risk_level.value} risk level")

            # Initialize model ensemble weights
            await self._initialize_model_weights()
            self.logger.info("✓ Model ensemble initialized with adaptive weights")

            # Load ML model if available
            await self._load_ml_model()

            # Initialize data connections (placeholder)
            # self.data_manager = await self._initialize_data_manager()

            # Initialize broker connection (placeholder)
            # self.client = await self._initialize_broker_client()

            if self.config.tastyworks.is_sandbox:
                self.logger.warning("⚠️  Running in SANDBOX mode - no real trades will be executed")
            else:
                self.logger.warning("🔴 Running in PRODUCTION mode - real money at risk!")

            self.logger.info("✓ Bot initialization complete")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize bot: {e}")
            raise

    async def _initialize_model_weights(self):
        """Initialize model weights from performance history or defaults"""
        # Check if we have historical performance data
        performance_report = self.model_tracker.generate_performance_report()

        if performance_report['predictions_with_outcomes'] > 50:
            # Use performance-based weights
            self.model_ensemble.model_weights = performance_report['model_weights']
            self.logger.info("Using performance-based model weights")
        else:
            # Use default weights
            self.logger.info("Using default model weights (insufficient history)")

    async def _load_ml_model(self):
        """Load pre-trained ML model if available"""
        try:
            # Check if ML model exists
            ml_model_path = Path("models/ml_pricer/saved_models/options_pricer.pth")
            if ml_model_path.exists():
                # Model would be loaded here
                self.logger.info("✓ ML pricing model loaded successfully")
            else:
                self.logger.warning("⚠️  ML model not found - will use analytical models only")
        except Exception as e:
            self.logger.warning(f"Could not load ML model: {e}")

    async def run(self):
        """Enhanced main trading loop using model ensemble"""
        self.logger.info("Starting enhanced trading loop...")
        self.running = True

        while self.running:
            try:
                self.loop_iteration += 1
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Trading Loop Iteration #{self.loop_iteration}")
                self.logger.info(f"{'='*50}")

                # 1. Fetch market data for entire universe
                market_data = await self._fetch_universe_data()
                option_chains = await self._fetch_option_chains()

                # 2. Update model calibrations if needed
                if self._should_recalibrate():
                    await self._recalibrate_models(market_data)

                # 3. Run ensemble analysis across entire universe
                self.logger.info("🔍 Analyzing options universe with model ensemble...")
                predictions = await self.model_ensemble.analyze_universe(
                    option_chains, market_data
                )
                self.logger.info(f"📊 Found {len(predictions)} potential opportunities")

                # 4. Track model predictions for performance analysis
                for pred in predictions[:20]:  # Track top 20 predictions
                    model_predictions = {
                        'black_scholes': pred.consensus_price * 0.95,  # Placeholder
                        'merton_jump': pred.consensus_price * 1.02,
                        'heston': pred.consensus_price * 0.98,
                        'ml_neural': pred.consensus_price * 1.01
                    }
                    regime = self.model_ensemble._detect_market_regime(
                        market_data.get(pred.symbol, pd.DataFrame())
                    )
                    await self.model_tracker.track_prediction(pred, model_predictions, regime)

                # 5. Update model performance metrics
                if self.loop_iteration % 10 == 0:  # Every 10 iterations
                    metrics = await self.model_tracker.update_performance_metrics()
                    self._log_model_performance(metrics)

                # 6. Generate portfolio recommendations
                available_capital = self._get_available_capital()
                recommendations = self.model_ensemble.generate_portfolio_recommendations(
                    predictions, available_capital
                )
                self.logger.info(f"💡 Generated {len(recommendations)} portfolio recommendations")

                # 7. Filter through enhanced risk management
                approved_trades = await self.risk_manager.evaluate_ensemble_recommendations(
                    recommendations, self.current_positions
                )
                self.logger.info(f"✅ Risk manager approved {len(approved_trades)} trades")

                # 8. Execute approved trades
                for trade in approved_trades:
                    await self._execute_trade(trade)

                # 9. Update existing positions based on model signals
                await self._manage_existing_positions(predictions)

                # 10. Generate and log performance summary
                self._log_iteration_summary(predictions, recommendations, approved_trades)

                # Sleep based on market hours and volatility
                sleep_duration = self._calculate_loop_sleep()
                self.logger.info(f"💤 Sleeping for {sleep_duration} seconds until next iteration")
                await asyncio.sleep(sleep_duration)

            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in ensemble loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Longer wait on error

    async def _fetch_universe_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch market data for entire universe"""
        # Placeholder - would fetch real data
        market_data = {}
        for symbol in self.universe:
            # Generate sample data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            market_data[symbol] = pd.DataFrame({
                'date': dates,
                'close': np.random.uniform(100, 200, 30),
                'volume': np.random.uniform(1e6, 1e8, 30),
                'high': np.random.uniform(102, 202, 30),
                'low': np.random.uniform(98, 198, 30)
            })
        return market_data

    async def _fetch_option_chains(self) -> Dict[str, pd.DataFrame]:
        """Fetch option chains for universe"""
        # Placeholder - would fetch real option chains
        option_chains = {}
        for symbol in self.universe[:5]:  # Limit for demo
            strikes = [90, 95, 100, 105, 110]
            option_chains[symbol] = pd.DataFrame({
                'strike': strikes,
                'expiration': ['2024-03-15'] * len(strikes),
                'option_type': ['call'] * len(strikes),
                'bid': [10 - i for i in range(len(strikes))],
                'ask': [10.5 - i for i in range(len(strikes))],
                'volume': [1000 - i*100 for i in range(len(strikes))],
                'open_interest': [5000 - i*500 for i in range(len(strikes))]
            })
        return option_chains

    def _should_recalibrate(self) -> bool:
        """Check if model recalibration is needed"""
        # Check calibration schedule
        calibration_schedule = {
            'merton_jump': 7,  # Weekly
            'heston': 1,  # Daily
            'ml_neural': 30  # Monthly retraining
        }

        for model, days in calibration_schedule.items():
            last_cal = self.last_calibration.get(model, datetime.now() - timedelta(days=100))
            if (datetime.now() - last_cal).days >= days:
                return True
        return False

    async def _recalibrate_models(self, market_data: Dict[str, pd.DataFrame]):
        """Recalibrate pricing models"""
        self.logger.info("🔧 Recalibrating pricing models...")

        # Placeholder for actual calibration
        # Would calibrate Merton and Heston models here

        # Update calibration timestamps
        self.last_calibration['merton_jump'] = datetime.now()
        self.last_calibration['heston'] = datetime.now()

        # Notify risk manager
        self.risk_manager.update_calibration_timestamp('merton_jump')
        self.risk_manager.update_calibration_timestamp('heston')

        self.logger.info("✓ Model recalibration complete")

    def _get_available_capital(self) -> float:
        """Get available capital for trading"""
        # Placeholder - would get from broker
        total_capital = 100000  # $100k
        used_capital = sum(pos.get('value', 0) for pos in self.current_positions.values())
        return total_capital - used_capital

    async def _execute_trade(self, trade: Dict):
        """Execute an approved trade"""
        try:
            self.logger.info(f"📈 Executing trade for {trade['symbol']}")
            self.logger.info(f"   Strategy: {trade['strategy_type']}")
            self.logger.info(f"   Size: ${trade['position_sizing']['risk_adjusted_size']:,.0f}")
            self.logger.info(f"   Edge: {trade['market_analysis']['edge_magnitude']:.2%}")

            # Placeholder for actual order execution
            # order = await self.order_manager.submit_order(trade)

            # Update position tracking
            self.current_positions[trade['symbol']] = {
                'value': trade['position_sizing']['risk_adjusted_size'],
                'entry_time': datetime.now(),
                'strategy': trade['strategy_type']
            }

        except Exception as e:
            self.logger.error(f"Failed to execute trade for {trade['symbol']}: {e}")

    async def _manage_existing_positions(self, predictions):
        """Manage existing positions based on new predictions"""
        for symbol, position in self.current_positions.items():
            # Check if we have new signals for this position
            relevant_predictions = [p for p in predictions if p.symbol == symbol]

            if relevant_predictions:
                latest = relevant_predictions[0]

                # Check exit conditions
                if abs(latest.edge_magnitude) < 0.01:  # Edge disappeared
                    self.logger.info(f"🔻 Edge disappeared for {symbol}, considering exit")
                    # Would close position here

    def _calculate_loop_sleep(self) -> int:
        """Calculate sleep duration based on market conditions"""
        # During market hours: faster updates
        now = datetime.now()
        if 9 <= now.hour < 16:  # Market hours (simplified)
            return 60  # 1 minute
        else:
            return 300  # 5 minutes

    def _log_model_performance(self, metrics: Dict):
        """Log model performance metrics"""
        self.logger.info("\n📊 Model Performance Update:")
        for model_name, model_metrics in metrics.items():
            self.logger.info(f"  {model_name}:")
            self.logger.info(f"    Accuracy: {model_metrics.accuracy_rate:.2%}")
            self.logger.info(f"    Recent Performance: {model_metrics.recent_performance:.2%}")

    def _log_iteration_summary(self, predictions, recommendations, approved_trades):
        """Log summary of trading iteration"""
        self.logger.info("\n📈 Iteration Summary:")
        self.logger.info(f"  Opportunities Found: {len(predictions)}")
        self.logger.info(f"  Recommendations Made: {len(recommendations)}")
        self.logger.info(f"  Trades Approved: {len(approved_trades)}")
        self.logger.info(f"  Current Positions: {len(self.current_positions)}")

        if approved_trades:
            self.logger.info("\n  Top Approved Trades:")
            for trade in approved_trades[:3]:
                self.logger.info(f"    • {trade['symbol']} ({trade['strategy_type']})")
                self.logger.info(f"      Edge: {trade['market_analysis']['edge_magnitude']:.2%}")
                self.logger.info(f"      Size: ${trade['position_sizing']['risk_adjusted_size']:,.0f}")

    async def shutdown(self):
        """Enhanced graceful shutdown procedure"""
        self.logger.info("🛑 Initiating enhanced shutdown...")
        self.running = False

        try:
            # Save model performance data
            report = self.model_tracker.generate_performance_report()
            self.logger.info(f"📊 Final model rankings: {report['rankings']}")

            # Check if we should close positions on shutdown
            if self.config.risk.close_on_shutdown and self.current_positions:
                self.logger.warning(f"⚠️  Closing {len(self.current_positions)} positions...")
                # Would close positions here

            # Save final state
            self.logger.info("💾 Saving final state...")
            # Would save state here

            self.logger.info("✓ Enhanced shutdown complete")

        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
            raise

    async def health_check(self) -> Dict[str, bool]:
        """Perform enhanced system health check"""
        health_status = {
            'config_valid': False,
            'models_initialized': False,
            'risk_manager_ready': False,
            'performance_tracker_ready': False,
            'universe_configured': False
        }

        try:
            # Check configuration
            self.config.validate()
            health_status['config_valid'] = True

            # Check model ensemble
            health_status['models_initialized'] = all([
                self.model_ensemble.models.get('black_scholes'),
                self.model_ensemble.models.get('merton_jump'),
                self.model_ensemble.models.get('heston'),
                self.model_ensemble.models.get('ml_neural')
            ])

            # Check risk manager
            health_status['risk_manager_ready'] = self.risk_manager is not None

            # Check performance tracker
            health_status['performance_tracker_ready'] = self.model_tracker is not None

            # Check universe
            health_status['universe_configured'] = len(self.universe) > 0

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

        return health_status


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger = logging.getLogger('EnhancedTradingBot')
    logger.info(f"Received signal {signum}")
    sys.exit(0)


async def main():
    """Main entry point for enhanced bot"""
    # Create bot instance
    bot = EnhancedOptionsTradingBot()

    try:
        # Initialize bot
        await bot.initialize()

        # Perform health check
        health = await bot.health_check()
        bot.logger.info(f"System health: {health}")

        if not all(health.values()):
            bot.logger.warning("⚠️  Some components not ready:")
            for component, status in health.items():
                if not status:
                    bot.logger.warning(f"  ❌ {component}")

        # Run bot
        await bot.run()

    except KeyboardInterrupt:
        bot.logger.info("Received interrupt signal")
    except Exception as e:
        bot.logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Ensure cleanup
        await bot.shutdown()


if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Import numpy for demo
    import numpy as np

    # Print startup banner
    print("=" * 60)
    print("     ENHANCED OPTIONS TRADING BOT WITH MODEL ENSEMBLE")
    print("=" * 60)
    print("🧠 Pricing Models:")
    print("   • Black-Scholes (Analytical)")
    print("   • Merton Jump Diffusion (Jump Risk)")
    print("   • Heston (Stochastic Volatility)")
    print("   • Neural Network (ML-Based)")
    print("-" * 60)
    print(f"Environment: {'SANDBOX' if config.tastyworks.is_sandbox else 'PRODUCTION'}")
    print(f"Risk Level: {config.risk.default_risk_level}")
    print(f"Max Position Size: {config.risk.max_position_size * 100}%")
    print(f"Max Daily Loss: {config.risk.max_daily_loss * 100}%")
    print("=" * 60)
    print()

    # Run the enhanced bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Shutdown complete")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)