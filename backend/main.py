"""
Main entry point for the Options Trading Bot
"""
import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional
from pathlib import Path

from config import config, ComprehensiveRiskLimits
from config.settings import Config


class OptionsTradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config_override: Optional[Config] = None):
        """Initialize the trading bot
        
        Args:
            config_override: Optional configuration override for testing
        """
        self.config = config_override or config
        self.risk_limits = ComprehensiveRiskLimits()
        self.client = None
        self.order_manager = None
        self.risk_manager = None
        self.portfolio_manager = None
        self.strategies = []
        self.running = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_config = self.config.logging
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize all bot components"""
        self.logger.info("Initializing Options Trading Bot...")
        
        try:
            # Validate configuration
            self.config.validate()
            self.logger.info("Configuration validated successfully")
            
            # Initialize components based on what's available
            # Note: Actual implementations will be added as we build them
            
            if self.config.tastyworks.is_sandbox:
                self.logger.warning("Running in SANDBOX mode - no real trades will be executed")
            else:
                self.logger.warning("Running in PRODUCTION mode - real money at risk!")
            
            # Adjust risk limits based on configured risk level
            from config.risk_limits import RiskLevel
            risk_level = RiskLevel(self.config.risk.default_risk_level.lower())
            self.risk_limits.adjust_limits_for_risk_level(risk_level)
            self.logger.info(f"Risk limits adjusted for {risk_level.value} risk level")
            
            self.logger.info("Bot initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bot: {e}")
            raise
    
    async def run(self):
        """Main trading loop"""
        self.logger.info("Starting main trading loop...")
        self.running = True
        
        while self.running:
            try:
                # Placeholder for main trading logic
                # This will be implemented as we build out the components
                
                await asyncio.sleep(1)  # 1-second loop interval
                
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retrying
    
    async def shutdown(self):
        """Graceful shutdown procedure"""
        self.logger.info("Initiating graceful shutdown...")
        self.running = False
        
        try:
            # Check if we should close positions on shutdown
            if self.config.risk.close_on_shutdown:
                self.logger.warning("Closing all positions as configured...")
                # await self._close_all_positions()
            
            # Disconnect from services
            if self.client:
                self.logger.info("Disconnecting from Tastyworks...")
                # await self.client.disconnect()
            
            self.logger.info("Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform system health check"""
        health_status = {
            'config_valid': False,
            'client_connected': False,
            'database_connected': False,
            'redis_connected': False,
            'strategies_loaded': False
        }
        
        try:
            # Check configuration
            self.config.validate()
            health_status['config_valid'] = True
            
            # Check client connection
            if self.client:
                # health_status['client_connected'] = await self.client.is_connected()
                pass
            
            # Check strategies
            health_status['strategies_loaded'] = len(self.strategies) > 0
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
        
        return health_status


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}")
    # The actual shutdown will be handled by the main function
    sys.exit(0)


async def main():
    """Main entry point"""
    # Create bot instance
    bot = OptionsTradingBot()
    
    try:
        # Initialize bot
        await bot.initialize()
        
        # Perform health check
        health = await bot.health_check()
        bot.logger.info(f"System health: {health}")
        
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
    
    # Print startup banner
    print("=" * 60)
    print("          OPTIONS TRADING BOT")
    print("=" * 60)
    print(f"Environment: {'SANDBOX' if config.tastyworks.is_sandbox else 'PRODUCTION'}")
    print(f"Risk Level: {config.risk.default_risk_level}")
    print(f"Max Position Size: {config.risk.max_position_size * 100}%")
    print(f"Max Daily Loss: {config.risk.max_daily_loss * 100}%")
    print("=" * 60)
    print()
    
    # Run the bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)