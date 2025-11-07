"""
Core configuration settings for the Options Trading Bot
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


@dataclass
class TastyworksConfig:
    """Tastyworks API configuration"""
    username: str = field(default_factory=lambda: os.getenv('TW_USERNAME', ''))
    password: str = field(default_factory=lambda: os.getenv('TW_PASSWORD', ''))
    account_number: str = field(default_factory=lambda: os.getenv('TW_ACCOUNT', ''))
    is_sandbox: bool = field(default_factory=lambda: os.getenv('TW_SANDBOX', 'True') == 'True')
    
    def validate(self) -> bool:
        """Validate required configuration"""
        if not all([self.username, self.password, self.account_number]):
            raise ValueError("Missing required Tastyworks credentials")
        return True


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.getenv('DB_PORT', '5432')))
    name: str = field(default_factory=lambda: os.getenv('DB_NAME', 'options_trader'))
    user: str = field(default_factory=lambda: os.getenv('DB_USER', 'trader'))
    password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', ''))
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RedisConfig:
    """Redis cache configuration"""
    host: str = field(default_factory=lambda: os.getenv('REDIS_HOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.getenv('REDIS_PORT', '6379')))
    password: Optional[str] = field(default_factory=lambda: os.getenv('REDIS_PASSWORD') or None)
    db: int = field(default_factory=lambda: int(os.getenv('REDIS_DB', '0')))
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        """Generate Redis connection parameters"""
        params = {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'decode_responses': True
        }
        if self.password:
            params['password'] = self.password
        return params


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = field(
        default_factory=lambda: float(os.getenv('MAX_POSITION_SIZE', '0.25'))
    )
    max_daily_loss: float = field(
        default_factory=lambda: float(os.getenv('MAX_DAILY_LOSS', '0.05'))
    )
    default_risk_level: str = field(
        default_factory=lambda: os.getenv('DEFAULT_RISK_LEVEL', 'moderate')
    )
    var_confidence: float = field(
        default_factory=lambda: float(os.getenv('VAR_CONFIDENCE', '0.95'))
    )
    leverage_limit: float = field(
        default_factory=lambda: float(os.getenv('LEVERAGE_LIMIT', '2.0'))
    )
    close_on_shutdown: bool = field(
        default_factory=lambda: os.getenv('CLOSE_ON_SHUTDOWN', 'False') == 'True'
    )
    
    def validate(self) -> bool:
        """Validate risk parameters"""
        if self.default_risk_level not in ['low', 'moderate', 'high']:
            raise ValueError(f"Invalid risk level: {self.default_risk_level}")
        if not 0 < self.max_position_size <= 1:
            raise ValueError(f"Invalid max position size: {self.max_position_size}")
        if not 0 < self.max_daily_loss <= 1:
            raise ValueError(f"Invalid max daily loss: {self.max_daily_loss}")
        if not 0.9 <= self.var_confidence < 1:
            raise ValueError(f"Invalid VaR confidence: {self.var_confidence}")
        if self.leverage_limit < 1:
            raise ValueError(f"Invalid leverage limit: {self.leverage_limit}")
        return True


@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    min_option_volume: int = field(
        default_factory=lambda: int(os.getenv('MIN_OPTION_VOLUME', '100'))
    )
    min_option_oi: int = field(
        default_factory=lambda: int(os.getenv('MIN_OPTION_OI', '500'))
    )
    max_spread_percent: float = field(
        default_factory=lambda: float(os.getenv('MAX_SPREAD_PERCENT', '0.02'))
    )
    rebalance_threshold: float = field(
        default_factory=lambda: float(os.getenv('REBALANCE_THRESHOLD', '0.1'))
    )
    min_profit_threshold: float = field(
        default_factory=lambda: float(os.getenv('MIN_PROFIT_THRESHOLD', '50'))
    )
    max_orders_per_minute: int = field(
        default_factory=lambda: int(os.getenv('MAX_ORDERS_PER_MINUTE', '60'))
    )
    max_api_calls_per_second: int = field(
        default_factory=lambda: int(os.getenv('MAX_API_CALLS_PER_SECOND', '10'))
    )


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    dashboard_port: int = field(
        default_factory=lambda: int(os.getenv('DASHBOARD_PORT', '8050'))
    )
    enable_email_alerts: bool = field(
        default_factory=lambda: os.getenv('ENABLE_EMAIL_ALERTS', 'False') == 'True'
    )
    alert_email: str = field(
        default_factory=lambda: os.getenv('ALERT_EMAIL', '')
    )
    smtp_server: str = field(
        default_factory=lambda: os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    )
    smtp_port: int = field(
        default_factory=lambda: int(os.getenv('SMTP_PORT', '587'))
    )
    smtp_username: str = field(
        default_factory=lambda: os.getenv('SMTP_USERNAME', '')
    )
    smtp_password: str = field(
        default_factory=lambda: os.getenv('SMTP_PASSWORD', '')
    )
    enable_prometheus: bool = field(
        default_factory=lambda: os.getenv('ENABLE_PROMETHEUS', 'True') == 'True'
    )
    prometheus_port: int = field(
        default_factory=lambda: int(os.getenv('PROMETHEUS_PORT', '9090'))
    )
    metrics_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv('METRICS_INTERVAL_SECONDS', '60'))
    )


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = field(
        default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO')
    )
    log_file: str = field(
        default_factory=lambda: os.getenv('LOG_FILE', 'logs/trading_bot.log')
    )
    max_log_size_mb: int = field(
        default_factory=lambda: int(os.getenv('MAX_LOG_SIZE_MB', '100'))
    )
    log_backup_count: int = field(
        default_factory=lambda: int(os.getenv('LOG_BACKUP_COUNT', '5'))
    )
    
    @property
    def log_path(self) -> Path:
        """Get full log file path"""
        return Path(self.log_file).resolve()


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str = field(
        default_factory=lambda: os.getenv('BACKTEST_START_DATE', '2022-01-01')
    )
    end_date: str = field(
        default_factory=lambda: os.getenv('BACKTEST_END_DATE', '2023-12-31')
    )
    initial_capital: float = field(
        default_factory=lambda: float(os.getenv('BACKTEST_INITIAL_CAPITAL', '100000'))
    )
    commission: float = field(
        default_factory=lambda: float(os.getenv('BACKTEST_COMMISSION', '0.65'))
    )


@dataclass
class MLConfig:
    """Machine Learning configuration"""
    model_path: str = field(
        default_factory=lambda: os.getenv('ML_MODEL_PATH', 'models/saved/')
    )
    retrain_interval_days: int = field(
        default_factory=lambda: int(os.getenv('RETRAIN_INTERVAL_DAYS', '30'))
    )
    min_training_samples: int = field(
        default_factory=lambda: int(os.getenv('MIN_TRAINING_SAMPLES', '10000'))
    )
    
    @property
    def model_dir(self) -> Path:
        """Get model directory path"""
        path = Path(self.model_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class DataConfig:
    """Data source configuration"""
    use_live_data: bool = field(
        default_factory=lambda: os.getenv('USE_LIVE_DATA', 'True') == 'True'
    )
    historical_data_path: str = field(
        default_factory=lambda: os.getenv('HISTORICAL_DATA_PATH', 'data/historical/')
    )
    cache_expiry_seconds: int = field(
        default_factory=lambda: int(os.getenv('CACHE_EXPIRY_SECONDS', '300'))
    )
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path"""
        path = Path(self.historical_data_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


class Config:
    """Main configuration class aggregating all settings"""
    
    def __init__(self):
        self.tastyworks = TastyworksConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.risk = RiskConfig()
        self.trading = TradingConfig()
        self.monitoring = MonitoringConfig()
        self.logging = LoggingConfig()
        self.backtest = BacktestConfig()
        self.ml = MLConfig()
        self.data = DataConfig()
    
    def validate(self) -> bool:
        """Validate all configuration sections"""
        try:
            if not self.tastyworks.is_sandbox:
                self.tastyworks.validate()
            self.risk.validate()
            return True
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    @classmethod
    def load(cls) -> 'Config':
        """Load and validate configuration"""
        config = cls()
        config.validate()
        return config


# Global config instance
config = Config.load()