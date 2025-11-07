-- Options Trading Bot Database Schema
-- ====================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set default schema
SET search_path TO trading, public;

-- =====================================
-- Core Trading Tables
-- =====================================

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    option_type VARCHAR(10) CHECK (option_type IN ('call', 'put')),
    strike_price DECIMAL(10, 2) NOT NULL,
    expiration_date DATE NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10, 4) NOT NULL,
    current_price DECIMAL(10, 4),
    strategy_name VARCHAR(100),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed', 'expired')),
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    pnl DECIMAL(12, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    position_id INTEGER REFERENCES positions(id),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    limit_price DECIMAL(10, 4),
    stop_price DECIMAL(10, 4),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'submitted', 'filled', 'cancelled', 'rejected')),
    filled_price DECIMAL(10, 4),
    filled_quantity INTEGER,
    broker_order_id VARCHAR(100),
    submitted_at TIMESTAMP,
    filled_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trades table (execution history)
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    position_id INTEGER REFERENCES positions(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10, 4) NOT NULL,
    commission DECIMAL(8, 2),
    executed_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Strategies table
CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    capital_allocated DECIMAL(12, 2),
    max_positions INTEGER DEFAULT 10,
    stop_loss_pct DECIMAL(5, 4),
    take_profit_pct DECIMAL(5, 4),
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================
-- Market Data Tables
-- =====================================

SET search_path TO market_data, public;

-- Options chain data
CREATE TABLE IF NOT EXISTS options_chain (
    id SERIAL,
    symbol VARCHAR(20) NOT NULL,
    underlying_price DECIMAL(10, 2) NOT NULL,
    option_type VARCHAR(10) NOT NULL,
    strike_price DECIMAL(10, 2) NOT NULL,
    expiration_date DATE NOT NULL,
    bid DECIMAL(10, 4),
    ask DECIMAL(10, 4),
    last_price DECIMAL(10, 4),
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility DECIMAL(8, 6),
    delta DECIMAL(8, 6),
    gamma DECIMAL(8, 6),
    theta DECIMAL(8, 6),
    vega DECIMAL(8, 6),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('options_chain', 'timestamp', if_not_exists => TRUE);

-- Stock prices
CREATE TABLE IF NOT EXISTS stock_prices (
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2) NOT NULL,
    volume BIGINT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('stock_prices', 'timestamp', if_not_exists => TRUE);

-- Volatility surface
CREATE TABLE IF NOT EXISTS volatility_surface (
    id SERIAL,
    symbol VARCHAR(20) NOT NULL,
    expiration_date DATE NOT NULL,
    strike_price DECIMAL(10, 2) NOT NULL,
    implied_volatility DECIMAL(8, 6) NOT NULL,
    moneyness DECIMAL(8, 4),
    time_to_expiry DECIMAL(8, 6),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('volatility_surface', 'timestamp', if_not_exists => TRUE);

-- =====================================
-- Analytics Tables
-- =====================================

SET search_path TO analytics, public;

-- Model predictions
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL,
    model_name VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strike_price DECIMAL(10, 2),
    expiration_date DATE,
    predicted_price DECIMAL(10, 4) NOT NULL,
    market_price DECIMAL(10, 4),
    confidence DECIMAL(5, 4),
    prediction_error DECIMAL(10, 4),
    features JSONB,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('model_predictions', 'timestamp', if_not_exists => TRUE);

-- Portfolio metrics
CREATE TABLE IF NOT EXISTS portfolio_metrics (
    id SERIAL,
    portfolio_value DECIMAL(12, 2) NOT NULL,
    daily_pnl DECIMAL(10, 2),
    cumulative_pnl DECIMAL(12, 2),
    positions_count INTEGER,
    win_rate DECIMAL(5, 4),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    var_95 DECIMAL(10, 2),
    portfolio_delta DECIMAL(8, 4),
    portfolio_gamma DECIMAL(8, 4),
    portfolio_theta DECIMAL(8, 4),
    portfolio_vega DECIMAL(8, 4),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('portfolio_metrics', 'timestamp', if_not_exists => TRUE);

-- Strategy performance
CREATE TABLE IF NOT EXISTS strategy_performance (
    id SERIAL,
    strategy_name VARCHAR(100) NOT NULL,
    trades_count INTEGER,
    win_count INTEGER,
    loss_count INTEGER,
    total_pnl DECIMAL(12, 2),
    avg_win DECIMAL(10, 2),
    avg_loss DECIMAL(10, 2),
    win_rate DECIMAL(5, 4),
    profit_factor DECIMAL(8, 4),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('strategy_performance', 'timestamp', if_not_exists => TRUE);

-- Risk metrics
CREATE TABLE IF NOT EXISTS risk_metrics (
    id SERIAL,
    var_95 DECIMAL(10, 2),
    var_99 DECIMAL(10, 2),
    expected_shortfall DECIMAL(10, 2),
    max_drawdown DECIMAL(8, 4),
    current_drawdown DECIMAL(8, 4),
    kelly_fraction DECIMAL(6, 4),
    risk_score DECIMAL(4, 2),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('risk_metrics', 'timestamp', if_not_exists => TRUE);

-- =====================================
-- Indexes for Performance
-- =====================================

SET search_path TO trading, public;

-- Positions indexes
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_strategy ON positions(strategy_name);
CREATE INDEX idx_positions_opened_at ON positions(opened_at);

-- Orders indexes
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_position_id ON orders(position_id);

-- Trades indexes
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_executed_at ON trades(executed_at);

SET search_path TO market_data, public;

-- Market data indexes
CREATE INDEX idx_options_chain_symbol_exp ON options_chain(symbol, expiration_date);
CREATE INDEX idx_stock_prices_symbol ON stock_prices(symbol);

SET search_path TO analytics, public;

-- Analytics indexes
CREATE INDEX idx_model_predictions_model ON model_predictions(model_name);
CREATE INDEX idx_strategy_performance_name ON strategy_performance(strategy_name);

-- =====================================
-- Views for Common Queries
-- =====================================

SET search_path TO trading, public;

-- Active positions view
CREATE OR REPLACE VIEW active_positions AS
SELECT
    p.*,
    s.name as strategy_name_full,
    s.is_active as strategy_active
FROM positions p
LEFT JOIN strategies s ON p.strategy_name = s.name
WHERE p.status = 'open';

-- Daily PnL view
CREATE OR REPLACE VIEW daily_pnl AS
SELECT
    DATE(closed_at) as date,
    COUNT(*) as trades_count,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl
FROM positions
WHERE status = 'closed' AND closed_at IS NOT NULL
GROUP BY DATE(closed_at)
ORDER BY date DESC;

-- =====================================
-- Functions and Triggers
-- =====================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add update triggers
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =====================================
-- Initial Data
-- =====================================

-- Insert default strategies
INSERT INTO strategies (name, description, is_active, capital_allocated, max_positions) VALUES
    ('Model Ensemble', 'Ensemble of pricing models for option valuation', true, 50000, 20),
    ('Volatility Arbitrage', 'Exploits IV vs HV discrepancies', true, 30000, 15),
    ('Delta Neutral', 'Maintains delta-neutral portfolio', false, 20000, 10),
    ('Calendar Spread', 'Time decay strategy with calendar spreads', false, 15000, 8)
ON CONFLICT (name) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA market_data TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA market_data TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO trader;