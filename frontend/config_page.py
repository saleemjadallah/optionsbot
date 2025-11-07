"""
Configuration Page for Options Trading Bot
==========================================

Advanced configuration interface for:
- Strategy parameters and risk settings
- Model ensemble weights and calibration
- Trading universe selection
- Risk management controls
- API and system settings
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import plotly.graph_objects as go

def render_configuration_page():
    """Main configuration page"""
    
    st.title("⚙️ Trading Bot Configuration")
    st.markdown("Configure trading strategies, risk parameters, and system settings")
    
    # Configuration tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Strategy Config", "🤖 Model Ensemble", "⚖️ Risk Management", 
        "🌐 Trading Universe", "🔧 System Settings"
    ])
    
    with tab1:
        render_strategy_config()
    
    with tab2:
        render_model_ensemble_config()
    
    with tab3:
        render_risk_management_config()
    
    with tab4:
        render_universe_config()
    
    with tab5:
        render_system_config()

def render_strategy_config():
    """Strategy configuration section"""
    
    st.header("🎯 Strategy Configuration")
    
    # Model Ensemble Strategy
    with st.expander("🤖 Model Ensemble Strategy", expanded=True):
        st.markdown("**AI-powered multi-model consensus trading**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ensemble_enabled = st.checkbox("Enable Model Ensemble", value=True)
            min_edge_magnitude = st.slider("Minimum Edge Magnitude (%)", 1.0, 10.0, 3.0, 0.5)
            min_confidence_score = st.slider("Minimum Confidence Score", 0.5, 1.0, 0.7, 0.05)
            max_model_disagreement = st.slider("Max Model Disagreement (%)", 5.0, 50.0, 25.0, 2.5)
        
        with col2:
            position_sizing_method = st.selectbox(
                "Position Sizing Method",
                ["Kelly Criterion", "Fixed Fraction", "Volatility Adjusted", "Model Confidence"]
            )
            max_positions_per_symbol = st.number_input("Max Positions per Symbol", 1, 10, 3)
            rebalance_frequency = st.selectbox(
                "Rebalancing Frequency",
                ["Real-time", "Hourly", "Daily", "Weekly"]
            )
        
        # Strategy-specific settings
        st.markdown("**Strategy Selection Logic:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            directional_threshold = st.number_input("Directional Signal Threshold", 0.01, 0.20, 0.05, 0.01)
        with col2:
            vol_arb_disagreement = st.number_input("Vol Arb Disagreement Min", 0.05, 0.50, 0.15, 0.01)
        with col3:
            gamma_scalp_threshold = st.number_input("Gamma Scalping Threshold", 0.005, 0.050, 0.015, 0.001)
    
    # Gamma Scalping Strategy
    with st.expander("⚡ Gamma Scalping Strategy"):
        st.markdown("**Delta-neutral gamma harvesting**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gamma_enabled = st.checkbox("Enable Gamma Scalping", value=True)
            rebalance_delta_threshold = st.slider("Delta Rebalance Threshold", 0.05, 0.30, 0.10, 0.01)
            min_gamma_exposure = st.number_input("Minimum Gamma Exposure", 0.01, 0.20, 0.03, 0.01)
            max_theta_exposure = st.number_input("Maximum Theta Exposure ($)", 100, 2000, 500, 50)
        
        with col2:
            rebalance_frequency_gamma = st.selectbox(
                "Rebalancing Frequency (Gamma)",
                ["30 seconds", "1 minute", "5 minutes", "15 minutes"],
                index=1
            )
            transaction_cost_threshold = st.number_input("Min Profit vs Transaction Cost ($)", 10, 200, 50, 10)
            vol_regime_filter = st.checkbox("Only Trade in High Vol Regime", value=True)
    
    # Dispersion Trading Strategy  
    with st.expander("📊 Dispersion Trading Strategy"):
        st.markdown("**Index vs components volatility arbitrage**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dispersion_enabled = st.checkbox("Enable Dispersion Trading", value=False)
            correlation_threshold = st.slider("Max Correlation Threshold", 0.3, 0.8, 0.5, 0.05)
            min_vol_spread = st.slider("Minimum Volatility Spread (%)", 2.0, 15.0, 5.0, 0.5)
            component_coverage = st.slider("Component Coverage (%)", 50.0, 90.0, 75.0, 5.0)
        
        with col2:
            index_symbols = st.multiselect(
                "Index Symbols for Dispersion",
                ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"],
                default=["SPY", "QQQ"]
            )
            max_component_weight = st.slider("Max Single Component Weight (%)", 5.0, 25.0, 15.0, 1.0)
            vega_neutral_tolerance = st.number_input("Vega Neutral Tolerance", 0.01, 0.20, 0.05, 0.01)
    
    # Volatility Arbitrage Strategy
    with st.expander("📈 Volatility Arbitrage Strategy"):
        st.markdown("**Implied vs realized volatility trading**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vol_arb_enabled = st.checkbox("Enable Volatility Arbitrage", value=True)
            iv_rank_threshold = st.slider("IV Rank Threshold", 70, 95, 80, 1)
            vol_forecast_period = st.selectbox(
                "Volatility Forecast Period",
                ["10 days", "20 days", "30 days", "60 days"],
                index=1
            )
        
        with col2:
            min_vol_premium = st.slider("Minimum Vol Premium (%)", 2.0, 20.0, 8.0, 1.0)
            max_dte = st.number_input("Maximum Days to Expiration", 7, 90, 45, 1)
            calendar_spread_enabled = st.checkbox("Enable Calendar Spreads", value=True)

def render_model_ensemble_config():
    """Model ensemble configuration section"""
    
    st.header("🤖 Model Ensemble Configuration")
    
    # Model Weights
    st.subheader("⚖️ Model Weights")
    st.markdown("Adjust the relative importance of each pricing model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model weight sliders
        weights = {}
        weights['Black-Scholes'] = st.slider("Black-Scholes Weight", 0.0, 1.0, 0.20, 0.05)
        weights['Merton Jump Diffusion'] = st.slider("Merton Jump Weight", 0.0, 1.0, 0.25, 0.05)
        weights['Heston Stochastic Vol'] = st.slider("Heston Weight", 0.0, 1.0, 0.30, 0.05)
        weights['ML Neural Network'] = st.slider("ML Neural Weight", 0.0, 1.0, 0.25, 0.05)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            st.warning(f"Weights sum to {total_weight:.2f}. They will be normalized to 1.0.")
    
    with col2:
        # Weight visualization
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.4
        )])
        
        fig.update_layout(
            title="Current Model Weights",
            height=300,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Dynamic Weight Adjustment
    st.subheader("🔄 Dynamic Weight Adjustment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_dynamic_weights = st.checkbox("Enable Dynamic Weight Adjustment", value=True)
        performance_lookback = st.number_input("Performance Lookback Days", 7, 90, 30, 1)
        min_trades_for_rebalance = st.number_input("Min Trades for Rebalance", 5, 50, 20, 1)
        weight_change_threshold = st.slider("Weight Change Threshold", 0.05, 0.30, 0.10, 0.01)
    
    with col2:
        max_weight_per_model = st.slider("Maximum Weight per Model", 0.30, 0.70, 0.50, 0.05)
        min_weight_per_model = st.slider("Minimum Weight per Model", 0.05, 0.25, 0.10, 0.01)
        rebalance_frequency_models = st.selectbox(
            "Weight Rebalancing Frequency",
            ["Daily", "Weekly", "Bi-weekly", "Monthly"],
            index=1
        )
    
    # Model Calibration Settings
    st.subheader("🎯 Model Calibration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Merton Jump Diffusion**")
        merton_frequency = st.selectbox("Merton Calibration", ["Daily", "Weekly", "Bi-weekly"], index=1, key="merton_calib")
        merton_data_window = st.number_input("Merton Data Window (days)", 30, 365, 90, key="merton_window")
        merton_max_iterations = st.number_input("Merton Max Iterations", 20, 100, 40, key="merton_iter")

        st.markdown("**ML Neural Network**")
        ml_retrain_frequency = st.selectbox("ML Retraining", ["Weekly", "Monthly", "Quarterly"], index=1, key="ml_retrain")
        ml_training_data_size = st.number_input("Training Samples", 50000, 500000, 200000, 10000, key="ml_samples")
        ml_validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.02, key="ml_val_split")

    with col2:
        st.markdown("**Heston Model**")
        heston_frequency = st.selectbox("Heston Calibration", ["Daily", "Weekly", "Bi-weekly"], index=0, key="heston_calib")
        heston_data_window = st.number_input("Heston Data Window (days)", 30, 365, 60, key="heston_window")
        heston_optimization_method = st.selectbox("Heston Optimization", ["Differential Evolution", "L-BFGS-B", "Nelder-Mead"], key="heston_opt")
    
    # Market Regime Detection
    st.subheader("🌪️ Market Regime Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        regime_detection_enabled = st.checkbox("Enable Regime Detection", value=True)
        regime_lookback = st.number_input("Regime Detection Lookback", 10, 60, 20, 1)
        volatility_threshold_high = st.slider("High Vol Threshold", 20.0, 50.0, 30.0, 1.0)
        volatility_threshold_low = st.slider("Low Vol Threshold", 5.0, 20.0, 12.0, 1.0)
    
    with col2:
        trend_strength_threshold = st.slider("Trend Strength Threshold", 0.1, 1.0, 0.3, 0.05)
        crisis_detection_enabled = st.checkbox("Crisis Detection", value=True)
        crisis_drawdown_threshold = st.slider("Crisis Drawdown Threshold (%)", 3.0, 15.0, 8.0, 0.5)

def render_risk_management_config():
    """Risk management configuration section"""
    
    st.header("⚖️ Risk Management Configuration")
    
    # Position Limits
    st.subheader("📊 Position Limits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_position_size_pct = st.slider("Max Position Size (% of portfolio)", 5.0, 50.0, 25.0, 1.0)
        max_single_symbol_exposure = st.slider("Max Single Symbol Exposure (%)", 10.0, 40.0, 20.0, 1.0)
        max_sector_exposure = st.slider("Max Sector Exposure (%)", 20.0, 60.0, 40.0, 2.5)
        max_leverage_ratio = st.slider("Maximum Leverage Ratio", 1.0, 3.0, 2.0, 0.1)
    
    with col2:
        max_correlation_positions = st.number_input("Max Correlated Positions", 2, 10, 3)
        correlation_threshold = st.slider("Correlation Threshold", 0.50, 0.90, 0.70, 0.05)
        concentration_risk_limit = st.slider("Concentration Risk Limit", 0.15, 0.50, 0.30, 0.05)
        liquidity_requirement = st.selectbox(
            "Minimum Liquidity Requirement",
            ["Low", "Medium", "High", "Very High"],
            index=2
        )
    
    # Greeks Limits
    st.subheader("📈 Greeks Risk Limits")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Delta Limits**")
        max_portfolio_delta = st.slider("Max Portfolio Delta", 0.1, 1.0, 0.3, 0.05)
        delta_neutral_tolerance = st.slider("Delta Neutral Tolerance", 0.05, 0.20, 0.10, 0.01)
        auto_hedge_delta_threshold = st.slider("Auto-hedge Delta Threshold", 0.15, 0.50, 0.25, 0.05)
    
    with col2:
        st.markdown("**Gamma Limits**")
        max_portfolio_gamma = st.slider("Max Portfolio Gamma", 0.05, 0.50, 0.15, 0.01)
        gamma_concentration_limit = st.slider("Gamma Concentration Limit", 0.20, 0.80, 0.40, 0.05)
    
    with col3:
        st.markdown("**Vega/Theta Limits**")
        max_portfolio_vega = st.number_input("Max Portfolio Vega ($)", 1000, 20000, 5000, 250)
        max_portfolio_theta = st.number_input("Max Daily Theta ($)", 100, 2000, 500, 50)
        vega_concentration_limit = st.slider("Vega Concentration Limit", 0.20, 0.80, 0.40, 0.05)
    
    # Loss Limits and Circuit Breakers
    st.subheader("🛑 Loss Limits & Circuit Breakers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        daily_loss_limit_pct = st.slider("Daily Loss Limit (% of portfolio)", 2.0, 15.0, 5.0, 0.5)
        monthly_loss_limit_pct = st.slider("Monthly Loss Limit (%)", 10.0, 30.0, 15.0, 1.0)
        max_drawdown_limit = st.slider("Max Drawdown Limit (%)", 8.0, 25.0, 12.0, 1.0)
        consecutive_loss_limit = st.number_input("Consecutive Loss Limit", 3, 10, 5)
    
    with col2:
        enable_circuit_breakers = st.checkbox("Enable Circuit Breakers", value=True)
        volatility_circuit_breaker = st.slider("Volatility Circuit Breaker (VIX)", 25, 50, 35, 1)
        liquidity_circuit_breaker = st.checkbox("Liquidity Circuit Breaker", value=True)
        news_event_pause = st.checkbox("Pause on Major News Events", value=True)
    
    # VaR Settings
    st.subheader("📊 Value at Risk (VaR) Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        var_confidence_level = st.selectbox("VaR Confidence Level", ["95%", "99%", "99.5%"], index=0)
        var_calculation_method = st.selectbox(
            "VaR Calculation Method",
            ["Historical", "Parametric", "Monte Carlo", "Combined"],
            index=3
        )
        var_lookback_period = st.number_input("VaR Lookback Period (days)", 30, 500, 250)
    
    with col2:
        var_limit_portfolio = st.number_input("Portfolio VaR Limit ($)", 1000, 50000, 10000, 500)
        expected_shortfall_multiplier = st.slider("Expected Shortfall Multiplier", 1.2, 2.0, 1.5, 0.1)
        stress_test_scenarios = st.multiselect(
            "Stress Test Scenarios",
            ["2008 Crisis", "COVID-19", "Flash Crash", "Custom Scenario"],
            default=["2008 Crisis", "COVID-19"]
        )
    
    # Dynamic Risk Adjustment
    st.subheader("🔄 Dynamic Risk Adjustment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_dynamic_risk = st.checkbox("Enable Dynamic Risk Adjustment", value=True)
        risk_regime_detection = st.checkbox("Market Regime-Based Risk", value=True)
        performance_based_adjustment = st.checkbox("Performance-Based Risk Scaling", value=True)
    
    with col2:
        risk_scale_lookback = st.number_input("Risk Scaling Lookback (days)", 5, 30, 14)
        min_risk_multiplier = st.slider("Minimum Risk Multiplier", 0.25, 0.75, 0.50, 0.05)
        max_risk_multiplier = st.slider("Maximum Risk Multiplier", 1.25, 3.0, 2.0, 0.25)

def render_universe_config():
    """Trading universe configuration"""
    
    st.header("🌐 Trading Universe Configuration")
    
    # Universe Selection
    st.subheader("📊 Universe Selection Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        universe_size = st.number_input("Maximum Universe Size", 20, 200, 50, 5)
        min_market_cap = st.selectbox(
            "Minimum Market Cap",
            ["$1B", "$5B", "$10B", "$25B", "$50B"],
            index=2
        )
        min_avg_volume = st.selectbox(
            "Minimum Average Daily Volume",
            ["100K", "500K", "1M", "5M", "10M"],
            index=2
        )
    
    with col2:
        min_option_volume = st.number_input("Minimum Option Volume", 100, 10000, 1000, 100)
        max_bid_ask_spread_pct = st.slider("Max Bid-Ask Spread (%)", 2.0, 15.0, 5.0, 0.5)
        exclude_earnings_days = st.number_input("Exclude Days Around Earnings", 0, 7, 2)
    
    # Sector Allocation
    st.subheader("🏭 Sector Allocation")
    
    sectors = [
        "Technology", "Healthcare", "Financial", "Consumer Discretionary",
        "Communication Services", "Industrial", "Consumer Staples", 
        "Energy", "Utilities", "Real Estate", "Materials"
    ]
    
    sector_weights = {}
    cols = st.columns(3)
    
    for i, sector in enumerate(sectors):
        with cols[i % 3]:
            sector_weights[sector] = st.slider(f"{sector} Max %", 0, 50, 15 if sector == "Technology" else 10, 1)
    
    # Custom Universe
    st.subheader("🎯 Custom Symbols")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Always Include (Core Holdings)**")
        core_symbols = st.text_area(
            "Core Symbols (one per line)",
            value="AAPL\nMSFT\nGOOGL\nTSLA\nSPY\nQQQ",
            height=150
        )
    
    with col2:
        st.markdown("**Never Include (Blacklist)**")
        blacklist_symbols = st.text_area(
            "Blacklisted Symbols (one per line)",
            value="",
            height=150
        )
    
    # ETF Configuration
    st.subheader("📈 ETF Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_etfs = st.checkbox("Include ETFs", value=True)
        max_etf_allocation = st.slider("Max ETF Allocation (%)", 10, 50, 25, 5) if include_etfs else 0
        
        preferred_etfs = st.multiselect(
            "Preferred ETFs",
            ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLU"],
            default=["SPY", "QQQ", "IWM"]
        ) if include_etfs else []
    
    with col2:
        if include_etfs:
            include_sector_etfs = st.checkbox("Include Sector ETFs", value=True)
            include_international = st.checkbox("Include International ETFs", value=False)
            include_commodity = st.checkbox("Include Commodity ETFs", value=False)
    
    # Dynamic Universe Management
    st.subheader("🔄 Dynamic Universe Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_universe_updates = st.checkbox("Auto Universe Updates", value=True)
        update_frequency = st.selectbox(
            "Update Frequency",
            ["Daily", "Weekly", "Monthly", "Quarterly"],
            index=2
        ) if auto_universe_updates else "Manual"
        
        performance_based_removal = st.checkbox("Remove Poor Performers", value=True)
        min_performance_period = st.number_input("Min Performance Period (days)", 30, 180, 60) if performance_based_removal else 0
    
    with col2:
        liquidity_based_removal = st.checkbox("Remove Illiquid Options", value=True)
        spread_deterioration_threshold = st.slider("Spread Deterioration Threshold", 1.5, 5.0, 2.0, 0.1) if liquidity_based_removal else 0
        
        add_trending_symbols = st.checkbox("Add Trending Symbols", value=False)
        trending_volume_threshold = st.slider("Trending Volume Threshold", 2.0, 10.0, 5.0, 0.5) if add_trending_symbols else 0

def render_system_config():
    """System configuration section"""
    
    st.header("🔧 System Configuration")
    
    # API Configuration
    st.subheader("🌐 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tastyworks Settings**")
        api_environment = st.selectbox("API Environment", ["Sandbox", "Production"], index=0)
        connection_timeout = st.number_input("Connection Timeout (seconds)", 5, 60, 30)
        request_rate_limit = st.number_input("Request Rate Limit (per minute)", 30, 120, 60)
        retry_attempts = st.number_input("Retry Attempts", 1, 5, 3)
    
    with col2:
        st.markdown("**Data Feed Settings**")
        enable_streaming = st.checkbox("Enable Real-time Streaming", value=True)
        streaming_symbols_limit = st.number_input("Max Streaming Symbols", 50, 500, 200)
        heartbeat_interval = st.number_input("Heartbeat Interval (seconds)", 30, 300, 60)
        reconnect_delay = st.number_input("Reconnect Delay (seconds)", 5, 30, 10)
    
    # Database Configuration
    st.subheader("🗄️ Database Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        db_type = st.selectbox("Database Type", ["PostgreSQL + TimescaleDB", "InfluxDB", "MongoDB"], index=0)
        data_retention_days = st.number_input("Data Retention (days)", 90, 1095, 365)
        compression_enabled = st.checkbox("Enable Data Compression", value=True)
        backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    
    with col2:
        max_connections = st.number_input("Max DB Connections", 10, 100, 25)
        query_timeout = st.number_input("Query Timeout (seconds)", 10, 120, 30)
        enable_query_caching = st.checkbox("Enable Query Caching", value=True)
        cache_ttl = st.number_input("Cache TTL (minutes)", 1, 60, 10) if enable_query_caching else 0
    
    # Performance & Monitoring
    st.subheader("📊 Performance & Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Logging Configuration**")
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
        log_rotation = st.selectbox("Log Rotation", ["Daily", "Weekly", "Size-based"], index=0)
        max_log_size = st.selectbox("Max Log Size", ["10MB", "50MB", "100MB", "500MB"], index=1)
        retain_logs_days = st.number_input("Retain Logs (days)", 7, 90, 30)
    
    with col2:
        st.markdown("**Performance Monitoring**")
        enable_metrics = st.checkbox("Enable Performance Metrics", value=True)
        metrics_interval = st.number_input("Metrics Collection Interval (seconds)", 5, 300, 30)
        enable_profiling = st.checkbox("Enable Code Profiling", value=False)
        alert_on_errors = st.checkbox("Alert on Errors", value=True)
    
    # Execution Settings
    st.subheader("⚡ Execution Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Order Execution**")
        max_orders_per_minute = st.number_input("Max Orders per Minute", 10, 120, 60)
        order_timeout = st.number_input("Order Timeout (seconds)", 30, 300, 120)
        enable_dry_run = st.checkbox("Enable Dry Run Mode", value=False)
        confirm_all_orders = st.checkbox("Manual Order Confirmation", value=False)
    
    with col2:
        st.markdown("**Threading & Concurrency**")
        max_worker_threads = st.number_input("Max Worker Threads", 2, 16, 4)
        async_io_enabled = st.checkbox("Enable Async I/O", value=True)
        batch_processing = st.checkbox("Enable Batch Processing", value=True)
        batch_size = st.number_input("Batch Size", 10, 100, 25) if batch_processing else 0
    
    # Security Settings
    st.subheader("🔐 Security Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_encryption = st.checkbox("Enable Data Encryption", value=True)
        session_timeout = st.number_input("Session Timeout (hours)", 1, 24, 8)
        enable_2fa = st.checkbox("Enable 2FA (when available)", value=True)
        ip_whitelist = st.text_area("IP Whitelist (one per line)", height=100)
    
    with col2:
        audit_logging = st.checkbox("Enable Audit Logging", value=True)
        sensitive_data_masking = st.checkbox("Mask Sensitive Data in Logs", value=True)
        enable_ssl = st.checkbox("Enforce SSL/TLS", value=True)
        certificate_validation = st.checkbox("Validate Certificates", value=True)
    
    # Save Configuration
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            st.success("Configuration saved successfully!")

    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.warning("Configuration reset to defaults!")

    with col3:
        st.download_button(
            "📥 Export Configuration",
            data=json.dumps({"status": "exported"}, indent=2),
            file_name=f"trading_bot_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    with col4:
        if st.button("📤 Import Configuration", use_container_width=True):
            st.info("Upload configuration file...")

# Main configuration runner
if __name__ == "__main__":
    render_configuration_page()
