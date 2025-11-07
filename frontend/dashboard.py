"""
Advanced Options Trading Bot - Frontend Dashboard
================================================

A comprehensive Streamlit-based frontend for the options trading bot providing:
- Real-time portfolio monitoring
- Strategy management and configuration
- Risk analysis and controls
- Model ensemble performance tracking
- Trade execution interface
- Historical performance analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from utils.tooltip_component import render_term_with_tooltip

# Configure Streamlit page
st.set_page_config(
    page_title="Options Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }

    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }

    .status-running {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }

    .status-stopped {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }

    .status-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }

    .strategy-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }

    .risk-gauge {
        text-align: center;
        padding: 1rem;
    }

    .sidebar-section {
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.initialize_session_state()
        self.mock_data = self.generate_mock_data()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'bot_status' not in st.session_state:
            st.session_state.bot_status = 'stopped'
        if 'risk_level' not in st.session_state:
            st.session_state.risk_level = 'moderate'
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'selected_strategies' not in st.session_state:
            st.session_state.selected_strategies = ['model_ensemble']

    def generate_mock_data(self):
        """Generate realistic mock data for demonstration"""
        np.random.seed(42)

        # Portfolio metrics
        portfolio_value = 125000 + np.random.normal(0, 5000)
        daily_pnl = np.random.normal(500, 1500)

        # Generate recent P&L history
        dates = pd.date_range(start='2024-01-01', end='2024-03-15', freq='D')
        pnl_history = np.cumsum(np.random.normal(200, 800, len(dates)))

        # Current positions
        positions = [
            {
                'symbol': 'AAPL',
                'strategy': 'Model Ensemble',
                'type': 'Call Spread',
                'quantity': 10,
                'entry_price': 2.50,
                'current_price': 3.20,
                'pnl': 700,
                'delta': 0.35,
                'gamma': 0.02,
                'theta': -12.5,
                'vega': 45.2,
                'expiration': '2024-04-19'
            },
            {
                'symbol': 'TSLA',
                'strategy': 'Gamma Scalping',
                'type': 'Long Straddle',
                'quantity': 5,
                'entry_price': 8.40,
                'current_price': 9.10,
                'pnl': 350,
                'delta': 0.02,
                'gamma': 0.08,
                'theta': -25.0,
                'vega': 120.5,
                'expiration': '2024-03-22'
            },
            {
                'symbol': 'SPY',
                'strategy': 'Dispersion',
                'type': 'Iron Condor',
                'quantity': 20,
                'entry_price': 1.20,
                'current_price': 0.95,
                'pnl': -500,
                'delta': -0.05,
                'gamma': 0.01,
                'theta': 15.0,
                'vega': -25.0,
                'expiration': '2024-04-12'
            }
        ]

        # Model performance data
        model_performance = {
            'black_scholes': 0.72,
            'merton_jump': 0.84,
            'heston': 0.78,
            'ml_neural': 0.89
        }

        # Recent trade history
        trade_history = []
        for i in range(50):
            trade_history.append({
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 720)),
                'symbol': np.random.choice(['AAPL', 'TSLA', 'MSFT', 'SPY', 'QQQ']),
                'action': np.random.choice(['BUY', 'SELL']),
                'strategy': np.random.choice(['Model Ensemble', 'Gamma Scalping', 'Dispersion']),
                'quantity': np.random.randint(1, 25),
                'price': np.random.uniform(0.50, 15.00),
                'pnl': np.random.normal(100, 500)
            })

        return {
            'portfolio_value': portfolio_value,
            'daily_pnl': daily_pnl,
            'pnl_history': pnl_history,
            'dates': dates,
            'positions': positions,
            'model_performance': model_performance,
            'trade_history': trade_history
        }

    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.markdown("## 🎛️ Trading Controls")

        # Bot status control
        with st.sidebar.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### Bot Status")

            col1, col2 = st.sidebar.columns(2)

            if col1.button("🟢 Start", use_container_width=True):
                st.session_state.bot_status = 'running'
                st.success("Trading bot started!")

            if col2.button("🔴 Stop", use_container_width=True):
                st.session_state.bot_status = 'stopped'
                st.warning("Trading bot stopped!")

            # Display current status
            status_color = "🟢" if st.session_state.bot_status == 'running' else "🔴"
            st.markdown(f"**Current Status:** {status_color} {st.session_state.bot_status.upper()}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Risk level control
        st.sidebar.markdown("### ⚖️ Risk Level")
        risk_level = st.sidebar.selectbox(
            "Select Risk Level",
            options=['low', 'moderate', 'high'],
            index=['low', 'moderate', 'high'].index(st.session_state.risk_level)
        )

        if risk_level != st.session_state.risk_level:
            st.session_state.risk_level = risk_level
            st.sidebar.success(f"Risk level changed to: {risk_level}")

        # Strategy selection
        st.sidebar.markdown("### 📊 Active Strategies")
        available_strategies = [
            'model_ensemble',
            'gamma_scalping',
            'dispersion',
            'volatility_arbitrage'
        ]

        selected = st.sidebar.multiselect(
            "Select Active Strategies",
            options=available_strategies,
            default=st.session_state.selected_strategies,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        st.session_state.selected_strategies = selected

        # Auto-refresh toggle
        st.sidebar.markdown("### 🔄 Auto Refresh")
        auto_refresh = st.sidebar.checkbox(
            "Enable auto-refresh (30s)",
            value=st.session_state.auto_refresh
        )
        st.session_state.auto_refresh = auto_refresh

        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()

        # Quick actions
        st.sidebar.markdown("### ⚡ Quick Actions")

        if st.sidebar.button("📊 Rebalance Portfolio", use_container_width=True):
            st.sidebar.info("Portfolio rebalancing initiated...")

        if st.sidebar.button("🛡️ Emergency Stop All", use_container_width=True):
            st.sidebar.error("Emergency stop activated!")

        # Account info
        st.sidebar.markdown("### 💰 Account Info")
        st.sidebar.metric("Available Cash", "$45,230", "2.3%")
        st.sidebar.metric("Buying Power", "$180,920", "-1.2%")
        st.sidebar.metric("Day Trade Count", "2/3", "0")

    def render_main_dashboard(self):
        """Render main dashboard content"""

        # Header
        st.markdown('<h1 class="main-header">📈 Advanced Options Trading Dashboard</h1>',
                   unsafe_allow_html=True)

        # Key metrics row
        self.render_key_metrics()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Portfolio", "🎯 Strategies", "📈 Performance",
            "🤖 Model Ensemble", "⚖️ Risk Analysis", "📋 Trade Log"
        ])

        with tab1:
            self.render_portfolio_tab()

        with tab2:
            self.render_strategies_tab()

        with tab3:
            self.render_performance_tab()

        with tab4:
            self.render_model_ensemble_tab()

        with tab5:
            self.render_risk_analysis_tab()

        with tab6:
            self.render_trade_log_tab()

    def render_key_metrics(self):
        """Render key portfolio metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value">${:,.0f}</div>
                <div>+{:,.0f} ({:.1f}%)</div>
            </div>
            """.format(
                self.mock_data['portfolio_value'],
                self.mock_data['daily_pnl'],
                (self.mock_data['daily_pnl'] / self.mock_data['portfolio_value']) * 100
            ), unsafe_allow_html=True)

        with col2:
            total_delta = sum(pos['delta'] * pos['quantity'] for pos in self.mock_data['positions'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" style="display: flex; align-items: center; gap: 0.3em;">
                    Portfolio Delta
            """, unsafe_allow_html=True)
            render_term_with_tooltip("Delta", icon_size="0.6em")
            st.markdown(f"""
                </div>
                <div class="metric-value">{total_delta:.2f}</div>
                <div>Net directional exposure</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            total_gamma = sum(pos['gamma'] * pos['quantity'] for pos in self.mock_data['positions'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" style="display: flex; align-items: center; gap: 0.3em;">
                    Portfolio Gamma
            """, unsafe_allow_html=True)
            render_term_with_tooltip("Gamma", icon_size="0.6em")
            st.markdown(f"""
                </div>
                <div class="metric-value">{total_gamma:.2f}</div>
                <div>Convexity exposure</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            total_theta = sum(pos['theta'] * pos['quantity'] for pos in self.mock_data['positions'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" style="display: flex; align-items: center; gap: 0.3em;">
                    Portfolio Theta
            """, unsafe_allow_html=True)
            render_term_with_tooltip("Theta", icon_size="0.6em")
            st.markdown(f"""
                </div>
                <div class="metric-value">${total_theta:.0f}</div>
                <div>Daily time decay</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            total_vega = sum(pos['vega'] * pos['quantity'] for pos in self.mock_data['positions'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" style="display: flex; align-items: center; gap: 0.3em;">
                    Portfolio Vega
            """, unsafe_allow_html=True)
            render_term_with_tooltip("Vega", icon_size="0.6em")
            st.markdown(f"""
                </div>
                <div class="metric-value">${total_vega:.0f}</div>
                <div>Volatility exposure</div>
            </div>
            """, unsafe_allow_html=True)

    def render_portfolio_tab(self):
        """Render portfolio overview tab"""

        col1, col2 = st.columns([2, 1])

        with col1:
            # P&L Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.mock_data['dates'],
                y=self.mock_data['pnl_history'],
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='#1f77b4', width=2)
            ))

            fig.update_layout(
                title="📈 Cumulative P&L Performance",
                xaxis_title="Date",
                yaxis_title="P&L ($)",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Risk metrics gauge
            col_header1, col_header2 = st.columns([5, 1])
            with col_header1:
                st.markdown("### 🎯 Risk Metrics")
            with col_header2:
                render_term_with_tooltip("VaR", icon_size="0.8em")

            # VaR gauge
            var_value = -2500  # Example VaR
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = abs(var_value),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Daily VaR (95%)"},
                delta = {'reference': 3000},
                gauge = {
                    'axis': {'range': [None, 5000]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 2000], 'color': "lightgreen"},
                        {'range': [2000, 4000], 'color': "yellow"},
                        {'range': [4000, 5000], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 4000
                    }
                }
            ))

            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk level indicator
            risk_colors = {'low': '#28a745', 'moderate': '#ffc107', 'high': '#dc3545'}
            st.markdown(f"""
            <div class="status-card" style="border-color: {risk_colors[st.session_state.risk_level]};">
                <strong>Current Risk Level:</strong> {st.session_state.risk_level.upper()}
            </div>
            """, unsafe_allow_html=True)

        # Current positions table
        st.markdown("### 📊 Current Positions")

        positions_df = pd.DataFrame(self.mock_data['positions'])
        positions_df['P&L %'] = (positions_df['pnl'] /
                               (positions_df['entry_price'] * positions_df['quantity'] * 100)) * 100

        # Color code P&L
        def color_pnl(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}'

        styled_df = positions_df.style.applymap(color_pnl, subset=['pnl', 'P&L %'])
        st.dataframe(styled_df, use_container_width=True)

    def render_strategies_tab(self):
        """Render strategies management tab"""

        st.markdown("### 🎯 Strategy Performance & Configuration")

        # Strategy cards
        strategies = [
            {
                'name': 'Model Ensemble',
                'status': 'Active',
                'positions': 15,
                'daily_pnl': 850,
                'success_rate': 72.5,
                'description': 'AI-powered multi-model options pricing with consensus signals'
            },
            {
                'name': 'Gamma Scalping',
                'status': 'Active',
                'positions': 8,
                'daily_pnl': 340,
                'success_rate': 68.2,
                'description': 'Delta-neutral scalping strategy capturing gamma profits'
            },
            {
                'name': 'Dispersion Trading',
                'status': 'Paused',
                'positions': 3,
                'daily_pnl': -120,
                'success_rate': 45.8,
                'description': 'Index vs components volatility arbitrage'
            }
        ]

        for strategy in strategies:
            status_color = {'Active': '#28a745', 'Paused': '#ffc107', 'Stopped': '#dc3545'}[strategy['status']]

            st.markdown(f"""
            <div class="strategy-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="color: #1f77b4; margin: 0;">{strategy['name']}</h4>
                    <span style="background: {status_color}; color: white; padding: 0.2rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">
                        {strategy['status']}
                    </span>
                </div>
                <p style="color: #666; margin: 0.5rem 0;">{strategy['description']}</p>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div>
                        <strong>Positions:</strong><br>
                        <span style="font-size: 1.2rem; color: #1f77b4;">{strategy['positions']}</span>
                    </div>
                    <div>
                        <strong>Daily P&L:</strong><br>
                        <span style="font-size: 1.2rem; color: {'green' if strategy['daily_pnl'] > 0 else 'red'};">
                            ${strategy['daily_pnl']:+,.0f}
                        </span>
                    </div>
                    <div>
                        <strong>Success Rate:</strong><br>
                        <span style="font-size: 1.2rem; color: #1f77b4;">{strategy['success_rate']:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Strategy controls
            col1, col2, col3 = st.columns(3)
            with col1:
                st.button(f"⚙️ Configure", key=f"config_{strategy['name']}")
            with col2:
                if strategy['status'] == 'Active':
                    st.button(f"⏸️ Pause", key=f"pause_{strategy['name']}")
                else:
                    st.button(f"▶️ Resume", key=f"resume_{strategy['name']}")
            with col3:
                st.button(f"📊 Details", key=f"details_{strategy['name']}")

    def render_performance_tab(self):
        """Render performance analysis tab"""

        col1, col2 = st.columns(2)

        with col1:
            # Monthly performance
            months = pd.date_range('2024-01-01', '2024-03-01', freq='M')
            monthly_returns = np.random.normal(3.2, 8.5, len(months))

            fig = go.Figure(data=[
                go.Bar(x=months, y=monthly_returns,
                      marker_color=['green' if x > 0 else 'red' for x in monthly_returns])
            ])

            fig.update_layout(
                title="📊 Monthly Returns (%)",
                xaxis_title="Month",
                yaxis_title="Return (%)",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Strategy allocation pie chart
            strategy_values = {
                'Model Ensemble': 45000,
                'Gamma Scalping': 28000,
                'Dispersion': 15000,
                'Cash': 37000
            }

            fig = go.Figure(data=[go.Pie(
                labels=list(strategy_values.keys()),
                values=list(strategy_values.values()),
                hole=0.4
            )])

            fig.update_layout(
                title="💼 Portfolio Allocation",
                height=350,
                annotations=[dict(text='$125K<br>Total', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )

            st.plotly_chart(fig, use_container_width=True)

        # Performance metrics table
        st.markdown("### 📈 Key Performance Metrics")

        metrics_data = {
            'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate',
                      'Profit Factor', 'Average Win', 'Average Loss'],
            'Value': ['25.3%', '1.84', '-8.2%', '68.5%', '2.1', '$485', '-$230'],
            'Benchmark': ['18.4%', '1.12', '-12.1%', '52.3%', '1.4', '$340', '-$242']
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    def render_model_ensemble_tab(self):
        """Render model ensemble performance tab"""

        st.markdown("### 🤖 Model Ensemble Performance")

        col1, col2 = st.columns(2)

        with col1:
            # Model weights chart
            models = list(self.mock_data['model_performance'].keys())
            weights = [0.25, 0.30, 0.25, 0.20]  # Current weights

            fig = go.Figure(data=[
                go.Bar(x=[m.replace('_', ' ').title() for m in models],
                      y=weights,
                      marker_color='#1f77b4')
            ])

            fig.update_layout(
                title="⚖️ Current Model Weights",
                yaxis_title="Weight",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Model accuracy over time
            days = pd.date_range('2024-03-01', '2024-03-15', freq='D')

            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

            for i, model in enumerate(models):
                accuracy = self.mock_data['model_performance'][model]
                daily_acc = accuracy + np.random.normal(0, 0.05, len(days))
                daily_acc = np.clip(daily_acc, 0, 1)

                fig.add_trace(go.Scatter(
                    x=days, y=daily_acc,
                    mode='lines+markers',
                    name=model.replace('_', ' ').title(),
                    line=dict(color=colors[i])
                ))

            fig.update_layout(
                title="🎯 Model Accuracy Trends",
                yaxis_title="Accuracy",
                height=350,
                yaxis=dict(range=[0.5, 1.0])
            )

            st.plotly_chart(fig, use_container_width=True)

        # Model consensus analysis
        st.markdown("### 🧠 Current Model Predictions")

        # Sample options for analysis
        sample_options = [
            {'symbol': 'AAPL', 'strike': 175, 'expiry': '2024-04-19', 'type': 'Call'},
            {'symbol': 'TSLA', 'strike': 200, 'expiry': '2024-03-22', 'type': 'Put'},
            {'symbol': 'SPY', 'strike': 520, 'expiry': '2024-04-12', 'type': 'Call'}
        ]

        predictions_data = []
        for option in sample_options:
            base_price = np.random.uniform(2, 12)
            model_preds = {
                'Symbol': f"{option['symbol']} {option['strike']}{option['type'][0]} {option['expiry']}",
                'Market Price': f"${base_price:.2f}",
                'Black-Scholes': f"${base_price * np.random.uniform(0.95, 1.05):.2f}",
                'Merton Jump': f"${base_price * np.random.uniform(0.90, 1.10):.2f}",
                'Heston': f"${base_price * np.random.uniform(0.92, 1.08):.2f}",
                'ML Neural': f"${base_price * np.random.uniform(0.96, 1.04):.2f}",
                'Consensus': f"${base_price * np.random.uniform(0.98, 1.02):.2f}",
                'Edge': f"{np.random.uniform(-5, 8):.1f}%"
            }
            predictions_data.append(model_preds)

        predictions_df = pd.DataFrame(predictions_data)
        st.dataframe(predictions_df, use_container_width=True, hide_index=True)

    def render_risk_analysis_tab(self):
        """Render risk analysis tab"""

        st.markdown("### ⚖️ Risk Analysis & Controls")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Risk metrics
            st.markdown("#### 🛡️ Risk Metrics")

            risk_metrics = {
                'Portfolio VaR (95%)': '$2,450',
                'Expected Shortfall': '$3,820',
                'Maximum Drawdown': '8.2%',
                'Beta to SPY': '0.65',
                'Correlation to QQQ': '0.72'
            }

            for metric, value in risk_metrics.items():
                st.metric(metric, value)

        with col2:
            # Position limits
            st.markdown("#### 📊 Position Limits")

            limits = {
                'Max Position Size': ('18.5%', '25.0%'),
                'Portfolio Delta': ('0.32', '0.50'),
                'Daily Loss Limit': ('1.2%', '5.0%'),
                'Leverage Ratio': ('1.4x', '2.0x'),
                'Concentration Risk': ('Low', 'Medium')
            }

            for limit, (current, maximum) in limits.items():
                if limit == 'Concentration Risk':
                    color = 'green' if current == 'Low' else 'orange'
                    st.markdown(f"**{limit}:** <span style='color: {color}'>{current}</span>",
                              unsafe_allow_html=True)
                else:
                    progress = float(current.replace('%', '').replace('x', '')) / float(maximum.replace('%', '').replace('x', ''))
                    st.markdown(f"**{limit}:** {current} / {maximum}")
                    st.progress(min(progress, 1.0))

        with col3:
            # Risk controls
            st.markdown("#### ⚙️ Risk Controls")

            if st.button("🛑 Emergency Stop All", type="secondary", use_container_width=True):
                st.error("Emergency stop activated!")

            if st.button("🔄 Rebalance Delta", type="secondary", use_container_width=True):
                st.info("Delta rebalancing initiated...")

            if st.button("📊 Stress Test", type="secondary", use_container_width=True):
                st.info("Running stress test scenarios...")

            # Risk level adjustment
            st.markdown("**Risk Level Override:**")
            override_risk = st.selectbox(
                "Temporary Risk Level",
                options=['default', 'conservative', 'aggressive'],
                index=0,
                key='risk_override'
            )

        # Greeks exposure chart
        st.markdown("### 📈 Greeks Exposure Over Time")

        dates = pd.date_range('2024-03-01', '2024-03-15', freq='D')

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta Exposure', 'Gamma Exposure', 'Theta Exposure', 'Vega Exposure'),
            vertical_spacing=0.1
        )

        # Generate sample Greeks data
        greeks_data = {
            'delta': np.random.normal(0.2, 0.1, len(dates)),
            'gamma': np.random.normal(0.05, 0.02, len(dates)),
            'theta': np.random.normal(-150, 50, len(dates)),
            'vega': np.random.normal(200, 80, len(dates))
        }

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, (greek, values) in enumerate(greeks_data.items()):
            row, col = positions[i]
            fig.add_trace(
                go.Scatter(x=dates, y=values, mode='lines',
                          name=greek.title(), line=dict(color=colors[i])),
                row=row, col=col
            )

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def render_trade_log_tab(self):
        """Render trade log tab"""

        st.markdown("### 📋 Trade Execution Log")

        # Filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            date_filter = st.date_input("From Date", datetime.now() - timedelta(days=30))
        with col2:
            symbol_filter = st.multiselect("Symbols", ['All', 'AAPL', 'TSLA', 'MSFT', 'SPY'], ['All'])
        with col3:
            strategy_filter = st.multiselect("Strategies",
                                           ['All', 'Model Ensemble', 'Gamma Scalping', 'Dispersion'], ['All'])
        with col4:
            action_filter = st.selectbox("Action", ['All', 'BUY', 'SELL'])

        # Trade log table
        trades_df = pd.DataFrame(self.mock_data['trade_history'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp', ascending=False)

        # Apply filters
        if 'All' not in symbol_filter and symbol_filter:
            trades_df = trades_df[trades_df['symbol'].isin(symbol_filter)]

        if 'All' not in strategy_filter and strategy_filter:
            trades_df = trades_df[trades_df['strategy'].isin(strategy_filter)]

        if action_filter != 'All':
            trades_df = trades_df[trades_df['action'] == action_filter]

        # Format for display
        display_df = trades_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:+,.0f}")

        # Color code P&L
        def highlight_pnl(row):
            if 'pnl' in row.name:
                value = float(row.replace('$', '').replace('+', '').replace(',', ''))
                color = 'background-color: lightgreen' if value > 0 else 'background-color: lightcoral'
                return [color] * len(row)
            return [''] * len(row)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Trade statistics
        st.markdown("### 📊 Trading Statistics")

        col1, col2, col3, col4 = st.columns(4)

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0

        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Avg Win", f"${avg_win:.0f}")
        with col4:
            st.metric("Avg Loss", f"${avg_loss:.0f}")

    def run(self):
        """Main application runner"""
        self.render_sidebar()
        self.render_main_dashboard()

        # Auto-refresh functionality
        if st.session_state.auto_refresh:
            time.sleep(30)
            st.rerun()

# Initialize and run dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
