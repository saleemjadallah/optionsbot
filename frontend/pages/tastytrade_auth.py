"""
Tastytrade Authentication Page

Streamlit page that manages session-based authentication with the
Tastytrade sandbox directly from the frontend.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tastytrade_auth import get_auth_manager
from utils.api_client import TradingBotAPI


def _safe_number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

# Page configuration
st.set_page_config(
    page_title="Tastytrade Authentication",
    page_icon="🔐",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
<style>
    .auth-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
    }

    .status-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    .status-connected {
        border-left: 5px solid #4caf50;
    }

    .status-disconnected {
        border-left: 5px solid #f44336;
    }

    .account-card {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }

    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }

    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("# 🔐 Tastytrade Authentication")
st.markdown("---")

# Initialize auth manager
auth_manager = get_auth_manager()
api_client = TradingBotAPI()

if "tastytrade_environment" not in st.session_state:
    st.session_state.tastytrade_environment = "sandbox"

# Check current authentication status
with st.spinner("Checking authentication status..."):
    status = auth_manager.check_auth_status()

if status.get("token_valid"):
    env_label = "SANDBOX" if getattr(auth_manager, "is_test", True) else "PRODUCTION"
    st.markdown(
        f"""
        <div class="status-card status-connected">
            <h3>✅ Connected to Tastytrade</h3>
            <p>{env_label.title()} environment authenticated and ready for API calls.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📊 Account Overview")
    st.markdown(
        f"""
        <div class="account-card">
            <strong>Environment:</strong> {env_label}<br>
            <strong>Account Number:</strong> {status.get('account_number', 'N/A')}<br>
            <strong>Session Expires:</strong> {status.get('token_expiry', 'Unknown')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading account details..."):
            accounts = auth_manager.get_accounts()
    except Exception as exc:
        accounts = []
        st.error(f"Failed to load accounts: {exc}")

    if accounts:
        st.markdown("### 💼 Linked Accounts")
        for account in accounts:
            account_num = account.get("account-number")
            authority = account.get("authority-level")
            account_type = account.get("account-type")
            nickname = account.get("nickname", "N/A")

            with st.expander(f"Account {account_num}", expanded=True):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Type", account_type)
                with col_b:
                    st.metric("Authority", authority)
                with col_c:
                    st.metric("Nickname", nickname)

                try:
                    balance = auth_manager.get_account_balance(account_num)
                except Exception as exc:
                    balance = None
                    st.error(f"Failed to fetch balance: {exc}")

                if balance:
                    st.markdown("#### Balance Snapshot")
                    bal_col1, bal_col2, bal_col3 = st.columns(3)
                    with bal_col1:
                        cash = _safe_number(balance.get("cash-balance"))
                        st.metric("Cash Balance", f"${cash:,.2f}")
                    with bal_col2:
                        nlv = _safe_number(balance.get("net-liquidating-value"))
                        st.metric("Net Liquidating Value", f"${nlv:,.2f}")
                    with bal_col3:
                        bp = _safe_number(balance.get("equity-buying-power"))
                        st.metric("Buying Power", f"${bp:,.2f}")

                show_positions = st.checkbox(
                    "Show Positions",
                    key=f"positions_{account_num}",
                    value=False,
                )
                if show_positions:
                    try:
                        positions = auth_manager.get_account_positions(account_num)
                    except Exception as exc:
                        positions = []
                        st.error(f"Failed to fetch positions: {exc}")

                    if positions:
                        st.markdown("#### Open Positions")
                        for pos in positions:
                            symbol = pos.get("symbol")
                            qty = pos.get("quantity")
                            avg_price = _safe_number(pos.get("average-open-price"))
                            pnl = _safe_number(pos.get("unrealized-gain-loss"))
                            pnl_color = "green" if pnl >= 0 else "red"

                            st.markdown(
                                f"""
                                <div style="background: #f5f5f5; padding: 0.5rem; margin: 0.25rem 0; border-radius: 3px;">
                                    <strong>{symbol}</strong>: {qty} @ ${avg_price:.2f} |
                                    P&L: <span style="color: {pnl_color};">${pnl:.2f}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No open positions for this account.")

    strategy_context: Dict[str, List[Dict]] = {}
    if api_client.can_use_tastytrade():
        try:
            strategy_context = api_client.get_strategy_context()
        except Exception as exc:
            st.error(f"Failed to compute strategy context: {exc}")
            strategy_context = {}

    exposures = strategy_context.get("exposures", []) if strategy_context else []
    if exposures:
        st.markdown("### 📊 Exposure Summary")
        exp_df = pd.DataFrame(exposures)
        column_map = {
            "symbol": "Symbol",
            "long_shares": "Long Shares",
            "short_shares": "Short Shares",
            "net_quantity": "Net Shares",
            "net_delta": "Net Delta (sh)",
            "net_gamma": "Net Gamma",
            "net_theta": "Net Theta",
            "net_vega": "Net Vega",
            "unrealized_pnl": "Unrealized P&L",
        }
        cols_present = [c for c in column_map if c in exp_df.columns]
        st.dataframe(exp_df[cols_present].rename(columns=column_map), use_container_width=True)
        st.markdown("---")

    st.markdown("### 📈 Trade Opportunities")
    live_opps = strategy_context.get("ideas", []) if strategy_context else []
    universe_opps = strategy_context.get("universe_ideas", []) if strategy_context else []

    if live_opps:
        st.markdown("**Live Position Plays**")
        for opp in live_opps:
            header = f"{opp.get('symbol')} — {opp.get('suggested_strategy')}"
            with st.expander(header, expanded=False):
                st.write(f"**Signal:** {opp.get('signal')}")
                st.write(opp.get("rationale"))
                st.write(f"**Idea:** {opp.get('trade_idea')}")
                order_example = opp.get("order_example")
                if order_example:
                    st.caption("Example order structure (adjust before submitting):")
                    st.code(json.dumps(order_example, indent=2), language="json")

    if universe_opps:
        st.markdown("**Universe Scan Opportunities**")
        active_symbols = strategy_context.get("universe_symbols", [])
        if active_symbols:
            st.caption(f"Scanning: {', '.join(active_symbols)}")
        for opp in universe_opps:
            header = f"{opp.get('symbol')} — {opp.get('suggested_strategy')}"
            with st.expander(header, expanded=False):
                st.write(f"**Signal:** {opp.get('signal')}")
                st.write(opp.get("rationale"))
                st.write(f"**Idea:** {opp.get('trade_idea')}")
                order_example = opp.get("order_example")
                if order_example:
                    st.caption("Sample order (verify before use):")
                    st.code(json.dumps(order_example, indent=2), language="json")

    if not live_opps and not universe_opps:
        st.info("No trade opportunities generated yet.")

    st.markdown("### 🛒 Trade Execution")
    st.caption("Always run a dry-run preview before placing any order.")

    if "tt_last_order_payload_page" not in st.session_state:
        st.session_state.tt_last_order_payload_page = None
    if "tt_last_dry_run_page" not in st.session_state:
        st.session_state.tt_last_dry_run_page = None

    with st.form("tastytrade_order_form_page"):
        instrument_type = st.selectbox("Instrument Type", ["Equity", "Equity Option"], key="order_inst_page")
        symbol_input = st.text_input(
            "Symbol / Option Identifier",
            help="For options, use the full tastytrade option symbol (e.g., AAPL  240621C195).",
            key="order_symbol_page"
        )
        action = st.selectbox(
            "Action",
            ["Buy to Open", "Sell to Open", "Buy to Close", "Sell to Close", "Buy", "Sell"],
            key="order_action_page"
        )
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="order_qty_page")
        order_type = st.selectbox("Order Type", ["Market", "Limit"], key="order_type_page")
        limit_price = None
        if order_type == "Limit":
            limit_price = st.number_input("Limit Price", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="order_price_page")
        time_in_force = st.selectbox("Time in Force", ["Day", "GTC", "GTD"], key="order_tif_page")
        source = st.text_input("Order Source", value="streamlit-app", key="order_source_page")

        preview_clicked = st.form_submit_button("Preview Order (Dry Run)")

        if preview_clicked:
            try:
                if not symbol_input.strip():
                    raise ValueError("Symbol is required.")
                payload = {
                    "time-in-force": time_in_force,
                    "order-type": order_type,
                    "legs": [
                        {
                            "instrument-type": instrument_type,
                            "symbol": symbol_input.strip(),
                            "quantity": int(quantity),
                            "action": action,
                        }
                    ],
                    "source": source.strip() or "streamlit-app",
                }
                if order_type == "Limit":
                    if not limit_price or limit_price <= 0:
                        raise ValueError("Provide a positive limit price.")
                    payload["price"] = round(float(limit_price), 2)

                preview = auth_manager.dry_run_order(payload)
                st.session_state.tt_last_order_payload_page = payload
                st.session_state.tt_last_dry_run_page = preview
                st.success("Dry run successful. Review the details below.")
            except Exception as exc:
                st.session_state.tt_last_order_payload_page = None
                st.session_state.tt_last_dry_run_page = None
                st.error(f"Dry run failed: {exc}")

    if st.session_state.get("tt_last_dry_run_page"):
        st.subheader("Dry Run Result")
        st.json(st.session_state.tt_last_dry_run_page)

    if st.session_state.get("tt_last_order_payload_page"):
        if st.button("🚀 Place Order Now", key="place_order_page"):
            try:
                result = auth_manager.place_order(st.session_state.tt_last_order_payload_page)
                st.success("Order placed successfully!")
                st.json(result)
                st.session_state.tt_last_order_payload_page = None
                st.session_state.tt_last_dry_run_page = None
            except Exception as exc:
                st.error(f"Order placement failed: {exc}")

    st.markdown(
        """
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">24h</div>
                <div class="stat-label">Session Lifetime</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{env_label}</div>
                <div class="stat-label">Environment</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">Ready</div>
                <div class="stat-label">API Access</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔄 Session Controls")
    control_col1, control_col2 = st.columns(2)
    with control_col1:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    with control_col2:
        if st.button("🚪 Logout", key="tastytrade_auth_page_logout"):
            try:
                auth_manager.logout()
                st.success("Logged out successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as exc:
                st.error(f"Logout failed: {exc}")

    st.markdown(
        """
        <div class="success-box">
            <h4>✅ You're Securely Connected</h4>
            <p>Your session token is kept in Streamlit session state and will expire automatically after 24 hours.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="status-card status-disconnected">
            <h3>🔒 Not Connected</h3>
            <p>Sign in with your Tastytrade sandbox credentials to enable API access.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
            <h4>ℹ️ About Login</h4>
            <p>Both sandbox and production use direct username/password sessions (no OAuth in sandbox).</p>
            <p>Your credentials are only used to request a session token and are not stored by the app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("tastytrade_login_form_page", clear_on_submit=False):
        env_default = st.session_state.get("tastytrade_environment", "sandbox")
        env_label = st.selectbox(
            "Tastytrade Environment",
            ["Sandbox", "Production"],
            index=0 if env_default == "sandbox" else 1,
        )
        environment = "production" if env_label == "Production" else "sandbox"

        default_username = st.session_state.get("tastytrade_last_username", "")
        username = st.text_input("Username", value=default_username)
        password = st.text_input("Password", type="password")
        remember = st.checkbox("Keep me logged in for 24 hours", value=True)
        submitted = st.form_submit_button("Sign in to Tastytrade")

        if submitted:
            st.session_state.tastytrade_environment = environment
            if not username or not password:
                st.error("Username and password are required.")
            else:
                try:
                    auth_manager.login(
                        username,
                        password,
                        remember_me=remember,
                        environment=environment,
                    )
                    st.session_state.tastytrade_last_username = username
                    st.session_state.tastytrade_environment = environment
                    st.success("Authenticated successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Authentication failed: {exc}")

    st.markdown(
        """
        <div class="warning-box">
            <h4>Need Help?</h4>
            <p>
                Choose <strong>Sandbox</strong> for the cert environment (paper trading) or <strong>Production</strong> for live accounts.
                Confirm your username/password are correct for the environment you select.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("Session tokens are stored in Streamlit session state only and cleared when you log out or the session expires.")
