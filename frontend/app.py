"""
Live Options Trading Dashboard

The main Streamlit application that pulls data directly from Tastytrade and
surfaces portfolio analytics, trade ideas, risk metrics, and order routing.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from utils.api_client import TradingBotAPI
from utils.strategy_engine import StrategyEngine
from ui.chat_interface import render_chat_interface
from config.universe import get_default_universe, parse_symbol_list
# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Options Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HEADER_STYLE = """
<style>
    .status-banner {
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .status-ok {
        background-color: #e8f5e9;
        color: #1b5e20;
    }
    .status-warn {
        background-color: #fff8e1;
        color: #f57c00;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    .analysis-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        margin: 0.3rem 0 0.8rem;
        color: white;
    }
    .analysis-tag.on {
        background: linear-gradient(120deg, #1b5e20, #388e3c);
    }
    .analysis-tag.off {
        background: linear-gradient(120deg, #b71c1c, #f44336);
    }
</style>
"""
st.markdown(HEADER_STYLE, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Core application
# ----------------------------------------------------------------------

class LiveTradingApp:
    def __init__(self):
        self.api = TradingBotAPI()
        self.initialize_session_state()
        self.snapshot: Optional[Dict] = st.session_state.last_snapshot
        self.last_refresh_error: Optional[str] = None
        self.universe_min_edge = st.session_state.get(
            "universe_min_edge",
            StrategyEngine.ENSEMBLE_EDGE_THRESHOLD * 100,
        )
        self.universe_min_conf = st.session_state.get(
            "universe_min_conf",
            StrategyEngine.ENSEMBLE_CONFIDENCE_THRESHOLD * 100,
        )

    # ------------------------------------------------------------------
    # Session state & data collection
    # ------------------------------------------------------------------

    def initialize_session_state(self) -> None:
        if "auto_refresh" not in st.session_state:
            st.session_state.auto_refresh = False
        if "last_snapshot" not in st.session_state:
            st.session_state.last_snapshot = None
        if "last_snapshot_at" not in st.session_state:
            st.session_state.last_snapshot_at = None
        if "tt_order_payload" not in st.session_state:
            st.session_state.tt_order_payload = None
        if "tt_order_preview" not in st.session_state:
            st.session_state.tt_order_preview = None
        if "universe_symbols" not in st.session_state:
            defaults = get_default_universe()
            st.session_state.universe_symbols = defaults
            st.session_state.universe_input = ", ".join(defaults)
        elif "universe_input" not in st.session_state:
            st.session_state.universe_input = ", ".join(st.session_state.universe_symbols)
        if "watchlist_name" not in st.session_state:
            st.session_state.watchlist_name = ""
        if "universe_min_edge" not in st.session_state:
            st.session_state.universe_min_edge = StrategyEngine.ENSEMBLE_EDGE_THRESHOLD * 100
        if "universe_min_conf" not in st.session_state:
            st.session_state.universe_min_conf = StrategyEngine.ENSEMBLE_CONFIDENCE_THRESHOLD * 100
        if "idea_dry_runs" not in st.session_state:
            st.session_state.idea_dry_runs = {}
        if "favorite_cache" not in st.session_state:
            st.session_state.favorite_cache = {"loaded": False, "items": []}

    def _on_universe_change(self) -> None:
        """
        Update the cached universe list when the sidebar input changes.
        """
        raw_value = st.session_state.get("universe_input", "")
        parsed = parse_symbol_list(raw_value.split(","))
        if parsed:
            st.session_state.universe_symbols = parsed
        else:
            st.session_state.universe_symbols = get_default_universe()

    def collect_snapshot(self) -> Optional[Dict]:
        if not self.api.can_use_tastytrade():
            self.last_refresh_error = None
            return None

        try:
            universe = st.session_state.get("universe_symbols", get_default_universe())
            min_edge = st.session_state.get("universe_min_edge")
            min_conf = st.session_state.get("universe_min_conf")
            edge_threshold = (min_edge / 100.0) if isinstance(min_edge, (int, float)) else None
            confidence_threshold = (min_conf / 100.0) if isinstance(min_conf, (int, float)) else None
            strategy_context = self.api.get_strategy_context(
                universe_symbols=universe,
                edge_threshold=edge_threshold,
                confidence_threshold=confidence_threshold,
            )
            data = {
                "portfolio": self.api.get_portfolio_status(),
                "positions": strategy_context.get("positions", []),
                "risk": self.api.get_risk_metrics(),
                "strategy": strategy_context,
            }
            st.session_state.last_snapshot = data
            st.session_state.last_snapshot_at = datetime.now(timezone.utc)
            self.last_refresh_error = None
            return data
        except Exception as exc:
            self.last_refresh_error = str(exc)
            if st.session_state.last_snapshot:
                return st.session_state.last_snapshot
            return None

    def refresh_snapshot(self, force: bool = False) -> Optional[Dict]:
        """
        Refresh cached portfolio snapshots. When force=True, drop stale data if refresh fails.
        """
        snapshot = self.collect_snapshot()
        if snapshot is not None:
            self.snapshot = snapshot
        elif force:
            self.snapshot = None
        return self.snapshot

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def render_header(self) -> None:
        if st.session_state.get("auto_refresh") or self.snapshot is None:
            self.refresh_snapshot(force=False)

        st.markdown(
            """
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                        padding: 1.2rem 1.6rem; border-radius: 12px; color: white; margin-bottom: 1rem;">
                <div style="font-size: 2rem; font-weight: 600;">🤖 Options Trading Control Center</div>
                <div style="opacity: 0.85;">Live Tastytrade portfolio analytics, strategy insights, and execution.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if self.snapshot and self.api.can_use_tastytrade():
            portfolio = self.snapshot.get("portfolio", {})
            account = portfolio.get("account_number")
            pnl = portfolio.get("daily_pnl")
            env_label = "SANDBOX" if getattr(self.api.auth_manager, "is_test", True) else "PRODUCTION"
            if account is not None and pnl is not None:
                pnl_color = "#1b5e20" if pnl >= 0 else "#b71c1c"
                banner = (
                    f"<div class='status-banner status-ok'>"
                    f"Connected to Tastytrade ({env_label}) • Account <strong>{account}</strong> • "
                    f"Daily P&L <span style='color:{pnl_color};'>${pnl:+,.2f}</span>"
                    f"</div>"
                )
                st.markdown(banner, unsafe_allow_html=True)
                return

        if self.api.can_use_tastytrade():
            st.markdown(
                "<div class='status-banner status-warn'>Connected to Tastytrade but portfolio data "
                "is not yet available. Try refreshing in a moment.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='status-banner status-warn'>Not connected to Tastytrade. "
                "Authenticate via the <strong>Tastytrade</strong> tab.</div>",
                unsafe_allow_html=True,
            )

        if self.last_refresh_error:
            st.warning(f"Latest data refresh failed: {self.last_refresh_error}")

    def render_navigation(self) -> str:
        with st.sidebar:
            selected = option_menu(
                "Navigation",
                [
                    "Overview",
                    "Portfolio",
                    "Strategies",
                    "Risk",
                    "Trade Terminal",
                    "Jeffrey",
                    "Tastytrade",
                ],
                icons=["speedometer", "briefcase", "diagram-3", "activity", "cursor", "robot", "key"],
                menu_icon="cast",
                default_index=0,
            )
            st.sidebar.checkbox("Auto-refresh data", key="auto_refresh")
            st.sidebar.text_input(
                "Universe Symbols",
                key="universe_input",
                on_change=self._on_universe_change,
            )
            st.sidebar.caption(
                "Comma-separated tickers used for proactive scans (defaults to sandbox watchlist)."
            )
            if self.api.can_use_tastytrade():
                if "watchlist_notice" in st.session_state:
                    st.sidebar.success(st.session_state.watchlist_notice)
                    del st.session_state.watchlist_notice
                if "watchlist_error" in st.session_state:
                    st.sidebar.error(st.session_state.watchlist_error)
                    del st.session_state.watchlist_error
                st.sidebar.text_input(
                    "Watchlist Name (optional)",
                    key="watchlist_name",
                    help="Leave blank to use the first watchlist returned by Tastytrade."
                )
            if st.sidebar.button("Load Watchlist Symbols"):
                self._load_watchlist_symbols()
            if st.sidebar.button("Refresh now"):
                st.session_state.last_snapshot = None
                self.snapshot = None
                st.rerun()
        return selected

    def ensure_snapshot(self) -> bool:
        """
        Ensure a recent snapshot exists. Returns True when data is available.
        """
        if not self.snapshot:
            self.refresh_snapshot(force=False)
        if not self.snapshot:
            st.info("Live data not available. Authenticate on the Tastytrade tab.")
            if self.last_refresh_error:
                st.warning(f"Snapshot refresh failed: {self.last_refresh_error}")
            return False
        return True

    def _load_watchlist_symbols(self) -> None:
        """Fetch symbols from the user's Tastytrade watchlist and update universe."""
        if not self.api.can_use_tastytrade():
            st.sidebar.warning("Authenticate on the Tastytrade tab to import watchlists.")
            return

        name = (st.session_state.get("watchlist_name") or "").strip()
        try:
            symbols = self.api.get_watchlist_symbols(name or None)
        except Exception as exc:
            st.session_state.watchlist_error = f"Failed to load watchlist: {exc}"
            return

        if not symbols:
            st.session_state.watchlist_error = "No symbols found in the selected watchlist."
            return

        st.session_state.universe_symbols = symbols
        self.snapshot = None
        st.session_state.watchlist_notice = (
            f"Loaded {len(symbols)} symbols from watchlist{' ' + name if name else ''}."
        )
        if "watchlist_error" in st.session_state:
            del st.session_state.watchlist_error
        st.rerun()

    def _render_market_state_tag(self, market_state: Optional[Dict[str, Any]]) -> None:
        """Display whether the ensemble snapshot is based on on/off market data."""
        if not market_state:
            return
        is_open = bool(market_state.get("is_open"))
        tag_class = "on" if is_open else "off"
        emoji = "🟢" if is_open else "🌙"
        label = market_state.get("label", "Market Status")
        basis = market_state.get("basis", "")
        st.markdown(
            f"<div class='analysis-tag {tag_class}'>{emoji} {label}</div>",
            unsafe_allow_html=True,
        )
        if basis:
            st.caption(basis)

    def _idea_cache_key(self, idea: Dict[str, Any], group: str) -> str:
        symbol = idea.get("symbol", "N/A")
        strategy = idea.get("suggested_strategy", "Unknown")
        unique_payload = {
            "trade": idea.get("trade_idea"),
            "legs": idea.get("order_example", {}).get("legs", []),
        }
        unique_str = json.dumps(unique_payload, sort_keys=True, default=str)
        digest = hashlib.sha256(unique_str.encode("utf-8")).hexdigest()[:10]
        return f"{group}:{symbol}:{strategy}:{digest}"

    def _favorite_key(self, idea: Dict[str, Any]) -> str:
        return f"{idea.get('symbol')}::{idea.get('suggested_strategy')}::{idea.get('trade_idea')}"

    def _ensure_favorites_loaded(self) -> None:
        cache = st.session_state.favorite_cache
        if cache.get("loaded"):
            return
        if not self.api.can_use_tastytrade():
            cache["items"] = []
            cache["loaded"] = True
            return
        cache["items"] = self.api.fetch_favorites()
        cache["loaded"] = True

    def _favorite_items(self) -> List[Dict[str, Any]]:
        return st.session_state.favorite_cache.get("items", [])

    def _reload_favorites(self) -> None:
        st.session_state.favorite_cache["loaded"] = False
        self._ensure_favorites_loaded()

    def _preview_order(self, payload: Dict[str, Any], cache_key: str) -> None:
        """Send a dry-run request to Tastytrade and store the response."""
        if not payload:
            st.warning("No order payload available for preview.")
            return
        if not self.api.can_use_tastytrade():
            st.warning("Authenticate with Tastytrade to preview orders.")
            return
        try:
            preview = self.api.dry_run_order(payload)
        except Exception as exc:
            st.error(f"Dry run failed: {exc}")
            return
        st.session_state.idea_dry_runs[cache_key] = {
            "payload": payload,
            "preview": preview,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        st.success("Sent to Tastytrade as a preview (no execution).")

    def _render_preview_controls(self, idea: Dict[str, Any], group: str, idx: int = 0) -> None:
        order_payload = idea.get("order_example")
        cache_key = self._idea_cache_key(idea, group)
        # Add index to ensure unique button keys even if cache_key is identical
        button_key = f"preview_{cache_key}_{idx}"
        disabled = order_payload is None
        button_label = "Preview in Tastytrade" if not disabled else "No Order Example Available"
        if st.button(button_label, key=button_key, disabled=disabled):
            self._preview_order(order_payload, cache_key)
        cached = st.session_state.idea_dry_runs.get(cache_key)
        if cached:
            st.caption("Latest Tastytrade dry-run (not executed):")
            st.json(cached.get("preview"))

    def _compute_star_rating(self, idea: Dict[str, Any]) -> int:
        metrics = idea.get("metrics", {})
        edge = abs(metrics.get("edge_pct", 0.0))
        conf = metrics.get("confidence_pct", 0.0)
        consensus_gap = abs(metrics.get("consensus_price", 0.0) - metrics.get("market_price", 0.0))

        edge_norm = min(edge / 10.0, 1.0)  # 10% edge -> full credit
        conf_norm = min(conf / 80.0, 1.0)  # 80% confidence -> full
        gap_norm = min(consensus_gap / 2.0, 1.0)  # $2 gap -> full

        score = 1.0 + edge_norm * 2.0 + conf_norm * 1.5 + gap_norm * 0.5
        return max(1, min(5, round(score)))

    def _star_badge(self, stars: int) -> str:
        return "★" * stars + "☆" * (5 - stars)

    def _render_favorite_button(self, idea: Dict[str, Any], key_suffix: str) -> None:
        self._ensure_favorites_loaded()
        fav_key = self._favorite_key(idea)
        items = self._favorite_items()
        exists = any(entry.get("idea_id") == fav_key for entry in items)
        if exists:
            if st.button("★ Remove Favorite", key=f"fav_remove_{key_suffix}"):
                try:
                    removed = self.api.delete_favorite(fav_key)
                except RuntimeError as exc:
                    st.error(f"Failed to remove favorite: {exc}")
                else:
                    if removed:
                        self._reload_favorites()
                    st.rerun()
        else:
            if st.button("☆ Save as Favorite", key=f"fav_save_{key_suffix}"):
                try:
                    saved = self.api.save_favorite(fav_key, idea)
                except RuntimeError as exc:
                    st.error(f"Failed to save favorite: {exc}")
                else:
                    if saved:
                        self._reload_favorites()
                        st.success("Saved to favorites.")
                    else:
                        st.warning("Favorites service unavailable.")

    def _render_favorites_section(self) -> None:
        self._ensure_favorites_loaded()
        favorites = self._favorite_items()
        st.markdown("### ⭐️ Saved Favorite Ideas")
        if not favorites:
            st.info("No favorites saved yet. Use the ☆ button on any idea to pin it here.")
            return
        for fav in favorites:
            idea = fav.get("snapshot", {})
            stars = self._star_badge(self._compute_star_rating(idea))
            with st.expander(f"{idea.get('symbol')} — {idea.get('suggested_strategy')} {stars}", expanded=False):
                st.write(f"**Signal:** {idea.get('signal')}")
                st.write(idea.get("rationale"))
                st.write(idea.get("trade_idea"))
                metrics = idea.get("metrics", {})
                if metrics:
                    st.table(self._metrics_table(metrics))
                st.caption("Saved snapshot. Re-run ensemble for up-to-date pricing before trading.")

    # ------------------------------------------------------------------
    # Page renderers
    # ------------------------------------------------------------------

    def render_overview(self) -> None:
        if not self.ensure_snapshot():
            return
        portfolio = self.snapshot["portfolio"]
        positions = self.snapshot["positions"]
        strategy_context = self.snapshot.get("strategy", {})
        opportunities = strategy_context.get("ideas", [])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Value", f"${portfolio['portfolio_value']:,.2f}")
        col2.metric("Available Cash", f"${portfolio['available_cash']:,.2f}")
        col3.metric("Buying Power", f"${portfolio['buying_power']:,.2f}")
        pnl_delta = f"{portfolio['daily_pnl']:+,.2f}"
        col4.metric("Daily P&L", f"${portfolio['daily_pnl']:,.2f}", pnl_delta)

        st.markdown("---")

        st.subheader("Top Positions by Unrealized P&L")
        if positions:
            df = self._positions_dataframe(positions)
            top_df = df.sort_values("Unrealized P&L", ascending=False).head(10)
            st.dataframe(top_df, use_container_width=True)

            fig = px.bar(
                top_df,
                x="Symbol",
                y="Unrealized P&L",
                color="Unrealized P&L",
                color_continuous_scale=["#e74c3c", "#f1c40f", "#2ecc71"],
                title="Unrealized P&L by Symbol",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions yet.")

        st.markdown("---")

        st.subheader("Strategy Ideas")
        live_ideas = opportunities
        universe_ideas = strategy_context.get("universe_ideas", [])

        if not live_ideas and not universe_ideas:
            st.info("No strategy suggestions generated yet. Universe scans refresh after authentication.")

        if live_ideas:
            st.markdown("**Live Position Plays**")
            for idea in live_ideas:
                with st.expander(f"{idea['symbol']} • {idea['suggested_strategy']}"):
                    st.write(f"**Signal:** {idea['signal']}")
                    st.write(idea["rationale"])
                    st.write(idea["trade_idea"])
                    metrics = idea.get("metrics", {})
                    if metrics:
                        st.table(self._metrics_table(metrics))
                    st.caption("Example structure (adjust symbol/strikes before trading):")
                    st.code(json.dumps(idea["order_example"], indent=2), language="json")

        if universe_ideas:
            st.markdown("**Universe Opportunities**")
            active_universe = strategy_context.get("universe_symbols", [])
            if active_universe:
                st.caption(f"Scanning: {', '.join(active_universe)}")
            for idea in universe_ideas:
                with st.expander(f"{idea['symbol']} • {idea['suggested_strategy']} ({idea.get('source', 'Universe')})"):
                    st.write(f"**Signal:** {idea['signal']}")
                    st.write(idea["rationale"])
                    st.write(idea["trade_idea"])
                    metrics = idea.get("metrics", {})
                        if metrics:
                            st.table(self._metrics_table(metrics))
                    st.caption("Sample order (validate prices before trading):")
                    st.code(json.dumps(idea["order_example"], indent=2), language="json")

    def render_portfolio(self) -> None:
        if not self.ensure_snapshot():
            return
        positions = self.snapshot["positions"]
        portfolio = self.snapshot["portfolio"]

        st.subheader("Account Overview")
        st.write(f"**Account Number:** {portfolio['account_number']}")
        st.write(
            f"**Last Refreshed:** "
            f"{st.session_state.last_snapshot_at.strftime('%Y-%m-%d %H:%M:%S UTC') if st.session_state.last_snapshot_at else 'N/A'}"
        )

        if not positions:
            st.info("No open positions for this account.")
            return

        df = self._positions_dataframe(positions)
        st.subheader("Open Positions")
        st.dataframe(df, use_container_width=True)

        st.subheader("Allocation by Underlying")
        alloc = df.groupby("Symbol")["Abs Exposure"].sum().reset_index()
        fig = px.pie(alloc, names="Symbol", values="Abs Exposure", title="Gross Exposure by Symbol")
        st.plotly_chart(fig, use_container_width=True)

    def render_strategies(self) -> None:
        if not self.ensure_snapshot():
            return
        strategy_context = self.snapshot.get("strategy", {})
        opportunities = strategy_context.get("ideas", [])
        exposures = strategy_context.get("exposures", [])

        st.subheader("Underlying Exposure Summary")
        if exposures:
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
            available_cols = [c for c in column_map if c in exp_df.columns]
            st.dataframe(
                exp_df[available_cols].rename(columns=column_map),
                use_container_width=True,
            )
        else:
            st.info("No significant equity exposure detected.")

        universe_ideas = strategy_context.get("universe_ideas", [])
        all_ideas = opportunities + universe_ideas
        if all_ideas:
            symbols = sorted({idea.get("symbol", "") for idea in all_ideas if idea.get("symbol")})
            strategies = sorted({idea.get("suggested_strategy", "") for idea in all_ideas if idea.get("suggested_strategy")})
            categories = sorted({idea.get("category", "Other") for idea in all_ideas})

            col_sym, col_strat, col_cat = st.columns(3)
            selected_symbols = col_sym.multiselect(
                "Filter by symbol",
                options=symbols,
                default=symbols,
            ) if symbols else []
            selected_strategies = col_strat.multiselect(
                "Filter by strategy",
                options=strategies,
                default=strategies,
            ) if strategies else []
            selected_categories = col_cat.multiselect(
                "Filter by category",
                options=categories,
                default=categories,
            ) if categories else []

            def _filter(ideas: List[Dict]) -> List[Dict]:
                filtered = []
                for idea in ideas:
                    symbol_ok = (not selected_symbols) or (idea.get("symbol") in selected_symbols)
                    strat_ok = (not selected_strategies) or (idea.get("suggested_strategy") in selected_strategies)
                    category_ok = (not selected_categories) or (idea.get("category", "Other") in selected_categories)
                    if symbol_ok and strat_ok and category_ok:
                        filtered.append(idea)
                return filtered

            opportunities = _filter(opportunities)
            universe_ideas = _filter(universe_ideas)

        st.markdown("---")

        st.subheader("Live Strategy Recommendations")
        st.caption("Recommendations generated from live positions and hedging logic.")

        if opportunities:
            for idx, idea in enumerate(opportunities):
                stars = self._compute_star_rating(idea)
                header = f"{idea['symbol']} — {idea['suggested_strategy']} {self._star_badge(stars)}"
                with st.expander(header, expanded=False):
                    st.markdown(f"**Rating:** {self._star_badge(stars)} ({stars}/5)")
                    st.write(f"**Signal:** {idea['signal']}")
                    st.write(idea["rationale"])
                    st.write(idea["trade_idea"])
                    metrics = idea.get("metrics", {})
                        if metrics:
                            st.table(self._metrics_table(metrics))
                    st.caption("Example order structure (edit before use):")
                    st.code(json.dumps(idea["order_example"], indent=2), language="json")
                    self._render_favorite_button(idea, f"live_{idx}")
                    self._render_preview_controls(idea, "live", idx)
        else:
            st.info("No trade ideas generated for the current portfolio.")

        st.markdown("---")
        st.subheader("Universe-Driven Opportunities")
        active_universe = strategy_context.get("universe_symbols", [])
        universe_caption = "Model Ensemble scans watchlist symbols for proactive trades."
        if active_universe:
            universe_caption += f" Tracking: {', '.join(active_universe)}."
        st.caption(universe_caption)
        self._render_market_state_tag(strategy_context.get("market_state"))
        col_edge, col_conf = st.columns(2)
        min_edge = col_edge.slider(
            "Minimum Edge (%)",
            min_value=0.0,
            max_value=10.0,
            value=float(self.universe_min_edge),
            step=0.1,
        )
        min_conf = col_conf.slider(
            "Minimum Confidence (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(self.universe_min_conf),
            step=1.0,
        )

        if (
            min_edge != st.session_state.get("universe_min_edge")
            or min_conf != st.session_state.get("universe_min_conf")
        ):
            st.session_state.universe_min_edge = min_edge
            st.session_state.universe_min_conf = min_conf
            self.snapshot = None
            st.rerun()

        if universe_ideas:
            for idx, idea in enumerate(universe_ideas):
                stars = self._compute_star_rating(idea)
                header = f"{idea['symbol']} — {idea['suggested_strategy']} {self._star_badge(stars)}"
                with st.expander(header, expanded=False):
                    st.markdown(f"**Rating:** {self._star_badge(stars)} ({stars}/5)")
                    st.write(f"**Signal:** {idea['signal']}")
                    st.write(idea["rationale"])
                    st.write(idea["trade_idea"])
                    metrics = idea.get("metrics", {})
                    if metrics:
                        st.table(self._metrics_table(metrics))
                    st.caption("Sample order (validate pricing before trading):")
                    st.code(json.dumps(idea["order_example"], indent=2), language="json")
                    self._render_favorite_button(idea, f"universe_{idx}")
                    self._render_preview_controls(idea, "universe", idx)
        else:
            st.info("No proactive opportunities met the edge thresholds this cycle.")

        self._render_favorites_section()

    def render_ai_assistant(self) -> None:
        if not self.api.can_use_tastytrade():
            st.info(
                "Authenticate with Tastytrade so Jeffrey can read live portfolio and risk context. "
                "He can still respond using cached information."
            )
        render_chat_interface()

    def render_risk(self) -> None:
        if not self.ensure_snapshot():
            return
        risk = self.snapshot["risk"]

        st.subheader("Portfolio Greeks & Exposure")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Delta", f"{risk['total_delta']:+.2f}")
        col2.metric("Total Gamma", f"{risk['total_gamma']:+.4f}")
        col3.metric("Total Theta", f"{risk['total_theta']:+.2f}")
        col4.metric("Total Vega", f"{risk['total_vega']:+.2f}")

        st.metric("Gross Exposure", f"${risk['gross_exposure']:,.2f}")
        st.metric("Risk Score (Heuristic)", f"{risk['risk_score']:.2f}/10")

        st.caption(
            "Risk score is a simple heuristic using delta magnitude and gross unrealized P&L exposure."
        )

    def render_trade_terminal(self) -> None:
        if not self.api.can_use_tastytrade():
            st.info("Authenticate on the Tastytrade tab to place trades.")
            return

        st.subheader("Submit Orders to Tastytrade")
        st.caption("Always validate option symbols and limit prices before submitting.")

        with st.form("trade_terminal_form"):
            instrument_type = st.selectbox("Instrument Type", ["Equity", "Equity Option"])
            symbol = st.text_input(
                "Symbol / Option Identifier",
                help="For options use the full tastytrade option symbol (e.g., AAPL  240621C195).",
            )
            action = st.selectbox(
                "Action",
                ["Buy to Open", "Sell to Open", "Buy to Close", "Sell to Close", "Buy", "Sell"],
            )
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            order_type = st.selectbox("Order Type", ["Market", "Limit"])
            limit_price = None
            if order_type == "Limit":
                limit_price = st.number_input("Limit Price", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            time_in_force = st.selectbox("Time in Force", ["Day", "GTC", "GTD"])
            source = st.text_input("Order Source", value="streamlit-app")

            preview = st.form_submit_button("Preview Order (Dry Run)")

            if preview:
                try:
                    if not symbol.strip():
                        raise ValueError("Symbol is required.")
                    payload = {
                        "time-in-force": time_in_force,
                        "order-type": order_type,
                        "legs": [
                            {
                                "instrument-type": instrument_type,
                                "symbol": symbol.strip(),
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

                    preview_data = self.api.dry_run_order(payload)
                    st.success("Dry run successful. Review below, then place the order.")
                    st.session_state.tt_order_payload = payload
                    st.session_state.tt_order_preview = preview_data
                except Exception as exc:
                    st.session_state.tt_order_payload = None
                    st.session_state.tt_order_preview = None
                    st.error(f"Dry run failed: {exc}")

        if st.session_state.tt_order_preview:
            st.subheader("Dry Run Result")
            st.json(st.session_state.tt_order_preview)

        if st.session_state.tt_order_payload:
            if st.button("🚀 Place Order", type="primary"):
                try:
                    result = self.api.place_order(st.session_state.tt_order_payload)
                    st.success("Order placed successfully!")
                    st.json(result)
                    st.session_state.tt_order_payload = None
                    st.session_state.tt_order_preview = None
                except Exception as exc:
                    st.error(f"Order placement failed: {exc}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        if isinstance(value, (int, float)):
            if abs(value) >= 100:
                return f"{value:,.0f}"
            if abs(value) >= 1:
                return f"{value:,.2f}"
            return f"{value:.4f}"
        if isinstance(value, dict):
            return json.dumps(value, default=str)
        if value is None:
            return "—"
        return str(value)

    @classmethod
    def _metrics_table(cls, metrics: Dict[str, Any]) -> pd.DataFrame:
        rows = [
            {
                "Metric": key.replace("_", " ").title(),
                "Value": cls._format_metric_value(val),
            }
            for key, val in metrics.items()
        ]
        if not rows:
            return pd.DataFrame(columns=["Metric", "Value"])
        return pd.DataFrame(rows, dtype="object")

    @staticmethod
    def _positions_dataframe(positions: List[Dict]) -> pd.DataFrame:
        rows = []
        for pos in positions:
            symbol = pos.get("symbol", "N/A").strip()
            instrument_type = pos.get("instrument-type", "Instrument")
            quantity = pos.get("quantity", 0)
            avg_price = TradingBotAPI._safe_float(pos.get("average-open-price"))
            mark = TradingBotAPI._safe_float(
                pos.get("mark")
                or pos.get("mark-price")
                or pos.get("close-price")
                or pos.get("last-price")
                or avg_price
            )
            pnl = TradingBotAPI._safe_float(pos.get("unrealized-gain-loss"))
            rows.append(
                {
                    "Symbol": symbol,
                    "Instrument": instrument_type,
                    "Quantity": quantity,
                    "Entry Price": avg_price,
                    "Mark": mark,
                    "Unrealized P&L": pnl,
                    "Delta": TradingBotAPI._safe_float(pos.get("delta")),
                    "Gamma": TradingBotAPI._safe_float(pos.get("gamma")),
                    "Theta": TradingBotAPI._safe_float(pos.get("theta")),
                    "Vega": TradingBotAPI._safe_float(pos.get("vega")),
                    "Expiration": pos.get("expiration-date", "N/A"),
                    "Abs Exposure": abs(mark * quantity),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df["Entry Price"] = df["Entry Price"].map(lambda x: round(x, 2))
            df["Mark"] = df["Mark"].map(lambda x: round(x, 2))
            df["Unrealized P&L"] = df["Unrealized P&L"].map(lambda x: round(x, 2))
            df["Delta"] = df["Delta"].map(lambda x: round(x, 4))
            df["Gamma"] = df["Gamma"].map(lambda x: round(x, 4))
            df["Theta"] = df["Theta"].map(lambda x: round(x, 4))
            df["Vega"] = df["Vega"].map(lambda x: round(x, 4))
        return df

    # ------------------------------------------------------------------
    # Existing Tastytrade tab (authentication, account mgmt)
    # ------------------------------------------------------------------

    def render_tastytrade_page(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))

        from utils.tastytrade_auth import get_auth_manager

        auth_manager = get_auth_manager()

        st.markdown("# 🔐 Tastytrade Authentication")
        st.markdown("---")

        with st.spinner("Checking authentication status..."):
            status = auth_manager.check_auth_status()

        if status.get("token_valid"):
            st.success("Connected to Tastytrade.")
        else:
            st.warning("Not connected to Tastytrade.")

        try:
            with st.spinner("Loading account details..."):
                accounts = auth_manager.get_accounts()
        except Exception as exc:
            accounts = []
            st.error(f"Failed to load accounts: {exc}")

        if accounts:
            for account in accounts:
                account_num = account.get("account-number")
                with st.expander(f"Account {account_num}", expanded=True):
                    st.json(account)
        else:
            st.info("No accounts returned.")

        st.markdown("### Trade Opportunities")
        opportunities = []
        if self.api.can_use_tastytrade():
            try:
                opportunities = self.api.get_trade_opportunities()
            except Exception:
                opportunities = []
        if opportunities:
            for idea in opportunities:
                with st.expander(f"{idea['symbol']} — {idea['suggested_strategy']}", expanded=False):
                    st.write(f"**Signal:** {idea['signal']}")
                    st.write(idea["rationale"])
                    st.write(idea["trade_idea"])
                    st.caption("Example order structure (edit before use):")
                    st.code(json.dumps(idea["order_example"], indent=2), language="json")
        else:
            st.info("No trade opportunities generated yet.")

        st.markdown("### 🛒 Trade Execution")
        st.caption("Use this form to run a dry-run preview and then submit orders.")

        if "tt_last_order_payload_page" not in st.session_state:
            st.session_state.tt_last_order_payload_page = None
        if "tt_last_dry_run_page" not in st.session_state:
            st.session_state.tt_last_dry_run_page = None

        with st.form("tastytrade_order_form_page"):
            instrument_type = st.selectbox("Instrument Type", ["Equity", "Equity Option"], key="tt_form_inst")
            symbol_input = st.text_input(
                "Symbol / Option Identifier",
                help="For options, use the full tastytrade option symbol (e.g., AAPL  240621C195).",
                key="tt_form_symbol",
            )
            action = st.selectbox(
                "Action",
                ["Buy to Open", "Sell to Open", "Buy to Close", "Sell to Close", "Buy", "Sell"],
                key="tt_form_action",
            )
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="tt_form_qty")
            order_type = st.selectbox("Order Type", ["Market", "Limit"], key="tt_form_type")
            limit_price = None
            if order_type == "Limit":
                limit_price = st.number_input(
                    "Limit Price", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="tt_form_price"
                )
            time_in_force = st.selectbox("Time in Force", ["Day", "GTC", "GTD"], key="tt_form_tif")
            source = st.text_input("Order Source", value="streamlit-app", key="tt_form_source")

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
            if st.button("🚀 Place Order Now", key="place_order_tt_tab"):
                try:
                    result = auth_manager.place_order(st.session_state.tt_last_order_payload_page)
                    st.success("Order placed successfully!")
                    st.json(result)
                    st.session_state.tt_last_order_payload_page = None
                    st.session_state.tt_last_dry_run_page = None
                except Exception as exc:
                    st.error(f"Order placement failed: {exc}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.render_header()
        page = self.render_navigation()

        if page == "Overview":
            self.render_overview()
        elif page == "Portfolio":
            self.render_portfolio()
        elif page == "Strategies":
            self.render_strategies()
        elif page == "Risk":
            self.render_risk()
        elif page == "Trade Terminal":
            self.render_trade_terminal()
        elif page == "Jeffrey":
            self.render_ai_assistant()
        elif page == "Tastytrade":
            self.render_tastytrade_page()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        app = LiveTradingApp()
        app.run()
    except Exception as exc:
        st.exception(exc)
