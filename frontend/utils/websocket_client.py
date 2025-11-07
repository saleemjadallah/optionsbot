import streamlit as st
import asyncio
import websockets
import json
import threading
import time
from typing import Callable, Optional
import os
from datetime import datetime

class RealTimeUpdates:
    def __init__(self, websocket_url: str = None):
        self.websocket_url = websocket_url or os.getenv("TRADING_BOT_WS_URL", "ws://localhost:8000/ws/updates")
        self.connected = False
        self.websocket = None
        self.loop = None
        self.thread = None
        self.running = False

    def connect(self, callback: Callable = None):
        """Connect to WebSocket and listen for updates"""
        if self.connected:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_websocket, args=(callback,))
        self.thread.daemon = True
        self.thread.start()

    def _run_websocket(self, callback: Callable = None):
        """Run WebSocket connection in separate thread"""
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.loop = asyncio.get_event_loop()
            self.loop.run_until_complete(self._connect_and_listen(callback))
        except Exception as e:
            print(f"WebSocket thread error: {e}")

    async def _connect_and_listen(self, callback: Callable = None):
        """Connect to WebSocket and listen for updates"""
        retry_count = 0
        max_retries = 3

        while self.running and retry_count < max_retries:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    self.websocket = websocket
                    self.connected = True
                    retry_count = 0  # Reset retry count on successful connection
                    print(f"Connected to WebSocket at {self.websocket_url}")

                    async for message in websocket:
                        if not self.running:
                            break

                        try:
                            data = json.loads(message)

                            # Update session state
                            self.update_portfolio(data)

                            # Call custom callback if provided
                            if callback:
                                callback(data)

                        except json.JSONDecodeError as e:
                            print(f"Failed to parse WebSocket message: {e}")

            except websockets.exceptions.ConnectionClosedError:
                self.connected = False
                print("WebSocket connection closed")
                if self.running:
                    await asyncio.sleep(5)  # Wait before reconnecting

            except Exception as e:
                self.connected = False
                retry_count += 1
                print(f"WebSocket connection failed (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries and self.running:
                    await asyncio.sleep(5)  # Wait before retrying

    def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False
        self.connected = False
        if self.websocket:
            asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)

    def update_portfolio(self, data: dict):
        """Update portfolio data in session state"""
        if 'portfolio_updates' not in st.session_state:
            st.session_state.portfolio_updates = []

        # Store update in history
        st.session_state.portfolio_updates.append({
            **data,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 100 updates
        if len(st.session_state.portfolio_updates) > 100:
            st.session_state.portfolio_updates = st.session_state.portfolio_updates[-100:]

        # Update current values based on message type
        if data.get('type') == 'portfolio_update':
            portfolio_data = data.get('data', {})
            st.session_state.portfolio_value = portfolio_data.get('portfolio_value',
                                                                  st.session_state.get('portfolio_value', 0))
            st.session_state.daily_pnl = portfolio_data.get('daily_pnl',
                                                           st.session_state.get('daily_pnl', 0))
            st.session_state.positions_count = portfolio_data.get('positions_count',
                                                                  st.session_state.get('positions_count', 0))
            st.session_state.last_update = datetime.now()

        elif data.get('type') == 'trade_executed':
            trade_data = data.get('data', {})
            if 'recent_trades' not in st.session_state:
                st.session_state.recent_trades = []
            st.session_state.recent_trades.append(trade_data)
            # Keep only last 50 trades
            if len(st.session_state.recent_trades) > 50:
                st.session_state.recent_trades = st.session_state.recent_trades[-50:]

        elif data.get('type') == 'alert':
            alert_data = data.get('data', {})
            if 'alerts' not in st.session_state:
                st.session_state.alerts = []
            st.session_state.alerts.append(alert_data)
            # Keep only last 20 alerts
            if len(st.session_state.alerts) > 20:
                st.session_state.alerts = st.session_state.alerts[-20:]

    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.connected

    def get_status(self) -> str:
        """Get connection status"""
        if self.connected:
            return "Connected"
        elif self.running:
            return "Connecting..."
        else:
            return "Disconnected"

# Singleton instance
_websocket_client = None

def get_websocket_client() -> RealTimeUpdates:
    """Get singleton WebSocket client instance"""
    global _websocket_client
    if _websocket_client is None:
        _websocket_client = RealTimeUpdates()
    return _websocket_client