"""
API Integration Tests
=====================

Test suite for verifying frontend-backend integration
"""

import pytest
import requests
import asyncio
import websockets
import json
import time
from typing import Dict, List

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/updates"

class TestAPIEndpoints:
    """Test all API endpoints"""

    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_portfolio_status(self):
        """Test portfolio status endpoint"""
        response = requests.get(f"{API_BASE_URL}/api/portfolio/status")
        assert response.status_code == 200
        data = response.json()

        # Check required fields
        required_fields = ["portfolio_value", "daily_pnl", "positions_count", "available_cash"]
        for field in required_fields:
            assert field in data

        # Validate data types
        assert isinstance(data["portfolio_value"], (int, float))
        assert isinstance(data["daily_pnl"], (int, float))
        assert isinstance(data["positions_count"], int)

    def test_get_positions(self):
        """Test positions endpoint"""
        response = requests.get(f"{API_BASE_URL}/api/positions")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        if data:  # If positions exist
            position = data[0]
            required_fields = ["symbol", "type", "quantity", "pnl"]
            for field in required_fields:
                assert field in position

    def test_get_strategies(self):
        """Test strategies endpoint"""
        response = requests.get(f"{API_BASE_URL}/api/strategies")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        if data:
            strategy = data[0]
            required_fields = ["name", "status", "positions", "daily_pnl"]
            for field in required_fields:
                assert field in strategy

    def test_model_performance(self):
        """Test model performance endpoint"""
        response = requests.get(f"{API_BASE_URL}/api/models/performance")
        assert response.status_code == 200
        data = response.json()

        assert "weights" in data
        assert "accuracy" in data
        assert isinstance(data["weights"], dict)
        assert isinstance(data["accuracy"], dict)

    def test_risk_metrics(self):
        """Test risk metrics endpoint"""
        response = requests.get(f"{API_BASE_URL}/api/risk/metrics")
        assert response.status_code == 200
        data = response.json()

        risk_fields = ["var_95", "max_drawdown", "portfolio_delta"]
        for field in risk_fields:
            assert field in data

    def test_start_stop_trading(self):
        """Test trading control endpoints"""
        # Test start trading
        response = requests.post(f"{API_BASE_URL}/api/trading/start")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["started", "already_running"]

        time.sleep(1)

        # Test stop trading
        response = requests.post(f"{API_BASE_URL}/api/trading/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["stopped", "not_running"]

    def test_pause_resume_strategy(self):
        """Test strategy control endpoints"""
        strategy_name = "Model Ensemble"

        # Pause strategy
        response = requests.post(f"{API_BASE_URL}/api/strategies/{strategy_name}/pause")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"

        # Resume strategy
        response = requests.post(f"{API_BASE_URL}/api/strategies/{strategy_name}/resume")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "resumed"


class TestWebSocketConnection:
    """Test WebSocket real-time updates"""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection and message reception"""
        try:
            async with websockets.connect(WS_URL) as websocket:
                # Wait for first message
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)

                # Validate message structure
                assert "type" in data
                assert "data" in data

                if data["type"] == "portfolio_update":
                    assert "portfolio_value" in data["data"]
                    assert "daily_pnl" in data["data"]
                    assert "timestamp" in data["data"]

                print(f"Received WebSocket message: {data}")

        except asyncio.TimeoutError:
            pytest.fail("WebSocket connection timed out")
        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {e}")

    @pytest.mark.asyncio
    async def test_multiple_websocket_messages(self):
        """Test receiving multiple WebSocket messages"""
        messages_received = []

        try:
            async with websockets.connect(WS_URL) as websocket:
                # Receive 5 messages
                for _ in range(5):
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(message)
                    messages_received.append(data)

                # Verify we received multiple messages
                assert len(messages_received) >= 5
                print(f"Received {len(messages_received)} WebSocket messages")

        except asyncio.TimeoutError:
            if not messages_received:
                pytest.fail("No WebSocket messages received")
            else:
                print(f"Received {len(messages_received)} messages before timeout")


class TestFrontendBackendIntegration:
    """Test full integration between frontend and backend"""

    def test_api_client_import(self):
        """Test that API client can be imported"""
        try:
            from frontend.utils.api_client import TradingBotAPI
            client = TradingBotAPI()
            assert client is not None
        except ImportError as e:
            pytest.fail(f"Failed to import API client: {e}")

    def test_websocket_client_import(self):
        """Test that WebSocket client can be imported"""
        try:
            from frontend.utils.websocket_client import RealTimeUpdates
            client = RealTimeUpdates()
            assert client is not None
        except ImportError as e:
            pytest.fail(f"Failed to import WebSocket client: {e}")

    def test_full_data_flow(self):
        """Test complete data flow from backend to frontend"""
        try:
            from frontend.utils.api_client import TradingBotAPI

            client = TradingBotAPI()

            # Test fetching data
            portfolio = client.get_portfolio_status()
            assert portfolio is not None
            assert "portfolio_value" in portfolio

            positions = client.get_positions()
            assert isinstance(positions, list)

            strategies = client.get_strategies()
            assert isinstance(strategies, list)

            print("Full data flow test passed!")

        except Exception as e:
            pytest.fail(f"Full data flow test failed: {e}")


def run_integration_tests():
    """Run all integration tests"""
    print("=" * 50)
    print("Running API Integration Tests")
    print("=" * 50)

    # Check if backend is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("⚠️  Backend API is not healthy")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend API is not running!")
        print("Please start the backend with: uvicorn backend.api.endpoints:app --reload")
        return False

    print("✅ Backend API is running")

    # Run tests
    test_results = []

    # Test API endpoints
    api_tests = TestAPIEndpoints()
    test_methods = [
        ("Health Check", api_tests.test_health_check),
        ("Portfolio Status", api_tests.test_portfolio_status),
        ("Get Positions", api_tests.test_get_positions),
        ("Get Strategies", api_tests.test_get_strategies),
        ("Model Performance", api_tests.test_model_performance),
        ("Risk Metrics", api_tests.test_risk_metrics),
        ("Start/Stop Trading", api_tests.test_start_stop_trading),
        ("Pause/Resume Strategy", api_tests.test_pause_resume_strategy),
    ]

    for test_name, test_func in test_methods:
        try:
            test_func()
            print(f"✅ {test_name} - PASSED")
            test_results.append((test_name, True))
        except Exception as e:
            print(f"❌ {test_name} - FAILED: {e}")
            test_results.append((test_name, False))

    # Test WebSocket
    ws_tests = TestWebSocketConnection()
    try:
        asyncio.run(ws_tests.test_websocket_connection())
        print("✅ WebSocket Connection - PASSED")
        test_results.append(("WebSocket Connection", True))
    except Exception as e:
        print(f"❌ WebSocket Connection - FAILED: {e}")
        test_results.append(("WebSocket Connection", False))

    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n🎉 All tests passed! Integration is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)