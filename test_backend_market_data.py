"""
Test the backend market data service endpoints
"""
import os
import sys
import asyncio
from pprint import pprint

# Set environment variables for testing
os.environ["TASTYTRADE_USERNAME"] = "saleem86@gmail.com"
os.environ["TASTYTRADE_PASSWORD"] = "olaabdel88"
os.environ["TASTYTRADE_SANDBOX"] = "false"  # Production
os.environ["ENSEMBLE_RISK_LEVEL"] = "moderate"

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("="*70)
print("Testing Backend Market Data Service")
print("="*70)

# Test 1: MarketDataService initialization
print("\n[TEST 1] MarketDataService Initialization")
print("-" * 70)

try:
    from service.market_data import MarketDataService, TastytradeSessionManager

    print("✓ Imports successful")

    # Test session manager
    print("\nTesting TastytradeSessionManager...")
    async def test_session():
        manager = TastytradeSessionManager()
        print(f"  Username: {manager.username}")
        print(f"  Is Test: {manager.is_test}")

        session = await manager.get_session()
        print(f"  ✓ Session created: {session.session_token[:20]}...")
        return session

    session = asyncio.run(test_session())
    print("✓ Session manager working")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: DxLinkCollector
print("\n[TEST 2] DxLinkCollector - Fetch Greeks and Quotes")
print("-" * 70)

try:
    from service.market_data import DxLinkCollector

    async def test_dxlink():
        manager = TastytradeSessionManager()
        session = await manager.get_session()

        # Get a test symbol
        from tastytrade.instruments import get_option_chain
        print("Fetching COST option chain...")
        chain = get_option_chain(session, "COST")

        first_exp = list(chain.keys())[0]
        options = chain[first_exp]

        # Get a few streamer symbols
        test_symbols = [opt.streamer_symbol for opt in options[:3]]
        print(f"\nTest symbols: {test_symbols}")

        collector = DxLinkCollector(timeout=5.0)
        print("\nCollecting market data...")

        data = await collector.collect(session, test_symbols)

        print(f"\n✓ Collected data for {len(data)} symbols")

        # Show detailed data for first symbol
        if data:
            first_sym = list(data.keys())[0]
            print(f"\nDetailed data for {first_sym}:")
            print("  Quote:", data[first_sym].get("quote"))
            print("  Greeks:", data[first_sym].get("greeks"))
            print("  Theo:", data[first_sym].get("theo"))

        return data

    market_data = asyncio.run(test_dxlink())
    print("\n✓ DxLinkCollector working")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Full MarketDataService.build_snapshot
print("\n[TEST 3] MarketDataService.build_snapshot()")
print("-" * 70)

try:
    async def test_full_service():
        service = MarketDataService()

        print("Building snapshot for COST...")
        snapshot = await service.build_snapshot(
            symbols=["COST"],
            max_options=10,  # Limit for testing
            strike_span=0.10
        )

        print(f"\n✓ Snapshot built successfully")
        print(f"  Symbols: {snapshot['symbols']}")
        print(f"  Underlying prices: {snapshot.get('underlying_prices', {})}")

        if snapshot.get('option_chains'):
            for symbol, chain in snapshot['option_chains'].items():
                print(f"\n  {symbol} option chain:")
                print(f"    Total options: {len(chain)}")
                if chain:
                    print(f"    First option: {chain[0].get('symbol', 'N/A')}")
                    print(f"      Strike: ${chain[0].get('strike', 'N/A')}")
                    print(f"      Type: {chain[0].get('option_type', 'N/A')}")

        if snapshot.get('market_data'):
            print(f"\n  Market data streams:")
            for symbol, streams in snapshot['market_data'].items():
                if streams.get('close'):
                    print(f"    {symbol}: {len(streams['close'])} price points")

        return snapshot

    snapshot = asyncio.run(test_full_service())
    print("\n✓ Full service working")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: FastAPI endpoint (if running)
print("\n[TEST 4] FastAPI /market-data/options endpoint")
print("-" * 70)

try:
    import requests

    # Try to hit the endpoint
    backend_url = os.getenv("TRADING_BOT_API_URL", "http://localhost:8000")

    print(f"Testing endpoint: {backend_url}/market-data/options")

    payload = {
        "symbols": ["COST"],
        "max_options": 10,
        "strike_span": 0.10
    }

    response = requests.post(
        f"{backend_url}/market-data/options",
        json=payload,
        timeout=30
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Endpoint responded successfully")
        print(f"  Symbols: {data.get('symbols', [])}")
        print(f"  Underlying prices: {data.get('underlying_prices', {})}")

        if data.get('option_chains'):
            for symbol, chain in data['option_chains'].items():
                print(f"  {symbol}: {len(chain)} options")
    else:
        print(f"✗ Endpoint returned status {response.status_code}")
        print(f"  Response: {response.text}")

except requests.exceptions.ConnectionError:
    print("⚠ Backend service not running - skipping endpoint test")
    print("  To test the endpoint, start the backend with:")
    print("  cd backend/service && uvicorn app:app --reload")
except Exception as e:
    print(f"⚠ Endpoint test failed: {e}")

print("\n" + "="*70)
print("Testing Complete!")
print("="*70)
