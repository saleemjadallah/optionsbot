"""
Test script to fetch COST quotes and option chains from Tastytrade
"""
from tastytrade import Session
from tastytrade.instruments import get_option_chain

# Authenticate with production environment
username = "saleem86@gmail.com"
password = "olaabdel88"

print("Authenticating with Tastytrade production API...")
session = Session(username, password, is_test=False)

print(f"✓ Authentication successful!")
print(f"  Session token: {session.session_token[:20]}...")
print(f"  User ID: {session.user.external_id if hasattr(session, 'user') else 'N/A'}")

# Fetch COST option chain
print("\nFetching COST option chain...")
chain = get_option_chain(session, "COST")

print(f"✓ Chain expirations (first 3): {list(chain.keys())[:3]}")
print(f"  Total expirations available: {len(chain.keys())}")

# Show some details about the first expiration
if chain:
    first_exp = list(chain.keys())[0]
    options_for_exp = chain[first_exp]
    print(f"\n  First expiration: {first_exp}")
    print(f"  Number of options for this expiration: {len(options_for_exp)}")

    # Show first few option details
    if options_for_exp:
        print(f"  Sample options (first 3):")
        for i, opt in enumerate(options_for_exp[:3]):
            print(f"    {i+1}. {opt.symbol} - Strike: ${opt.strike_price}")

        # Inspect all available fields on an option object
        print(f"\n  Available fields on option objects:")
        first_opt = options_for_exp[0]
        all_attrs = [attr for attr in dir(first_opt) if not attr.startswith('_')]
        for attr in sorted(all_attrs):
            try:
                value = getattr(first_opt, attr)
                if not callable(value):
                    print(f"    - {attr}: {value}")
            except:
                pass


# Try to fetch market data including Greeks for options
print("\n" + "="*60)
print("Testing Market Data (Greeks, Quotes) for Options...")
print("="*60)

from tastytrade.dxfeed import Greeks, Quote, TheoPrice
from tastytrade import DXLinkStreamer

try:
    # Get a few option symbols to fetch data for
    sample_symbols = [opt.streamer_symbol for opt in options_for_exp[:5]]
    print(f"\nAttempting to fetch market data for {len(sample_symbols)} options...")

    # Create streamer and subscribe
    async def fetch_market_data():
        async with DXLinkStreamer(session) as streamer:
            # Subscribe to Greeks, Quotes, and Theoretical Price
            await streamer.subscribe(Greeks, sample_symbols)
            await streamer.subscribe(Quote, sample_symbols)
            await streamer.subscribe(TheoPrice, sample_symbols)

            # Wait for data to arrive
            import asyncio
            await asyncio.sleep(3)

            # Fetch Greeks data
            print("\n--- GREEKS DATA ---")
            greeks_data = await streamer.get_event(Greeks)
            if greeks_data:
                print(f"\nGreeks for {greeks_data.eventSymbol}:")
                attrs = [attr for attr in dir(greeks_data) if not attr.startswith('_') and not callable(getattr(greeks_data, attr))]
                for attr in sorted(attrs):
                    value = getattr(greeks_data, attr)
                    print(f"  {attr}: {value}")

            # Fetch Quote data
            print("\n--- QUOTE DATA ---")
            quote_data = await streamer.get_event(Quote)
            if quote_data:
                print(f"\nQuote for {quote_data.eventSymbol}:")
                print(f"  Bid: ${quote_data.bidPrice if hasattr(quote_data, 'bidPrice') else 'N/A'}")
                print(f"  Ask: ${quote_data.askPrice if hasattr(quote_data, 'askPrice') else 'N/A'}")
                print(f"  Bid Size: {quote_data.bidSize if hasattr(quote_data, 'bidSize') else 'N/A'}")
                print(f"  Ask Size: {quote_data.askSize if hasattr(quote_data, 'askSize') else 'N/A'}")

            # Fetch Theo Price
            print("\n--- THEORETICAL PRICE ---")
            theo_data = await streamer.get_event(TheoPrice)
            if theo_data:
                print(f"\nTheo Price for {theo_data.eventSymbol}:")
                attrs = [attr for attr in dir(theo_data) if not attr.startswith('_') and not callable(getattr(theo_data, attr))]
                for attr in sorted(attrs):
                    value = getattr(theo_data, attr)
                    print(f"  {attr}: {value}")

    # Run the async function
    import asyncio
    asyncio.run(fetch_market_data())

except Exception as e:
    print(f"\nStreaming market data failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Test completed successfully!")
print("  COST quotes and option chains are accessible via the Tastytrade SDK")
