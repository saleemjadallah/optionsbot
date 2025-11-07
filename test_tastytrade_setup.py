"""
Tastytrade Setup Verification Script

Run this script to verify your Tastytrade integration is properly configured.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))


def test_configuration():
    """Test configuration is valid."""
    print("=" * 60)
    print("Step 1: Testing Configuration")
    print("=" * 60)

    try:
        from backend.config.tastytrade_config import TastytradeConfig

        config = TastytradeConfig.get_config_summary()

        print("✅ Configuration loaded successfully!")
        print(f"   Environment: {config['environment']}")
        print(f"   Base URL: {config['base_url']}")
        print(f"   Client ID: {config['client_id']}")
        print(f"   Client Secret: {'Set' if config['client_secret_set'] else 'NOT SET'}")
        print(f"   Redirect URI: {config['redirect_uri']}")
        print(f"   Scopes: {config['scopes']}")

        # Validate
        is_valid, missing = TastytradeConfig.validate_config()
        if not is_valid:
            print(f"❌ Configuration incomplete. Missing: {', '.join(missing)}")
            return False

        print("✅ Configuration is valid!")
        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_auth_module():
    """Test authentication module can be imported."""
    print("\n" + "=" * 60)
    print("Step 2: Testing Authentication Module")
    print("=" * 60)

    try:
        from backend.api.tastytrade_auth import TastytradeAuth

        auth = TastytradeAuth()
        print("✅ Authentication module loaded successfully!")

        # Try to load existing tokens
        if auth.load_tokens():
            print("✅ Found existing tokens!")

            status = auth.get_auth_status()
            print(f"   Authenticated: {status['authenticated']}")
            print(f"   Token Valid: {status['token_valid']}")
            print(f"   Account Number: {status['account_number']}")

            if status['token_valid']:
                print("✅ Authentication is active and valid!")
                return True
            else:
                print("⚠️  Tokens found but expired. Need to authenticate.")
                return None  # Partial success
        else:
            print("ℹ️  No existing tokens found. Need to authenticate.")
            return None  # Partial success

    except Exception as e:
        print(f"❌ Authentication module test failed: {e}")
        return False


def test_client():
    """Test API client."""
    print("\n" + "=" * 60)
    print("Step 3: Testing API Client")
    print("=" * 60)

    try:
        from backend.api.tastytrade_client import TastytradeClient

        client = TastytradeClient()
        print("✅ API client initialized successfully!")

        # Try to make an API call if authenticated
        if client.auth.session_token:
            print("   Attempting to fetch accounts...")
            try:
                accounts = client.get_accounts()
                print(f"✅ Successfully fetched {len(accounts)} account(s)!")

                for acc in accounts:
                    print(f"   - {acc['account-number']} ({acc['authority-level']})")

                return True

            except Exception as e:
                print(f"⚠️  Could not fetch accounts: {e}")
                print("   You may need to re-authenticate.")
                return None
        else:
            print("ℹ️  Not authenticated yet. Client ready for use.")
            return None

    except Exception as e:
        print(f"❌ Client test failed: {e}")
        return False


def test_callback_server():
    """Test callback server can be imported."""
    print("\n" + "=" * 60)
    print("Step 4: Testing OAuth Callback Server")
    print("=" * 60)

    try:
        from backend.api.oauth_callback_server import OAuthCallbackServer

        print("✅ OAuth callback server module loaded successfully!")
        print("   Ready to handle OAuth authentication flow.")
        return True

    except Exception as e:
        print(f"❌ Callback server test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "TASTYTRADE SETUP VERIFICATION" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = []

    # Run tests
    results.append(("Configuration", test_configuration()))
    results.append(("Authentication Module", test_auth_module()))
    results.append(("API Client", test_client()))
    results.append(("OAuth Server", test_callback_server()))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    needs_auth = False

    for name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is None:
            status = "⚠️  PARTIAL"
            needs_auth = True
        else:
            status = "❌ FAIL"
            all_passed = False

        print(f"{status} - {name}")

    print("\n" + "=" * 60)

    if all_passed and not needs_auth:
        print("🎉 All tests passed! Your setup is complete and authenticated!")
        print()
        print("You can now use the Tastytrade API in your application.")
        print()
        print("Example:")
        print("  from backend.api.tastytrade_client import TastytradeClient")
        print("  client = TastytradeClient()")
        print("  accounts = client.get_accounts()")
        print()

    elif needs_auth:
        print("⚠️  Setup is complete but you need to authenticate!")
        print()
        print("Run the following command to authenticate:")
        print()
        print("  python backend/api/oauth_callback_server.py")
        print()
        print("This will:")
        print("  1. Open your browser to Tastytrade login")
        print("  2. Handle OAuth callback automatically")
        print("  3. Save tokens for future use")
        print()
        print("Login credentials:")
        print("  Username: tastyslim")
        print("  Password: saleemjadallah1986")
        print()

    else:
        print("❌ Some tests failed. Please check the errors above.")
        print()
        print("Common issues:")
        print("  - Missing dependencies: pip install requests python-dotenv")
        print("  - Check .env file exists and has correct values")
        print("  - Ensure all files were created correctly")
        print()

    print("=" * 60)


if __name__ == '__main__':
    main()
