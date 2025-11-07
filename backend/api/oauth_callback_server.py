"""
Tastytrade OAuth2 Callback Server

This module implements a simple HTTP server to handle OAuth2 callbacks
from Tastytrade authentication flow.
"""

import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import webbrowser
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.api.tastytrade_auth import TastytradeAuth
from backend.config.tastytrade_config import TastytradeConfig


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth2 callbacks."""

    auth_code = None
    state = None
    error = None

    def do_GET(self):
        """Handle GET requests (OAuth callback)."""
        # Parse the URL
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        # Check if this is the callback
        if parsed_path.path == '/callback':
            # Extract authorization code and state
            OAuthCallbackHandler.auth_code = query_params.get('code', [None])[0]
            OAuthCallbackHandler.state = query_params.get('state', [None])[0]
            OAuthCallbackHandler.error = query_params.get('error', [None])[0]

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            if OAuthCallbackHandler.error:
                html_content = f"""
                <html>
                <head><title>OAuth Error</title></head>
                <body style="font-family: Arial, sans-serif; padding: 50px; text-align: center;">
                    <h1 style="color: #d32f2f;">Authentication Failed</h1>
                    <p>Error: {OAuthCallbackHandler.error}</p>
                    <p>Please close this window and try again.</p>
                </body>
                </html>
                """
            elif OAuthCallbackHandler.auth_code:
                html_content = """
                <html>
                <head><title>OAuth Success</title></head>
                <body style="font-family: Arial, sans-serif; padding: 50px; text-align: center;">
                    <h1 style="color: #4caf50;">Authentication Successful!</h1>
                    <p>You have successfully authenticated with Tastytrade.</p>
                    <p>You can now close this window and return to your application.</p>
                    <script>
                        setTimeout(function() {
                            window.close();
                        }, 3000);
                    </script>
                </body>
                </html>
                """
            else:
                html_content = """
                <html>
                <head><title>OAuth Error</title></head>
                <body style="font-family: Arial, sans-serif; padding: 50px; text-align: center;">
                    <h1 style="color: #d32f2f;">Authentication Failed</h1>
                    <p>No authorization code received.</p>
                    <p>Please close this window and try again.</p>
                </body>
                </html>
                """

            self.wfile.write(html_content.encode())
        else:
            # Not the callback path
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Override to reduce logging noise."""
        pass


class OAuthCallbackServer:
    """Simple OAuth callback server manager."""

    def __init__(self, port: int = 8080):
        """
        Initialize the callback server.

        Args:
            port: Port to run the server on (default: 8080)
        """
        self.port = port
        self.server = None
        self.server_thread = None

    def start(self) -> None:
        """Start the callback server in a background thread."""
        self.server = HTTPServer(('localhost', self.port), OAuthCallbackHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"OAuth callback server started on http://localhost:{self.port}/callback")

    def stop(self) -> None:
        """Stop the callback server."""
        if self.server:
            self.server.shutdown()
            print("OAuth callback server stopped")

    def wait_for_callback(self, timeout: int = 300) -> tuple[str, str, str]:
        """
        Wait for OAuth callback.

        Args:
            timeout: Timeout in seconds (default: 300 = 5 minutes)

        Returns:
            tuple: (auth_code, state, error)
        """
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            if OAuthCallbackHandler.auth_code or OAuthCallbackHandler.error:
                return (
                    OAuthCallbackHandler.auth_code,
                    OAuthCallbackHandler.state,
                    OAuthCallbackHandler.error
                )
            time.sleep(0.5)

        raise TimeoutError(f"No callback received within {timeout} seconds")


def authenticate_with_tastytrade(auto_open_browser: bool = True) -> TastytradeAuth:
    """
    Complete OAuth2 authentication flow with Tastytrade.

    Args:
        auto_open_browser: Whether to automatically open browser for auth

    Returns:
        TastytradeAuth: Authenticated auth handler

    Raises:
        Exception: If authentication fails
    """
    print("=" * 60)
    print("Tastytrade OAuth2 Authentication")
    print("=" * 60)

    # Initialize auth handler
    auth = TastytradeAuth()

    # Try to load existing tokens
    if auth.load_tokens():
        print("\nExisting tokens found!")

        # Check if token is still valid
        if not auth.is_token_expired():
            print("Token is still valid.")

            # Test the token by fetching accounts
            try:
                accounts = auth.get_customer_accounts()
                print(f"\nAuthentication successful!")
                print(f"Found {len(accounts)} account(s)")
                for acc in accounts:
                    print(f"  - Account: {acc.get('account-number')} ({acc.get('authority-level')})")
                return auth
            except Exception as e:
                print(f"Token validation failed: {e}")
                print("Will re-authenticate...")
        else:
            print("Token expired, attempting refresh...")
            try:
                auth.refresh_session_token()
                accounts = auth.get_customer_accounts()
                print(f"\nToken refreshed successfully!")
                print(f"Found {len(accounts)} account(s)")
                return auth
            except Exception as e:
                print(f"Token refresh failed: {e}")
                print("Will re-authenticate...")

    # Need new authentication
    print("\nStarting new authentication flow...")

    # Reset callback handler state
    OAuthCallbackHandler.auth_code = None
    OAuthCallbackHandler.state = None
    OAuthCallbackHandler.error = None

    # Start callback server
    callback_server = OAuthCallbackServer(port=8080)
    callback_server.start()

    # Generate auth URL
    auth_url, state = auth.generate_auth_url()

    print("\n" + "=" * 60)
    print("STEP 1: Open the following URL in your browser")
    print("=" * 60)
    print(auth_url)
    print()

    if auto_open_browser:
        print("Opening browser automatically...")
        webbrowser.open(auth_url)
    else:
        print("Please copy and paste the URL above into your browser.")

    print("\n" + "=" * 60)
    print("STEP 2: Log in and authorize the application")
    print("=" * 60)
    print("Waiting for callback...")

    try:
        # Wait for callback
        auth_code, received_state, error = callback_server.wait_for_callback(timeout=300)

        # Stop server
        callback_server.stop()

        if error:
            raise Exception(f"Authentication error: {error}")

        if not auth_code:
            raise Exception("No authorization code received")

        # Validate state (CSRF protection)
        if received_state != state:
            raise Exception("State mismatch - possible CSRF attack")

        print("\n" + "=" * 60)
        print("STEP 3: Exchanging authorization code for token")
        print("=" * 60)

        # Exchange code for token
        token_data = auth.exchange_code_for_token(auth_code)
        print(f"Token exchange successful!")

        # Fetch accounts
        accounts = auth.get_customer_accounts()

        print("\n" + "=" * 60)
        print("Authentication Complete!")
        print("=" * 60)
        print(f"Found {len(accounts)} account(s):")
        for acc in accounts:
            account_num = acc.get('account-number')
            authority = acc.get('authority-level')
            account_type = acc.get('account-type')
            nickname = acc.get('nickname', 'N/A')
            print(f"  - {account_num} ({authority}) - {account_type} - {nickname}")

        return auth

    except TimeoutError as e:
        callback_server.stop()
        raise Exception(f"Authentication timeout: {e}")
    except Exception as e:
        callback_server.stop()
        raise


def main():
    """Main entry point for OAuth authentication."""
    try:
        auth = authenticate_with_tastytrade(auto_open_browser=True)

        print("\n" + "=" * 60)
        print("Testing API Access")
        print("=" * 60)

        # Test getting quote token
        quote_token = auth.get_api_quote_token()
        print(f"API Quote Token obtained successfully!")
        print(f"Token (first 20 chars): {quote_token[:20]}...")

        print("\n" + "=" * 60)
        print("Authentication Status")
        print("=" * 60)
        status = auth.get_auth_status()
        for key, value in status.items():
            print(f"{key}: {value}")

        print("\n" + "=" * 60)
        print("Success! You are now authenticated with Tastytrade.")
        print("=" * 60)
        print("\nYou can now use the Tastytrade API in your application.")
        print("Tokens have been saved and will be automatically refreshed.")

    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
