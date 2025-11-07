"""
Tastytrade OAuth2 Authentication Handler

This module handles OAuth2 authentication flow for Tastytrade API,
including token management, refresh, and session handling.
"""

import requests
import time
import json
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from backend.config.tastytrade_config import TastytradeConfig, APIHeaders


class TastytradeAuth:
    """Handles OAuth2 authentication and session management for Tastytrade API."""

    def __init__(self):
        """Initialize the authentication handler."""
        self.config = TastytradeConfig
        self.session_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self.api_quote_token: Optional[str] = None
        self.quote_token_expiry: Optional[datetime] = None
        self.account_number: Optional[str] = None

        # Token storage file
        self.token_file = Path(__file__).parent.parent / 'credentials' / 'tokens.json'
        self.token_file.parent.mkdir(parents=True, exist_ok=True)

    def generate_auth_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """
        Generate the OAuth2 authorization URL.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            tuple: (authorization_url, state)
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            'response_type': 'code',
            'client_id': self.config.CLIENT_ID,
            'redirect_uri': self.config.REDIRECT_URI,
            'scope': self.config.SCOPES,
            'state': state
        }

        auth_url = self.config.get_oauth_authorize_url()
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{auth_url}?{param_string}"

        return full_url, state

    def exchange_code_for_token(self, authorization_code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.

        Args:
            authorization_code: The code received from OAuth callback

        Returns:
            dict: Token response containing access_token, refresh_token, etc.

        Raises:
            Exception: If token exchange fails
        """
        token_url = self.config.get_oauth_token_url()

        payload = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'client_id': self.config.CLIENT_ID,
            'client_secret': self.config.CLIENT_SECRET,
            'redirect_uri': self.config.REDIRECT_URI
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }

        try:
            response = requests.post(token_url, data=payload, headers=headers)
            response.raise_for_status()

            token_data = response.json()

            # Store tokens
            self.session_token = token_data.get('access_token')
            self.refresh_token = token_data.get('refresh_token')

            # Calculate expiry (default 15 minutes)
            expires_in = token_data.get('expires_in', 900)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

            # Save tokens to file
            self._save_tokens()

            print(f"Successfully obtained access token. Expires at: {self.token_expiry}")

            return token_data

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to exchange code for token: {e}")

    def refresh_session_token(self) -> Dict[str, Any]:
        """
        Refresh the session token using the refresh token.

        Returns:
            dict: New token data

        Raises:
            Exception: If token refresh fails
        """
        if not self.refresh_token:
            raise Exception("No refresh token available. Please re-authenticate.")

        token_url = self.config.get_oauth_token_url()

        payload = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.config.CLIENT_ID,
            'client_secret': self.config.CLIENT_SECRET
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }

        try:
            response = requests.post(token_url, data=payload, headers=headers)
            response.raise_for_status()

            token_data = response.json()

            # Update tokens
            self.session_token = token_data.get('access_token')

            # Update refresh token if provided
            if 'refresh_token' in token_data:
                self.refresh_token = token_data['refresh_token']

            # Calculate expiry
            expires_in = token_data.get('expires_in', 900)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

            # Save updated tokens
            self._save_tokens()

            print(f"Token refreshed successfully. New expiry: {self.token_expiry}")

            return token_data

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to refresh token: {e}")

    def is_token_expired(self) -> bool:
        """
        Check if the current session token is expired or about to expire.

        Returns:
            bool: True if token is expired or expires in < 60 seconds
        """
        if not self.token_expiry:
            return True

        # Add 60-second buffer
        return datetime.now() >= (self.token_expiry - timedelta(seconds=60))

    def ensure_valid_token(self) -> str:
        """
        Ensure we have a valid session token, refreshing if necessary.

        Returns:
            str: Valid session token

        Raises:
            Exception: If unable to obtain valid token
        """
        if self.is_token_expired():
            if self.refresh_token:
                print("Token expired, refreshing...")
                self.refresh_session_token()
            else:
                raise Exception("Session expired and no refresh token available. Please re-authenticate.")

        return self.session_token

    def get_api_quote_token(self) -> str:
        """
        Get API quote token for streaming market data.

        Returns:
            str: API quote token

        Raises:
            Exception: If unable to obtain quote token
        """
        # Check if we have a valid quote token
        if self.api_quote_token and self.quote_token_expiry:
            if datetime.now() < self.quote_token_expiry:
                return self.api_quote_token

        # Get new quote token
        session_token = self.ensure_valid_token()
        base_url = self.config.get_base_url()
        url = f"{base_url}/api-quote-tokens"

        headers = APIHeaders.get_default_headers(session_token)

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            token_data = data.get('data', {})

            self.api_quote_token = token_data.get('token')
            # Quote tokens expire after 24 hours
            self.quote_token_expiry = datetime.now() + timedelta(hours=24)

            print(f"API quote token obtained. Expires at: {self.quote_token_expiry}")

            return self.api_quote_token

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get API quote token: {e}")

    def get_customer_accounts(self) -> list[Dict[str, Any]]:
        """
        Get list of customer accounts.

        Returns:
            list: List of account dictionaries

        Raises:
            Exception: If unable to fetch accounts
        """
        session_token = self.ensure_valid_token()
        base_url = self.config.get_base_url()
        url = f"{base_url}/customers/me/accounts"

        headers = APIHeaders.get_default_headers(session_token)

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            accounts = data.get('data', {}).get('items', [])

            # Store first account number
            if accounts and not self.account_number:
                self.account_number = accounts[0].get('account-number')
                print(f"Account number set to: {self.account_number}")

            return accounts

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get customer accounts: {e}")

    def _save_tokens(self) -> None:
        """Save tokens to file for persistence."""
        token_data = {
            'session_token': self.session_token,
            'refresh_token': self.refresh_token,
            'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None,
            'api_quote_token': self.api_quote_token,
            'quote_token_expiry': self.quote_token_expiry.isoformat() if self.quote_token_expiry else None,
            'account_number': self.account_number
        }

        try:
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            print(f"Tokens saved to {self.token_file}")
        except Exception as e:
            print(f"Warning: Failed to save tokens to file: {e}")

    def load_tokens(self) -> bool:
        """
        Load tokens from file.

        Returns:
            bool: True if tokens loaded successfully
        """
        if not self.token_file.exists():
            return False

        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)

            self.session_token = token_data.get('session_token')
            self.refresh_token = token_data.get('refresh_token')
            self.api_quote_token = token_data.get('api_quote_token')
            self.account_number = token_data.get('account_number')

            # Parse expiry dates
            if token_data.get('token_expiry'):
                self.token_expiry = datetime.fromisoformat(token_data['token_expiry'])

            if token_data.get('quote_token_expiry'):
                self.quote_token_expiry = datetime.fromisoformat(token_data['quote_token_expiry'])

            print(f"Tokens loaded from {self.token_file}")
            return True

        except Exception as e:
            print(f"Warning: Failed to load tokens from file: {e}")
            return False

    def clear_tokens(self) -> None:
        """Clear all stored tokens."""
        self.session_token = None
        self.refresh_token = None
        self.token_expiry = None
        self.api_quote_token = None
        self.quote_token_expiry = None
        self.account_number = None

        # Delete token file
        if self.token_file.exists():
            self.token_file.unlink()
            print("Tokens cleared and file deleted")

    def get_auth_status(self) -> Dict[str, Any]:
        """
        Get current authentication status.

        Returns:
            dict: Authentication status information
        """
        return {
            'authenticated': bool(self.session_token),
            'token_valid': not self.is_token_expired() if self.session_token else False,
            'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None,
            'has_refresh_token': bool(self.refresh_token),
            'has_quote_token': bool(self.api_quote_token),
            'quote_token_expiry': self.quote_token_expiry.isoformat() if self.quote_token_expiry else None,
            'account_number': self.account_number
        }


if __name__ == '__main__':
    # Test authentication handler
    auth = TastytradeAuth()

    print("Tastytrade Authentication Handler Test")
    print("-" * 50)

    # Try to load existing tokens
    if auth.load_tokens():
        print("Existing tokens loaded!")
        status = auth.get_auth_status()
        print(f"Auth Status: {json.dumps(status, indent=2)}")
    else:
        print("No existing tokens found.")
        print("\nTo authenticate:")
        print("1. Run the OAuth2 callback server")
        print("2. Visit the authorization URL")
        print("3. Complete the OAuth flow")

        auth_url, state = auth.generate_auth_url()
        print(f"\nAuthorization URL:\n{auth_url}\n")
        print(f"State (for CSRF validation): {state}")
