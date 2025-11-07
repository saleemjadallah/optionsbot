"""
Tastytrade Session-Based Authentication (Alternative to OAuth2)

Tastyworks sandbox uses session-based authentication with username/password.
This is simpler than OAuth2 and works for the sandbox environment.
"""

import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from backend.config.tastytrade_config import TastytradeConfig, APIHeaders


class TastytradeSessionAuth:
    """Session-based authentication for Tastyworks API."""

    def __init__(self):
        """Initialize the session auth handler."""
        self.config = TastytradeConfig
        self.session_token: Optional[str] = None
        self.remember_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self.account_number: Optional[str] = None
        self.base_url = self.config.get_base_url()

        # Token storage file
        self.token_file = Path(__file__).parent.parent / 'credentials' / 'session_tokens.json'
        self.token_file.parent.mkdir(parents=True, exist_ok=True)

    def login(self, username: str, password: str, remember_me: bool = True) -> Dict[str, Any]:
        """
        Login with username and password.

        Args:
            username: Tastyworks username
            password: Password
            remember_me: Whether to get a remember token

        Returns:
            dict: Session data

        Raises:
            Exception: If login fails
        """
        url = f"{self.base_url}/sessions"

        payload = {
            "login": username,
            "password": password,
            "remember-me": remember_me
        }

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()
            session_data = data.get('data', {})

            # Store tokens
            self.session_token = session_data.get('session-token')
            self.remember_token = session_data.get('remember-token')

            # Session tokens typically last 24 hours
            self.token_expiry = datetime.now() + timedelta(hours=24)

            # Get user info
            user_data = session_data.get('user', {})

            # Save tokens
            self._save_tokens()

            print(f"✅ Successfully logged in as {user_data.get('username')}")

            return session_data

        except requests.exceptions.RequestException as e:
            error_msg = f"Login failed: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"Login failed: {error_data.get('error', {}).get('message', str(e))}"
                except:
                    pass
            raise Exception(error_msg)

    def is_token_expired(self) -> bool:
        """Check if session token is expired."""
        if not self.token_expiry:
            return True
        return datetime.now() >= self.token_expiry

    def ensure_valid_token(self) -> str:
        """
        Ensure we have a valid session token.

        Returns:
            str: Valid session token

        Raises:
            Exception: If no valid token available
        """
        if self.is_token_expired() or not self.session_token:
            raise Exception("Session expired. Please log in again.")

        return self.session_token

    def get_customer_accounts(self) -> list:
        """
        Get customer accounts.

        Returns:
            list: List of accounts
        """
        session_token = self.ensure_valid_token()
        url = f"{self.base_url}/customers/me/accounts"

        headers = APIHeaders.get_default_headers(session_token)

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            accounts = data.get('data', {}).get('items', [])

            # Store first account number
            if accounts and not self.account_number:
                self.account_number = accounts[0].get('account-number')
                self._save_tokens()

            return accounts

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get accounts: {e}")

    def logout(self) -> bool:
        """Logout and clear tokens."""
        if self.session_token:
            url = f"{self.base_url}/sessions"
            headers = APIHeaders.get_default_headers(self.session_token)

            try:
                requests.delete(url, headers=headers)
            except:
                pass  # Ignore errors on logout

        self.clear_tokens()
        return True

    def clear_tokens(self):
        """Clear all stored tokens."""
        self.session_token = None
        self.remember_token = None
        self.token_expiry = None
        self.account_number = None

        if self.token_file.exists():
            self.token_file.unlink()

    def _save_tokens(self):
        """Save tokens to file."""
        token_data = {
            'session_token': self.session_token,
            'remember_token': self.remember_token,
            'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None,
            'account_number': self.account_number
        }

        try:
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save tokens: {e}")

    def load_tokens(self) -> bool:
        """Load tokens from file."""
        if not self.token_file.exists():
            return False

        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)

            self.session_token = token_data.get('session_token')
            self.remember_token = token_data.get('remember_token')
            self.account_number = token_data.get('account_number')

            if token_data.get('token_expiry'):
                self.token_expiry = datetime.fromisoformat(token_data['token_expiry'])

            return True
        except Exception as e:
            print(f"Warning: Failed to load tokens: {e}")
            return False

    def get_auth_status(self) -> Dict[str, Any]:
        """Get current authentication status."""
        return {
            'authenticated': bool(self.session_token),
            'token_valid': not self.is_token_expired() if self.session_token else False,
            'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None,
            'account_number': self.account_number,
            'environment': 'sandbox' if self.config.IS_SANDBOX else 'production'
        }


if __name__ == '__main__':
    # Test session authentication
    import os
    from dotenv import load_dotenv

    load_dotenv()

    auth = TastytradeSessionAuth()

    username = os.getenv('TASTYTRADE_USERNAME', 'tastyslim')
    password = input(f"Enter password for {username}: ")

    try:
        auth.login(username, password)
        print("\n✅ Login successful!")

        accounts = auth.get_customer_accounts()
        print(f"\nAccounts: {len(accounts)}")
        for acc in accounts:
            print(f"  - {acc['account-number']} ({acc['account-type']})")

        status = auth.get_auth_status()
        print(f"\nAuth Status: {json.dumps(status, indent=2)}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
