"""
Tastytrade API Configuration Module

This module manages Tastytrade OAuth2 configuration and provides
settings for both sandbox and production environments.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TastytradeConfig:
    """Configuration class for Tastytrade API authentication and settings."""

    # OAuth2 Configuration
    CLIENT_ID: str = os.getenv('TASTYTRADE_CLIENT_ID', '')
    CLIENT_SECRET: str = os.getenv('TASTYTRADE_CLIENT_SECRET', '')
    REDIRECT_URI: str = os.getenv('TASTYTRADE_REDIRECT_URI', 'http://localhost:8080/callback')
    SCOPES: str = os.getenv('TASTYTRADE_SCOPES', 'read,trade,openid')

    # Environment Settings
    IS_SANDBOX: bool = os.getenv('TASTYTRADE_SANDBOX', 'True').lower() == 'true'

    # Account Information
    USERNAME: str = os.getenv('TASTYTRADE_USERNAME', '')
    EMAIL: str = os.getenv('TASTYTRADE_EMAIL', '')
    ACCOUNT_NUMBER: Optional[str] = os.getenv('TASTYTRADE_ACCOUNT_NUMBER', None)

    # API Endpoints
    SANDBOX_BASE_URL: str = 'https://api.cert.tastyworks.com'
    PRODUCTION_BASE_URL: str = 'https://api.tastytrade.com'

    # WebSocket URLs
    SANDBOX_WS_URL: str = 'wss://streamer.cert.tastyworks.com'
    PRODUCTION_WS_URL: str = 'wss://tasty-openapi-ws.dxfeed.com/realtime'

    # OAuth2 Endpoints
    OAUTH_AUTHORIZE_PATH: str = '/oauth/authorize'
    OAUTH_TOKEN_PATH: str = '/oauth/token'

    # Session Settings
    SESSION_TOKEN_EXPIRY_SECONDS: int = 900  # 15 minutes
    API_QUOTE_TOKEN_EXPIRY_HOURS: int = 24

    # Rate Limiting
    MAX_ORDERS_PER_MINUTE: int = int(os.getenv('MAX_ORDERS_PER_MINUTE', '60'))
    MAX_API_CALLS_PER_SECOND: int = int(os.getenv('MAX_API_CALLS_PER_SECOND', '10'))

    @classmethod
    def get_base_url(cls) -> str:
        """Get the appropriate base URL based on environment."""
        return cls.SANDBOX_BASE_URL if cls.IS_SANDBOX else cls.PRODUCTION_BASE_URL

    @classmethod
    def get_ws_url(cls) -> str:
        """Get the appropriate WebSocket URL based on environment."""
        return cls.SANDBOX_WS_URL if cls.IS_SANDBOX else cls.PRODUCTION_WS_URL

    @classmethod
    def get_oauth_authorize_url(cls) -> str:
        """Get the full OAuth2 authorization URL."""
        base_url = cls.get_base_url()
        return f"{base_url}{cls.OAUTH_AUTHORIZE_PATH}"

    @classmethod
    def get_oauth_token_url(cls) -> str:
        """Get the full OAuth2 token URL."""
        base_url = cls.get_base_url()
        return f"{base_url}{cls.OAUTH_TOKEN_PATH}"

    @classmethod
    def validate_config(cls) -> tuple[bool, list[str]]:
        """
        Validate that all required configuration is present.

        Returns:
            tuple: (is_valid, list_of_missing_fields)
        """
        missing_fields = []

        if not cls.CLIENT_ID:
            missing_fields.append('TASTYTRADE_CLIENT_ID')

        if not cls.CLIENT_SECRET:
            missing_fields.append('TASTYTRADE_CLIENT_SECRET')

        if not cls.REDIRECT_URI:
            missing_fields.append('TASTYTRADE_REDIRECT_URI')

        is_valid = len(missing_fields) == 0
        return is_valid, missing_fields

    @classmethod
    def get_config_summary(cls) -> dict:
        """
        Get a summary of current configuration (without sensitive data).

        Returns:
            dict: Configuration summary
        """
        return {
            'environment': 'sandbox' if cls.IS_SANDBOX else 'production',
            'base_url': cls.get_base_url(),
            'ws_url': cls.get_ws_url(),
            'redirect_uri': cls.REDIRECT_URI,
            'scopes': cls.SCOPES,
            'client_id': cls.CLIENT_ID[:8] + '...' if cls.CLIENT_ID else 'NOT SET',
            'client_secret_set': bool(cls.CLIENT_SECRET),
            'username': cls.USERNAME,
            'email': cls.EMAIL,
            'account_number': cls.ACCOUNT_NUMBER if cls.ACCOUNT_NUMBER else 'Not fetched yet'
        }


# API Headers Configuration
class APIHeaders:
    """Standard headers for Tastytrade API requests."""

    USER_AGENT: str = 'tastyslim-options-trader/1.0.0'
    CONTENT_TYPE: str = 'application/json'
    ACCEPT: str = 'application/json'

    @classmethod
    def get_default_headers(cls, token: Optional[str] = None) -> dict:
        """
        Get default headers for API requests.

        Args:
            token: Optional session token for authorization

        Returns:
            dict: Headers dictionary
        """
        headers = {
            'User-Agent': cls.USER_AGENT,
            'Content-Type': cls.CONTENT_TYPE,
            'Accept': cls.ACCEPT
        }

        if token:
            headers['Authorization'] = f'Bearer {token}'

        return headers


# Validation function for startup
def validate_tastytrade_config() -> None:
    """
    Validate Tastytrade configuration on startup.
    Raises an exception if configuration is invalid.
    """
    is_valid, missing_fields = TastytradeConfig.validate_config()

    if not is_valid:
        raise ValueError(
            f"Invalid Tastytrade configuration. Missing fields: {', '.join(missing_fields)}\n"
            "Please check your .env file and ensure all required fields are set."
        )

    print(f"Tastytrade configuration validated successfully!")
    print(f"Environment: {'Sandbox' if TastytradeConfig.IS_SANDBOX else 'Production'}")
    print(f"Base URL: {TastytradeConfig.get_base_url()}")


if __name__ == '__main__':
    # Test configuration
    print("Tastytrade Configuration Summary:")
    print("-" * 50)

    config_summary = TastytradeConfig.get_config_summary()
    for key, value in config_summary.items():
        print(f"{key}: {value}")

    print("\n" + "-" * 50)

    try:
        validate_tastytrade_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
