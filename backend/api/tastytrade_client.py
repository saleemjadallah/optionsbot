"""
Tastytrade API Client

High-level client for interacting with Tastytrade API.
Provides convenient methods for common operations.
"""

import requests
from typing import Optional, Dict, List, Any
from datetime import datetime, date

from backend.api.tastytrade_auth import TastytradeAuth
from backend.config.tastytrade_config import TastytradeConfig, APIHeaders


class TastytradeClient:
    """High-level client for Tastytrade API operations."""

    def __init__(self, auth: Optional[TastytradeAuth] = None):
        """
        Initialize the Tastytrade client.

        Args:
            auth: Optional TastytradeAuth instance. If not provided, will create new one.
        """
        self.auth = auth or TastytradeAuth()
        self.config = TastytradeConfig
        self.base_url = self.config.get_base_url()

        # Load existing tokens if available
        self.auth.load_tokens()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make an authenticated API request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            data: Request body data

        Returns:
            dict: Response data

        Raises:
            Exception: If request fails
        """
        # Ensure we have a valid token
        session_token = self.auth.ensure_valid_token()

        # Build URL
        url = f"{self.base_url}{endpoint}"

        # Prepare headers
        headers = APIHeaders.get_default_headers(session_token)

        # Make request
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            error_msg = f"API request failed: {e}"
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"API Error: {error_data}"
                except:
                    pass
            raise Exception(error_msg)

    # ==================== Account Methods ====================

    def get_accounts(self) -> List[Dict[str, Any]]:
        """
        Get list of customer accounts.

        Returns:
            list: List of account dictionaries
        """
        response = self._make_request('GET', '/customers/me/accounts')
        return response.get('data', {}).get('items', [])

    def get_account_balances(self, account_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Get account balances.

        Args:
            account_number: Account number (uses default if not provided)

        Returns:
            dict: Balance data
        """
        account_num = account_number or self.auth.account_number
        response = self._make_request('GET', f'/accounts/{account_num}/balances')
        return response.get('data', {})

    def get_balance_snapshots(
        self,
        account_number: Optional[str] = None,
        time_of_day: str = 'EOD',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get account balance snapshots.

        Args:
            account_number: Account number
            time_of_day: 'BOD' or 'EOD'
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            list: Balance snapshot data
        """
        account_num = account_number or self.auth.account_number
        params = {'time-of-day': time_of_day}

        if start_date:
            params['start-date'] = start_date
        if end_date:
            params['end-date'] = end_date

        response = self._make_request(
            'GET',
            f'/accounts/{account_num}/balance-snapshots',
            params=params
        )
        return response.get('data', {}).get('items', [])

    def get_positions(
        self,
        account_number: Optional[str] = None,
        include_closed: bool = False,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get account positions.

        Args:
            account_number: Account number
            include_closed: Include closed positions
            symbol: Filter by symbol

        Returns:
            list: Position data
        """
        account_num = account_number or self.auth.account_number
        params = {'include-closed-positions': include_closed}

        if symbol:
            params['symbol'] = symbol

        response = self._make_request(
            'GET',
            f'/accounts/{account_num}/positions',
            params=params
        )
        return response.get('data', {}).get('items', [])

    # ==================== Market Data Methods ====================

    def get_equity(self, symbol: str) -> Dict[str, Any]:
        """
        Get equity instrument information.

        Args:
            symbol: Stock symbol

        Returns:
            dict: Equity data
        """
        response = self._make_request('GET', f'/instruments/equities/{symbol}')
        return response.get('data', {})

    def get_option_chain(
        self,
        symbol: str,
        nested: bool = True
    ) -> Dict[str, Any]:
        """
        Get option chain for a symbol.

        Args:
            symbol: Underlying symbol
            nested: Use nested format (grouped by expiration)

        Returns:
            dict: Option chain data
        """
        endpoint = f'/option-chains/{symbol}/nested' if nested else f'/option-chains/{symbol}'
        response = self._make_request('GET', endpoint)
        return response.get('data', {})

    def get_futures(self, product_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get futures contracts.

        Args:
            product_code: Filter by product code (e.g., 'ES', 'NQ')

        Returns:
            list: Futures data
        """
        params = {}
        if product_code:
            params['product-code'] = product_code

        response = self._make_request('GET', '/instruments/futures', params=params)
        return response.get('data', {}).get('items', [])

    def get_cryptocurrencies(self) -> List[Dict[str, Any]]:
        """
        Get available cryptocurrencies.

        Returns:
            list: Cryptocurrency data
        """
        response = self._make_request('GET', '/instruments/cryptocurrencies')
        return response.get('data', {}).get('items', [])

    # ==================== Order Methods ====================

    def dry_run_order(
        self,
        account_number: Optional[str] = None,
        order_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Test an order without executing it.

        Args:
            account_number: Account number
            order_data: Order specification

        Returns:
            dict: Dry run results including buying power impact
        """
        account_num = account_number or self.auth.account_number
        response = self._make_request(
            'POST',
            f'/accounts/{account_num}/orders/dry-run',
            data=order_data
        )
        return response.get('data', {})

    def place_order(
        self,
        account_number: Optional[str] = None,
        order_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Place a live order.

        Args:
            account_number: Account number
            order_data: Order specification

        Returns:
            dict: Order confirmation
        """
        account_num = account_number or self.auth.account_number
        response = self._make_request(
            'POST',
            f'/accounts/{account_num}/orders',
            data=order_data
        )
        return response.get('data', {})

    def get_orders(
        self,
        account_number: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get orders for account.

        Args:
            account_number: Account number
            status: Filter by status

        Returns:
            list: Order data
        """
        account_num = account_number or self.auth.account_number
        params = {}
        if status:
            params['status'] = status

        response = self._make_request(
            'GET',
            f'/accounts/{account_num}/orders',
            params=params
        )
        return response.get('data', {}).get('items', [])

    def cancel_order(
        self,
        order_id: str,
        account_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            account_number: Account number

        Returns:
            dict: Cancellation confirmation
        """
        account_num = account_number or self.auth.account_number
        response = self._make_request(
            'DELETE',
            f'/accounts/{account_num}/orders/{order_id}'
        )
        return response.get('data', {})

    # ==================== Transaction Methods ====================

    def get_transactions(
        self,
        account_number: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
        per_page: int = 250
    ) -> List[Dict[str, Any]]:
        """
        Get account transactions.

        Args:
            account_number: Account number
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Filter by symbol
            per_page: Results per page

        Returns:
            list: Transaction data
        """
        account_num = account_number or self.auth.account_number
        params = {'per-page': per_page}

        if start_date:
            params['start-date'] = start_date
        if end_date:
            params['end-date'] = end_date
        if symbol:
            params['symbol'] = symbol

        response = self._make_request(
            'GET',
            f'/accounts/{account_num}/transactions',
            params=params
        )
        return response.get('data', {}).get('items', [])

    # ==================== Watchlist Methods ====================

    def get_watchlists(self) -> List[Dict[str, Any]]:
        """
        Get user watchlists.

        Returns:
            list: Watchlist data
        """
        response = self._make_request('GET', '/watchlists')
        return response.get('data', {}).get('items', [])

    def create_watchlist(
        self,
        name: str,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Create a new watchlist.

        Args:
            name: Watchlist name
            symbols: List of symbols

        Returns:
            dict: Created watchlist
        """
        watchlist_data = {
            'name': name,
            'watchlist-entries': [
                {'symbol': symbol, 'instrument-type': 'Equity'}
                for symbol in symbols
            ]
        }

        response = self._make_request('POST', '/watchlists', data=watchlist_data)
        return response.get('data', {})

    # ==================== Helper Methods ====================

    def get_account_summary(self, account_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a comprehensive account summary.

        Args:
            account_number: Account number

        Returns:
            dict: Account summary with balances and positions
        """
        account_num = account_number or self.auth.account_number

        # Get account info
        accounts = self.get_accounts()
        account_info = next(
            (acc for acc in accounts if acc['account-number'] == account_num),
            None
        )

        # Get balances
        balances = self.get_account_balances(account_num)

        # Get positions
        positions = self.get_positions(account_num)

        return {
            'account_info': account_info,
            'balances': balances,
            'positions': positions,
            'position_count': len(positions)
        }


if __name__ == '__main__':
    # Example usage
    print("Initializing Tastytrade Client...")
    client = TastytradeClient()

    print("\nFetching accounts...")
    accounts = client.get_accounts()
    print(f"Found {len(accounts)} account(s)")

    if accounts:
        account_num = accounts[0]['account-number']
        print(f"\nAccount Summary for {account_num}:")

        summary = client.get_account_summary(account_num)
        print(f"  Authority: {summary['account_info']['authority-level']}")
        print(f"  Type: {summary['account_info']['account-type']}")
        print(f"  Positions: {summary['position_count']}")

        balances = summary['balances']
        print(f"  Cash Balance: ${balances.get('cash-balance', 0):,.2f}")
        print(f"  Net Liquidating Value: ${balances.get('net-liquidating-value', 0):,.2f}")
