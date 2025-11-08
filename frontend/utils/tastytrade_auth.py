"""
Tastytrade Authentication Utilities for Streamlit

Implements session-based authentication directly against the Tastytrade
sandbox so the Streamlit app can run without a separate backend service.
"""

from __future__ import annotations

import requests
import streamlit as st
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from tastytrade.session import Session as TastytradeSession


class TastytradeAuthManager:
    """Manage session-based authentication with the Tastytrade sandbox API."""

    DEFAULT_BASE_URL = "https://api.cert.tastyworks.com"
    USER_AGENT = "tastyslim-options-trader/1.0.0"
    SESSION_STATE_KEY = "tastytrade_session"

    def __init__(self, api_base_url: Optional[str] = None):
        # Allow callers to override the base URL (e.g., for production).
        requested_base = (api_base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._initialize_session_state()
        state = self._get_session_info()

        stored_base = state.get("api_base_url")
        if stored_base:
            self.base_url = stored_base
            self.is_test = state.get("is_test", "cert" in stored_base)
        else:
            self.base_url = requested_base
            self.is_test = "cert" in self.base_url
            self._update_session_info(api_base_url=self.base_url, is_test=self.is_test)

        self._sdk_session: Optional[TastytradeSession] = None
        self.username: Optional[str] = None

    def _initialize_session_state(self) -> None:
        """Ensure all required Streamlit session state keys exist."""
        if self.SESSION_STATE_KEY not in st.session_state:
            st.session_state[self.SESSION_STATE_KEY] = {
                "session_token": None,
                "remember_token": None,
                "token_expiry": None,
                "account_number": None,
                "username": None,
                "api_base_url": self.DEFAULT_BASE_URL,
                "is_test": True,
            }

        if "tastytrade_accounts" not in st.session_state:
            st.session_state.tastytrade_accounts = []

        if "tastytrade_authenticated" not in st.session_state:
            st.session_state.tastytrade_authenticated = False

        if "tastytrade_auth_checked" not in st.session_state:
            st.session_state.tastytrade_auth_checked = False

    def _get_session_info(self) -> Dict[str, Any]:
        if self.SESSION_STATE_KEY not in st.session_state:
            self._initialize_session_state()
        return st.session_state[self.SESSION_STATE_KEY]

    def _update_session_info(self, **kwargs: Any) -> None:
        session_info = self._get_session_info()
        session_info.update(kwargs)
        st.session_state[self.SESSION_STATE_KEY] = session_info

    def _clear_session(self) -> None:
        if self.SESSION_STATE_KEY not in st.session_state:
            self._initialize_session_state()
        state = st.session_state[self.SESSION_STATE_KEY]
        base_url = state.get("api_base_url", self.DEFAULT_BASE_URL)
        is_test = state.get("is_test", "cert" in base_url)
        st.session_state[self.SESSION_STATE_KEY] = {
            "session_token": None,
            "remember_token": None,
            "token_expiry": None,
            "account_number": None,
            "username": None,
            "api_base_url": base_url,
            "is_test": is_test,
        }
        st.session_state.tastytrade_accounts = []
        st.session_state.tastytrade_authenticated = False
        self._sdk_session = None
        self.username = None
        self.base_url = base_url
        self.is_test = is_test

    def _token_expired(self) -> bool:
        token_expiry = self._get_session_info().get("token_expiry")

        if not token_expiry:
            return True

        try:
            expiry = datetime.fromisoformat(token_expiry)
        except ValueError:
            return True

        return datetime.utcnow() >= (expiry - timedelta(seconds=60))

    def _get_headers(self, include_auth: bool = True) -> Dict[str, str]:
        headers = {
            "User-Agent": self.USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if include_auth:
            session_token = self._get_session_info().get("session_token")

            if not session_token or self._token_expired():
                raise Exception("Session expired. Please log in again.")

            # Sandbox session authentication expects the raw session token header
            headers["Authorization"] = session_token

        return headers

    def set_environment(self, environment: str) -> None:
        env = environment.lower()
        if env == "production":
            new_base = "https://api.tastytrade.com"
            is_test = False
        else:
            new_base = self.DEFAULT_BASE_URL
            is_test = True

        if new_base != self.base_url or is_test != getattr(self, "is_test", None):
            self.base_url = new_base
            self.is_test = is_test
            self._sdk_session = None
            info = self._get_session_info()
            info.update({"api_base_url": new_base, "is_test": is_test})
            st.session_state[self.SESSION_STATE_KEY] = info

    def login(
        self,
        username: str,
        password: str,
        remember_me: bool = True,
        environment: str = "sandbox",
    ) -> Dict[str, Any]:
        """
        Authenticate with username/password using the sandbox session endpoint.

        Returns:
            dict: Session data returned by Tastytrade.
        """
        self.set_environment(environment)

        url = f"{self.base_url}/sessions"
        payload = {
            "login": username,
            "password": password,
            "remember-me": remember_me,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._get_headers(include_auth=False),
                timeout=10,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            message = "Login failed."
            if exc.response is not None:
                try:
                    error_payload = exc.response.json()
                    message = error_payload.get("error", {}).get("message", message)
                except ValueError:
                    message = exc.response.text or message
            self._clear_session()
            raise Exception(message) from exc
        except requests.exceptions.RequestException as exc:
            self._clear_session()
            raise Exception(f"Network error during login: {exc}") from exc

        session_data = response.json().get("data", {})
        session_token = session_data.get("session-token")

        if not session_token:
            self._clear_session()
            raise Exception("No session token returned by Tastytrade.")

        remember_token = session_data.get("remember-token")
        expiry = datetime.utcnow() + timedelta(hours=24)

        self._update_session_info(
            session_token=session_token,
            remember_token=remember_token,
            token_expiry=expiry.isoformat(),
            account_number=None,
            username=username,
        )
        self.username = username
        self._sdk_session = None

        try:
            self.get_accounts()
        except Exception:
            pass

        st.session_state.tastytrade_authenticated = True
        st.session_state.tastytrade_auth_checked = True

        return session_data

    def logout(self) -> bool:
        """Clear local session state and attempt to invalidate the session server-side."""
        session_token = self._get_session_info().get("session_token")

        if session_token:
            url = f"{self.base_url}/sessions"
            try:
                requests.delete(url, headers=self._get_headers(), timeout=5)
            except requests.exceptions.RequestException:
                # Ignore errors when tearing down the remote session.
                pass

        self._clear_session()
        self._sdk_session = None
        self.username = None
        return True

    def check_auth_status(self) -> Dict[str, Any]:
        """Return the current authentication state without making remote calls."""
        session_info = self._get_session_info()
        token_valid = bool(session_info.get("session_token")) and not self._token_expired()

        st.session_state.tastytrade_authenticated = token_valid
        st.session_state.tastytrade_auth_checked = True

        return {
            "authenticated": token_valid,
            "token_valid": token_valid,
            "token_expiry": session_info.get("token_expiry"),
            "has_refresh_token": bool(session_info.get("remember_token")),
            "account_number": session_info.get("account_number"),
            "environment": "sandbox" if "cert" in self.base_url else "production",
        }

    def get_accounts(self) -> List[Dict[str, Any]]:
        """Fetch customer accounts using the active session token."""
        url = f"{self.base_url}/customers/me/accounts"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise Exception(f"Failed to fetch accounts: {exc}") from exc

        raw_items = response.json().get("data", {}).get("items", [])
        parsed_accounts: List[Dict[str, Any]] = []

        for item in raw_items:
            if isinstance(item, dict):
                account_data = item.get("account") if isinstance(item.get("account"), dict) else item
                parsed_accounts.append(account_data)

        st.session_state.tastytrade_accounts = parsed_accounts

        if parsed_accounts:
            account_number = (
                parsed_accounts[0].get("account-number")
                or parsed_accounts[0].get("account_number")
            )
            if account_number:
                self._update_session_info(account_number=account_number)
                return parsed_accounts

        raise Exception("No accounts returned for the current credentials/environment.")

    def ensure_account(self) -> str:
        account_num = self._get_session_info().get("account_number")
        if account_num:
            return account_num
        accounts = self.get_accounts()
        if not accounts:
            raise Exception("No accounts available.")
        account_num = accounts[0].get("account-number")
        if not account_num:
            raise Exception("Account number missing from response.")
        self._update_session_info(account_number=account_num)
        return account_num

    def get_sdk_session(self) -> TastytradeSession:
        """
        Get or initialize a tastytrade SDK session using the stored remember token.
        """
        if self._sdk_session is not None:
            return self._sdk_session

        session_info = self._get_session_info()
        remember_token = session_info.get("remember_token")
        username = session_info.get("username") or self.username

        if not username:
            raise Exception("Username is not available. Please authenticate again.")

        if not remember_token:
            raise Exception("Remember token missing. Please log in again.")

        try:
            sdk_session = TastytradeSession(
                login=username,
                remember_token=remember_token,
                remember_me=True,
                is_test=self.is_test,
            )
        except Exception as exc:
            self._sdk_session = None
            raise Exception(f"Failed to initialize Tastytrade SDK session: {exc}") from exc

        self._sdk_session = sdk_session
        self.username = username
        self._update_session_info(
            session_token=sdk_session.session_token,
            remember_token=sdk_session.remember_token,
            username=username,
            account_number=session_info.get("account_number"),
        )

        return sdk_session

    def get_account_balance(self, account_number: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return balance data for a specific account."""
        account_num = account_number or self._get_session_info().get("account_number")
        if not account_num:
            account_num = self.ensure_account()

        url = f"{self.base_url}/accounts/{account_num}/balances"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise Exception(f"Failed to fetch account balance: {exc}") from exc

        return response.json().get("data", {})

    def get_account_positions(
        self,
        account_number: Optional[str] = None,
        include_closed: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return open (or optionally closed) positions for the account."""
        account_num = account_number or self._get_session_info().get("account_number")
        if not account_num:
            account_num = self.ensure_account()

        params = {"include-closed": str(include_closed).lower()}
        url = f"{self.base_url}/accounts/{account_num}/positions"

        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=10,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise Exception(f"Failed to fetch account positions: {exc}") from exc

        data = response.json().get("data", {})
        return data.get("items") or data.get("positions") or []

    def get_account_transactions(
        self,
        account_number: Optional[str] = None,
        per_page: int = 250,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        types: Optional[List[str]] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent transactions (trades, dividends, fees, etc.) for the account."""
        account_num = account_number or self._get_session_info().get("account_number")
        if not account_num:
            account_num = self.ensure_account()

        page_limit = max(1, min(per_page, 2000))
        params: Dict[str, Any] = {
            "per-page": page_limit,
            "page-offset": 0,
            "sort": "desc",
        }
        if start_date:
            params["start-date"] = start_date
        if end_date:
            params["end-date"] = end_date
        if symbol:
            params["symbol"] = symbol
        if types:
            params["types"] = types

        url = f"{self.base_url}/accounts/{account_num}/transactions"

        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=15,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise Exception(f"Failed to fetch account transactions: {exc}") from exc

        data = response.json().get("data", {})
        return data.get("items") or data.get("transactions") or []

    def get_watchlists(self) -> List[Dict[str, Any]]:
        """Return the user's private watchlists."""
        url = f"{self.base_url}/watchlists"

        try:
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:  # pragma: no cover - network
            raise Exception(f"Failed to fetch watchlists: {exc}") from exc

        data = response.json().get("data", {})
        watchlists = data.get("items") or data.get("watchlists") or []
        return watchlists

    def get_watchlist_symbols(self, name: Optional[str] = None) -> List[str]:
        """Return symbols from the specified (or first) watchlist."""
        watchlists = self.get_watchlists()
        if not watchlists:
            return []

        target = None
        if name:
            for watchlist in watchlists:
                if str(watchlist.get("name", "")).lower() == name.lower():
                    target = watchlist
                    break

        if target is None:
            target = watchlists[0]

        entries = (
            target.get("watchlist-entries")
            or target.get("watchlist_entries")
            or target.get("items")
            or []
        )

        symbols: List[str] = []
        for entry in entries:
            symbol = (
                entry.get("symbol")
                or entry.get("underlying-symbol")
                or entry.get("underlying_symbol")
            )
            if symbol:
                symbols.append(str(symbol).strip().upper())

        # Some watchlist responses store symbols nested under "instrument"
        if not symbols:
            for entry in entries:
                instrument = entry.get("instrument") if isinstance(entry, dict) else None
                symbol = (
                    (instrument or {}).get("symbol")
                    or (instrument or {}).get("underlying-symbol")
                )
                if symbol:
                    symbols.append(str(symbol).strip().upper())

        # Remove duplicates while preserving order
        seen = set()
        unique_symbols: List[str] = []
        for sym in symbols:
            if sym and sym not in seen:
                unique_symbols.append(sym)
                seen.add(sym)

        return unique_symbols

    def get_account_summary(self, account_number: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Alias to balance endpoint for compatibility with existing UI."""
        return self.get_account_balance(account_number=account_number)

    def is_authenticated(self) -> bool:
        return bool(self.check_auth_status().get("token_valid"))

    def require_authentication(self) -> bool:
        """Helper for pages that should only render when authenticated."""
        if not self.is_authenticated():
            st.warning("⚠️ You need to authenticate with Tastytrade to access this feature.")
            return False

        return True

    def get_account_number(self) -> str:
        """Return the cached account number, fetching accounts if needed."""
        account_num = self._get_session_info().get("account_number")
        if account_num:
            return account_num
        return self.ensure_account()

    def dry_run_order(self, order: Dict[str, Any], account_number: Optional[str] = None) -> Dict[str, Any]:
        """Perform a dry-run order placement to preview impact without executing."""
        account_num = account_number or self.get_account_number()
        payload = dict(order)
        payload.setdefault("source", "streamlit-app")

        url = f"{self.base_url}/accounts/{account_num}/orders/dry-run"

        response = requests.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=15,
        )
        self._raise_for_tastytrade_error(response, "dry run")
        return response.json()

    def place_order(self, order: Dict[str, Any], account_number: Optional[str] = None) -> Dict[str, Any]:
        """Place an order using the active session."""
        account_num = account_number or self.get_account_number()
        payload = dict(order)
        payload.setdefault("source", "streamlit-app")

        url = f"{self.base_url}/accounts/{account_num}/orders"

        response = requests.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=15,
        )
        self._raise_for_tastytrade_error(response, "order placement")
        return response.json()

    def _raise_for_tastytrade_error(self, response: requests.Response, context: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network errors
            detail = self._extract_error_detail(response)
            raise RuntimeError(
                f"Tastytrade {context} failed ({response.status_code}): {detail}"
            ) from exc

    @staticmethod
    def _extract_error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text or "Unknown error"

        sections: List[str] = []

        single_error = payload.get("error")
        if isinstance(single_error, dict):
            base_msg = single_error.get("message") or single_error.get("code")
            if base_msg:
                sections.append(base_msg)

        list_errors = payload.get("errors")
        if isinstance(list_errors, list):
            for err in list_errors:
                if not isinstance(err, dict):
                    continue
                msg = err.get("message") or err.get("code")
                detail = err.get("details")
                if isinstance(detail, dict):
                    detail_parts = [f"{k}: {v}" for k, v in detail.items()]
                    detail_text = "; ".join(detail_parts)
                    msg = f"{msg} ({detail_text})" if detail_text else msg
                if msg:
                    sections.append(msg)

        if not sections:
            detail = payload.get("detail") or payload.get("message")
            if detail:
                sections.append(str(detail))

        if sections:
            return " | ".join(sections)
        return str(payload) or "Unknown error"



_auth_manager: Optional[TastytradeAuthManager] = None


def get_auth_manager(api_base_url: Optional[str] = None) -> TastytradeAuthManager:
    """Return a singleton instance of the auth manager."""
    global _auth_manager

    if _auth_manager is None:
        _auth_manager = TastytradeAuthManager(api_base_url=api_base_url)

    return _auth_manager
