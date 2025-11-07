"""
Tastytrade OAuth2 API Endpoints for FastAPI

These endpoints handle the OAuth2 authentication flow and API interactions
with Tastytrade, accessible from the Streamlit frontend.
"""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.api.tastytrade_auth import TastytradeAuth
from backend.api.tastytrade_client import TastytradeClient
from backend.config.tastytrade_config import TastytradeConfig

# Create router
router = APIRouter(prefix="/api/tastytrade", tags=["tastytrade"])

# Global auth instance (shared across requests)
auth_instance = TastytradeAuth()
client_instance = None


# ==================== Pydantic Models ====================

class AuthStatusResponse(BaseModel):
    authenticated: bool
    token_valid: bool
    token_expiry: Optional[str]
    has_refresh_token: bool
    account_number: Optional[str]
    environment: str


class AccountBalance(BaseModel):
    cash_balance: float
    net_liquidating_value: float
    equity_buying_power: float
    available_trading_funds: float


class Position(BaseModel):
    symbol: str
    quantity: float
    average_price: float
    current_price: Optional[float]
    unrealized_pnl: Optional[float]
    instrument_type: str


# ==================== Helper Functions ====================

def get_client() -> TastytradeClient:
    """Get or create TastytradeClient instance."""
    global client_instance
    if client_instance is None:
        client_instance = TastytradeClient(auth=auth_instance)
    return client_instance


# ==================== Authentication Endpoints ====================

@router.get("/auth/status")
async def get_auth_status():
    """
    Get current authentication status.

    Returns authentication state including token validity and account info.
    """
    try:
        # Try to load existing tokens
        auth_instance.load_tokens()

        status = auth_instance.get_auth_status()
        status['environment'] = 'sandbox' if TastytradeConfig.IS_SANDBOX else 'production'

        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get auth status: {str(e)}")


@router.get("/auth/url")
async def get_auth_url():
    """
    Generate OAuth2 authorization URL.

    Returns the URL that the user should visit to authenticate.
    """
    try:
        auth_url, state = auth_instance.generate_auth_url()

        return {
            "auth_url": auth_url,
            "state": state,
            "redirect_uri": TastytradeConfig.REDIRECT_URI
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate auth URL: {str(e)}")


@router.get("/auth/callback")
async def oauth_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None)
):
    """
    Handle OAuth2 callback from Tastytrade.

    This endpoint receives the authorization code and exchanges it for tokens.
    """
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")

    if not code:
        raise HTTPException(status_code=400, detail="No authorization code received")

    try:
        # Exchange code for token
        token_data = auth_instance.exchange_code_for_token(code)

        # Get account information
        accounts = auth_instance.get_customer_accounts()

        # Return success page HTML
        html_content = f"""
        <html>
        <head>
            <title>Authentication Successful</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                    text-align: center;
                    max-width: 500px;
                }}
                .success-icon {{
                    font-size: 64px;
                    color: #4caf50;
                    margin-bottom: 20px;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 10px;
                }}
                p {{
                    color: #666;
                    margin: 10px 0;
                }}
                .account-info {{
                    background: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .close-btn {{
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 12px 30px;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    margin-top: 20px;
                }}
                .close-btn:hover {{
                    background: #764ba2;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">✓</div>
                <h1>Authentication Successful!</h1>
                <p>You have successfully connected to Tastytrade.</p>

                <div class="account-info">
                    <strong>Account Connected:</strong><br>
                    {accounts[0]['account-number'] if accounts else 'N/A'}<br>
                    <small>({accounts[0]['account-type'] if accounts else 'N/A'})</small>
                </div>

                <p>You can now close this window and return to the dashboard.</p>

                <button class="close-btn" onclick="window.close()">Close Window</button>

                <script>
                    // Auto-close after 5 seconds
                    setTimeout(function() {{
                        window.close();
                    }}, 5000);
                </script>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to complete authentication: {str(e)}")


@router.post("/auth/refresh")
async def refresh_token():
    """
    Refresh the session token.

    Uses the refresh token to get a new session token.
    """
    try:
        token_data = auth_instance.refresh_session_token()

        return {
            "success": True,
            "message": "Token refreshed successfully",
            "token_expiry": auth_instance.token_expiry.isoformat() if auth_instance.token_expiry else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh token: {str(e)}")


@router.post("/auth/logout")
async def logout():
    """
    Clear authentication tokens and log out.
    """
    try:
        auth_instance.clear_tokens()
        global client_instance
        client_instance = None

        return {
            "success": True,
            "message": "Logged out successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to logout: {str(e)}")


# ==================== Account Endpoints ====================

@router.get("/accounts")
async def get_accounts():
    """Get list of customer accounts."""
    try:
        client = get_client()
        accounts = client.get_accounts()

        return {
            "success": True,
            "accounts": accounts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get accounts: {str(e)}")


@router.get("/accounts/{account_number}/balance")
async def get_account_balance(account_number: str):
    """Get account balance."""
    try:
        client = get_client()
        balance = client.get_account_balances(account_number)

        return {
            "success": True,
            "balance": balance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get balance: {str(e)}")


@router.get("/accounts/{account_number}/positions")
async def get_account_positions(
    account_number: str,
    include_closed: bool = False
):
    """Get account positions."""
    try:
        client = get_client()
        positions = client.get_positions(account_number, include_closed=include_closed)

        return {
            "success": True,
            "positions": positions,
            "count": len(positions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get positions: {str(e)}")


@router.get("/accounts/{account_number}/summary")
async def get_account_summary(account_number: str):
    """Get comprehensive account summary."""
    try:
        client = get_client()
        summary = client.get_account_summary(account_number)

        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


# ==================== Market Data Endpoints ====================

@router.get("/market/equity/{symbol}")
async def get_equity_data(symbol: str):
    """Get equity instrument data."""
    try:
        client = get_client()
        equity = client.get_equity(symbol)

        return {
            "success": True,
            "equity": equity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get equity data: {str(e)}")


@router.get("/market/option-chain/{symbol}")
async def get_option_chain(
    symbol: str,
    nested: bool = True
):
    """Get option chain for symbol."""
    try:
        client = get_client()
        chain = client.get_option_chain(symbol, nested=nested)

        return {
            "success": True,
            "chain": chain
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get option chain: {str(e)}")


# ==================== Order Endpoints ====================

@router.post("/orders/dry-run")
async def dry_run_order(order_data: Dict[str, Any]):
    """Test an order without executing it."""
    try:
        client = get_client()
        account_number = order_data.pop('account_number', None) or auth_instance.account_number

        result = client.dry_run_order(account_number, order_data)

        return {
            "success": True,
            "dry_run_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to dry run order: {str(e)}")


@router.post("/orders/place")
async def place_order(order_data: Dict[str, Any]):
    """Place a live order."""
    try:
        client = get_client()
        account_number = order_data.pop('account_number', None) or auth_instance.account_number

        result = client.place_order(account_number, order_data)

        return {
            "success": True,
            "order": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to place order: {str(e)}")


@router.get("/orders/{account_number}")
async def get_orders(
    account_number: str,
    status: Optional[str] = None
):
    """Get orders for account."""
    try:
        client = get_client()
        orders = client.get_orders(account_number, status=status)

        return {
            "success": True,
            "orders": orders,
            "count": len(orders)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get orders: {str(e)}")


# ==================== Transaction Endpoints ====================

@router.get("/transactions/{account_number}")
async def get_transactions(
    account_number: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: Optional[str] = None
):
    """Get account transactions."""
    try:
        client = get_client()
        transactions = client.get_transactions(
            account_number,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol
        )

        return {
            "success": True,
            "transactions": transactions,
            "count": len(transactions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transactions: {str(e)}")


# ==================== Health Check ====================

@router.get("/health")
async def health_check():
    """Check API health and authentication status."""
    status = auth_instance.get_auth_status()

    return {
        "status": "healthy",
        "authenticated": status['authenticated'],
        "environment": 'sandbox' if TastytradeConfig.IS_SANDBOX else 'production',
        "base_url": TastytradeConfig.get_base_url()
    }
