# Tastytrade Integration Setup Guide

This guide will walk you through setting up Tastytrade API integration for your Options Trader application.

---

## Prerequisites

- Python 3.8+
- Virtual environment activated
- Tastytrade sandbox account created
- OAuth2 application registered

---

## Step 1: Environment Configuration

Your `.env` file has already been configured with the following credentials:

```bash
# Tastytrade OAuth2 Configuration
TASTYTRADE_CLIENT_ID=87a97b47-1714-4ed5-80e2-9b362e270fb3
TASTYTRADE_CLIENT_SECRET=2f1857deda8d116a16fd090bd2e8321a89bef172
TASTYTRADE_REDIRECT_URI=http://localhost:8080/callback
TASTYTRADE_SCOPES=read,trade,openid
TASTYTRADE_SANDBOX=True

# Tastytrade Sandbox Account
TASTYTRADE_USERNAME=tastyslim
TASTYTRADE_EMAIL=saleem86@gmail.com
```

---

## Step 2: Install Required Dependencies

Make sure you have the required packages installed:

```bash
pip install requests python-dotenv
```

---

## Step 3: Authenticate with Tastytrade

Run the OAuth2 authentication flow:

```bash
cd /Users/saleemjadallah/Desktop/OptionsTrader
python backend/api/oauth_callback_server.py
```

This will:
1. Start a local callback server on port 8080
2. Open your browser to the Tastytrade authorization page
3. After you log in and authorize, redirect back to localhost
4. Exchange the authorization code for access/refresh tokens
5. Save tokens to `backend/credentials/tokens.json`

**What to expect:**
- Browser opens automatically to Tastytrade login
- Log in with your sandbox credentials:
  - Username: `tastyslim`
  - Password: `saleemjadallah1986`
- Click "Authorize" to grant permissions
- You'll be redirected to a success page
- Close the browser and return to terminal

---

## Step 4: Verify Authentication

After successful authentication, you should see:

```
Authentication Complete!
Found X account(s):
  - [ACCOUNT_NUMBER] (owner) - Cash - Individual

Testing API Access
API Quote Token obtained successfully!

Success! You are now authenticated with Tastytrade.
```

---

## Step 5: Using the API in Your Application

### Import the Auth Handler

```python
from backend.api.tastytrade_auth import TastytradeAuth

# Initialize
auth = TastytradeAuth()

# Load saved tokens
auth.load_tokens()

# Ensure token is valid (auto-refreshes if needed)
session_token = auth.ensure_valid_token()

# Get accounts
accounts = auth.get_customer_accounts()
print(f"Account Number: {auth.account_number}")
```

### Make API Requests

```python
import requests
from backend.config.tastytrade_config import TastytradeConfig, APIHeaders

# Get session token
session_token = auth.ensure_valid_token()

# Prepare request
base_url = TastytradeConfig.get_base_url()
headers = APIHeaders.get_default_headers(session_token)

# Example: Get account positions
url = f"{base_url}/accounts/{auth.account_number}/positions"
response = requests.get(url, headers=headers)
positions = response.json()
```

---

## Step 6: Working with Market Data

### Get Quote Token for Streaming

```python
# Get API quote token
quote_token = auth.get_api_quote_token()

# Use for DXLink WebSocket streaming
# See TASTYTRADE_API_DOCUMENTATION.md for details
```

---

## API Endpoints Available

With your authenticated session, you can access:

### Account Management
- `GET /customers/me/accounts` - List accounts
- `GET /accounts/{account_number}/balances` - Get balances
- `GET /accounts/{account_number}/positions` - Get positions
- `GET /accounts/{account_number}/balance-snapshots` - Historical balances

### Market Data
- `GET /api-quote-tokens` - Get streaming token
- `GET /instruments/equities/{symbol}` - Get equity info
- `GET /option-chains/{symbol}/nested` - Get option chains
- `GET /instruments/futures` - Get futures

### Order Management
- `POST /accounts/{account_number}/orders/dry-run` - Test order
- `POST /accounts/{account_number}/orders` - Place order
- `GET /accounts/{account_number}/orders` - List orders
- `DELETE /accounts/{account_number}/orders/{order_id}` - Cancel order

### Transactions
- `GET /accounts/{account_number}/transactions` - Transaction history
- `GET /accounts/{account_number}/transactions/total-fees` - Fee summary

### Watchlists
- `GET /watchlists` - Get watchlists
- `POST /watchlists` - Create watchlist
- `PUT /watchlists/{name}` - Update watchlist

---

## Token Management

### Automatic Token Refresh

The `TastytradeAuth` class automatically handles token refresh:

```python
# This will auto-refresh if token is expired
session_token = auth.ensure_valid_token()
```

### Token Expiry Times

- **Session Token**: 15 minutes (auto-refreshed)
- **Refresh Token**: Never expires
- **Quote Token**: 24 hours (auto-refreshed)

### Check Authentication Status

```python
status = auth.get_auth_status()
print(status)
# {
#   'authenticated': True,
#   'token_valid': True,
#   'token_expiry': '2025-01-06T15:30:00',
#   'has_refresh_token': True,
#   'account_number': '5WV12345'
# }
```

### Clear Tokens (Logout)

```python
auth.clear_tokens()
```

---

## Testing Your Setup

### Test Configuration

```bash
cd backend/config
python tastytrade_config.py
```

Expected output:
```
Tastytrade Configuration Summary:
--------------------------------------------------
environment: sandbox
base_url: https://api.cert.tastytrade.com
ws_url: wss://tasty-openapi-ws.cert.dxfeed.com/realtime
client_id: 87a97b47...
...
Tastytrade configuration validated successfully!
```

### Test Authentication

```bash
cd backend/api
python tastytrade_auth.py
```

---

## Sandbox vs Production

### Current Setup (Sandbox)
```bash
TASTYTRADE_SANDBOX=True
```

- Base URL: `https://api.cert.tastytrade.com`
- WebSocket: `wss://tasty-openapi-ws.cert.dxfeed.com/realtime`
- Safe for testing, no real money

### Switching to Production

**⚠️ IMPORTANT: Only switch to production when ready for live trading**

1. Update `.env`:
```bash
TASTYTRADE_SANDBOX=False
```

2. Create production OAuth application at https://developer.tastytrade.com/
3. Update credentials in `.env` with production values
4. Re-authenticate using production credentials

---

## Security Best Practices

### Protect Your Credentials

1. **Never commit `.env` to git** (already in `.gitignore`)
2. **Never share your Client Secret**
3. **Token file is auto-generated** at `backend/credentials/tokens.json` (also gitignored)

### Stored Files

```
backend/
  credentials/
    tokens.json          # Auto-generated, gitignored
  .env                   # Your credentials, gitignored
```

### Rate Limiting

Configured in `.env`:
```bash
MAX_ORDERS_PER_MINUTE=60
MAX_API_CALLS_PER_SECOND=10
```

---

## Troubleshooting

### Authentication Failed

**Problem**: Can't authenticate or token exchange fails

**Solutions**:
1. Verify credentials in `.env` are correct
2. Check Client ID and Secret match OAuth app
3. Ensure redirect URI is exactly: `http://localhost:8080/callback`
4. Check if port 8080 is available
5. Try clearing tokens: `auth.clear_tokens()`

### Token Expired

**Problem**: Getting 401 Unauthorized errors

**Solutions**:
1. Token auto-refresh should handle this
2. Manually refresh: `auth.refresh_session_token()`
3. Re-authenticate if refresh token is invalid

### Cannot Access Accounts

**Problem**: Empty account list or permission errors

**Solutions**:
1. Verify scopes include: `read,trade,openid`
2. Check authority level is `owner` for full access
3. Confirm sandbox account is fully activated

### Port Already in Use

**Problem**: Port 8080 is already taken

**Solutions**:
```bash
# Option 1: Kill process using port 8080
lsof -ti:8080 | xargs kill -9

# Option 2: Use different port
# Update TASTYTRADE_REDIRECT_URI in .env
# Update OAuth app redirect URI in Tastytrade dashboard
```

---

## Next Steps

1. **Test API Access**: Try fetching positions and balances
2. **Implement Market Data**: Set up DXLink streaming
3. **Create Order Flow**: Implement dry-run and live order placement
4. **Build Trading Strategies**: Integrate with your existing options strategies
5. **Add Risk Management**: Implement position limits and monitoring

---

## Resources

- **Full API Documentation**: See `TASTYTRADE_API_DOCUMENTATION.md`
- **Official Docs**: https://developer.tastytrade.com/
- **Configuration**: `backend/config/tastytrade_config.py`
- **Auth Handler**: `backend/api/tastytrade_auth.py`
- **OAuth Server**: `backend/api/oauth_callback_server.py`

---

## Quick Reference Commands

```bash
# Authenticate
python backend/api/oauth_callback_server.py

# Test configuration
python backend/config/tastytrade_config.py

# Test auth status
python backend/api/tastytrade_auth.py

# Clear tokens and re-authenticate
rm backend/credentials/tokens.json
python backend/api/oauth_callback_server.py
```

---

**You're all set! Run the authentication script to get started.**

```bash
python backend/api/oauth_callback_server.py
```
