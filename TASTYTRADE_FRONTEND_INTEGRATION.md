# Tastytrade Frontend Integration Guide

Complete integration of Tastytrade authentication into your Streamlit application.

---

## 🎉 What Was Added

### Backend API Endpoints (`backend/api/tastytrade_endpoints.py`)

FastAPI endpoints for OAuth2 flow and Tastytrade API access:

- **Authentication Endpoints:**
  - `GET /api/tastytrade/auth/status` - Check authentication status
  - `GET /api/tastytrade/auth/url` - Get OAuth authorization URL
  - `GET /api/tastytrade/auth/callback` - Handle OAuth callback
  - `POST /api/tastytrade/auth/refresh` - Refresh session token
  - `POST /api/tastytrade/auth/logout` - Logout and clear tokens

- **Account Endpoints:**
  - `GET /api/tastytrade/accounts` - Get all accounts
  - `GET /api/tastytrade/accounts/{account_number}/balance` - Get balance
  - `GET /api/tastytrade/accounts/{account_number}/positions` - Get positions
  - `GET /api/tastytrade/accounts/{account_number}/summary` - Get full summary

- **Market Data Endpoints:**
  - `GET /api/tastytrade/market/equity/{symbol}` - Get equity data
  - `GET /api/tastytrade/market/option-chain/{symbol}` - Get option chains

- **Order Endpoints:**
  - `POST /api/tastytrade/orders/dry-run` - Test order
  - `POST /api/tastytrade/orders/place` - Place order
  - `GET /api/tastytrade/orders/{account_number}` - List orders

### Frontend Components

1. **Authentication Utility** (`frontend/utils/tastytrade_auth.py`)
   - `TastytradeAuthManager` class for managing authentication
   - Session state management
   - API communication helpers
   - Authentication requirements decorator

2. **Authentication Page** (`frontend/pages/tastytrade_auth.py`)
   - Standalone authentication page with full UI
   - Account information display
   - Position and balance viewing
   - Logout functionality

3. **Integrated Navigation** (`frontend/app.py`)
   - New "Tastytrade" tab in main navigation
   - Inline authentication and account viewing
   - Seamless integration with existing dashboard

---

## 🚀 How to Use

### Step 1: Start the Backend API

The FastAPI backend needs to be running to handle authentication:

```bash
cd /Users/saleemjadallah/Desktop/OptionsTrader
uvicorn backend.api.endpoints:app --reload --port 8000
```

The API will be available at: http://localhost:8000

### Step 2: Start the Streamlit Frontend

In a new terminal:

```bash
cd /Users/saleemjadallah/Desktop/OptionsTrader/frontend
streamlit run app.py
```

The dashboard will open at: http://localhost:8501

### Step 3: Authenticate

1. Click on the **"Tastytrade"** tab in the navigation menu
2. Click **"Connect to Tastytrade"** button
3. Click the authentication link that appears
4. Log in with your credentials:
   - Username: `tastyslim`
   - Password: `saleemjadallah1986`
5. Authorize the application
6. Return to the dashboard and click **"Refresh Status"**

---

## 📱 User Interface

### Not Authenticated View

When you first open the Tastytrade tab:

```
🔐 Tastytrade Integration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔒 Not connected to Tastytrade

ℹ️ Connect your Tastytrade account to:
  - View real-time account balances
  - Access current positions
  - Place and manage orders
  - Stream live market data

  Click the button below to securely authenticate using OAuth2.

┌────────────────────────────────────┐
│  🔐 Connect to Tastytrade          │
└────────────────────────────────────┘
```

### Authenticated View

After successful authentication:

```
🔐 Tastytrade Integration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Connected to Tastytrade

┌─────────────┬─────────────┬─────────────┐
│  Status     │ Environment │  Account    │
│  Connected  │  SANDBOX    │  5WV12345   │
└─────────────┴─────────────┴─────────────┘

┌─────────────────────────────────────┐
│ 📊 Account Info │ 💼 Positions │ 📈 Balance │
└─────────────────────────────────────┘

  [Account details, positions, and balance displayed here]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌────────────────────────────────────┐
│  🚪 Logout from Tastytrade         │
└────────────────────────────────────┘
```

---

## 🔧 Configuration

### API Base URL

The frontend communicates with the backend API. Default URL is `http://localhost:8000`.

To change the API URL, modify the initialization in your code:

```python
auth_manager = get_auth_manager(api_base_url="http://your-api:8000")
```

### Environment Variables

Ensure your `.env` file has the correct settings:

```bash
# Tastytrade OAuth2 Configuration
TASTYTRADE_CLIENT_ID=87a97b47-1714-4ed5-80e2-9b362e270fb3
TASTYTRADE_CLIENT_SECRET=2f1857deda8d116a16fd090bd2e8321a89bef172
TASTYTRADE_REDIRECT_URI=http://localhost:8080/callback
TASTYTRADE_SCOPES=read,trade,openid
TASTYTRADE_SANDBOX=True

# FastAPI Backend
API_BASE_URL=http://localhost:8000
```

---

## 🎯 Features

### 1. Authentication Status Check

Automatically checks authentication status when you open the Tastytrade tab:

```python
# In your code
auth_manager = get_auth_manager()
status = auth_manager.check_auth_status()

if status['token_valid']:
    # User is authenticated
    pass
```

### 2. Account Information

View all connected accounts with details:

```python
accounts = auth_manager.get_accounts()
# Returns list of account dictionaries
```

### 3. Real-time Balance

Get current account balance:

```python
balance = auth_manager.get_account_balance()
# Returns balance dictionary with cash, buying power, etc.
```

### 4. Position Tracking

View current positions:

```python
positions = auth_manager.get_account_positions()
# Returns list of position dictionaries
```

### 5. Logout

Clear authentication and tokens:

```python
auth_manager.logout()
# Clears all tokens and session state
```

---

## 💻 API Usage Examples

### Check Authentication

```python
import requests

response = requests.get("http://localhost:8000/api/tastytrade/auth/status")
status = response.json()

if status['token_valid']:
    print(f"Authenticated! Account: {status['account_number']}")
```

### Get Account Balance

```python
import requests

response = requests.get(
    "http://localhost:8000/api/tastytrade/accounts/5WV12345/balance"
)
balance = response.json()

print(f"Cash: ${balance['balance']['cash-balance']:,.2f}")
print(f"Buying Power: ${balance['balance']['equity-buying-power']:,.2f}")
```

### Place Dry-Run Order

```python
import requests

order = {
    "time-in-force": "Day",
    "order-type": "Limit",
    "price": 150.00,
    "legs": [{
        "instrument-type": "Equity",
        "symbol": "AAPL",
        "action": "Buy to Open",
        "quantity": 10
    }]
}

response = requests.post(
    "http://localhost:8000/api/tastytrade/orders/dry-run",
    json=order
)
result = response.json()

print(f"Buying power impact: ${result['dry_run_result']['buying-power-effect']}")
```

---

## 🔐 Security

### Token Storage

- Session tokens stored in memory (Streamlit session_state)
- Backend stores tokens in `backend/credentials/tokens.json` (gitignored)
- Tokens automatically refreshed when expired

### OAuth2 Flow

1. User clicks "Connect"
2. Frontend requests auth URL from backend
3. User redirected to Tastytrade login
4. Tastytrade redirects to callback URL
5. Backend exchanges code for tokens
6. Tokens saved and user redirected back

### HTTPS Requirement

For production, use HTTPS for all API calls:

```python
auth_manager = get_auth_manager(api_base_url="https://api.yourdomain.com")
```

---

## 🐛 Troubleshooting

### Backend Not Connected

**Problem:** Frontend shows "Failed to check authentication status"

**Solutions:**
1. Ensure FastAPI is running: `uvicorn backend.api.endpoints:app --port 8000`
2. Check API URL is correct: `http://localhost:8000`
3. Verify no firewall blocking port 8000

### Authentication Fails

**Problem:** Can't complete OAuth flow

**Solutions:**
1. Check credentials in `.env` are correct
2. Ensure redirect URI matches: `http://localhost:8080/callback`
3. Verify OAuth app is active in Tastytrade dashboard
4. Check backend logs for errors

### Token Expired

**Problem:** Getting 401 errors after some time

**Solutions:**
1. Tokens auto-refresh - wait a moment and try again
2. Manually refresh: click "Refresh Status" button
3. If persistent, logout and re-authenticate

### CORS Errors

**Problem:** Browser shows CORS policy errors

**Solutions:**
1. Ensure FastAPI CORS middleware is configured correctly
2. Check allowed origins include Streamlit URL
3. Verify both frontend and backend are running

---

## 📊 Integration with Trading Bot

### Use Authenticated Account Data

```python
from utils.tastytrade_auth import get_auth_manager

# Get auth manager
auth_manager = get_auth_manager()

# Require authentication
if auth_manager.require_authentication():
    # Get real account data
    balance = auth_manager.get_account_balance()
    positions = auth_manager.get_account_positions()

    # Use in your trading logic
    available_cash = balance['cash-balance']
    current_positions = len(positions)
```

### Protected Pages

```python
def render_trading_page():
    auth_manager = get_auth_manager()

    # Require authentication
    if not auth_manager.require_authentication():
        return  # Shows auth prompt automatically

    # Continue with authenticated content
    st.title("Live Trading Dashboard")
    # ... your trading logic ...
```

---

## 🚀 Next Steps

### 1. Test Authentication

```bash
# Start backend
uvicorn backend.api.endpoints:app --port 8000

# Start frontend
streamlit run frontend/app.py

# Navigate to Tastytrade tab and authenticate
```

### 2. Integrate with Trading Strategies

Update your trading strategies to use real Tastytrade data:

```python
# In your strategy code
from backend.api.tastytrade_client import TastytradeClient

client = TastytradeClient()
positions = client.get_positions()

# Use real positions in your strategy
for position in positions:
    # Analyze and trade based on real data
    pass
```

### 3. Enable Live Trading

Once tested in sandbox, switch to production:

```bash
# Update .env
TASTYTRADE_SANDBOX=False

# Restart backend and re-authenticate
```

---

## 📞 Support

- **API Documentation**: See `TASTYTRADE_API_DOCUMENTATION.md`
- **Setup Guide**: See `TASTYTRADE_SETUP_GUIDE.md`
- **Backend Code**: `backend/api/tastytrade_endpoints.py`
- **Frontend Code**: `frontend/utils/tastytrade_auth.py`
- **Auth Page**: `frontend/pages/tastytrade_auth.py`

---

## ✅ Quick Checklist

- [x] Backend API endpoints created
- [x] Frontend auth utilities created
- [x] Authentication page added
- [x] Navigation updated with Tastytrade tab
- [x] OAuth2 flow integrated
- [ ] **Start backend API**: `uvicorn backend.api.endpoints:app --port 8000`
- [ ] **Start frontend**: `streamlit run frontend/app.py`
- [ ] **Authenticate**: Click Tastytrade tab → Connect → Login
- [ ] **Test features**: View accounts, balances, positions

---

**You're ready to go! Start the backend and frontend, then authenticate through the app! 🎉**
