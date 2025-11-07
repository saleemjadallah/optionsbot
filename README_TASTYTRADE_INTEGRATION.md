# Tastytrade Integration - Quick Start

Welcome to your Tastytrade API integration! Everything is configured and ready to use.

---

## 📁 Files Created

### Configuration
- ✅ `backend/.env` - Your credentials (secured, not in git)
- ✅ `backend/.env.example` - Template for others
- ✅ `backend/config/tastytrade_config.py` - Configuration module

### Authentication
- ✅ `backend/api/tastytrade_auth.py` - OAuth2 handler
- ✅ `backend/api/oauth_callback_server.py` - OAuth callback server
- ✅ `backend/api/tastytrade_client.py` - High-level API client

### Documentation
- ✅ `TASTYTRADE_API_DOCUMENTATION.md` - Complete API reference
- ✅ `TASTYTRADE_SETUP_GUIDE.md` - Detailed setup instructions
- ✅ `README_TASTYTRADE_INTEGRATION.md` - This file

---

## 🚀 Quick Start (3 Steps)

### Step 1: Authenticate

Run the OAuth authentication flow:

```bash
cd /Users/saleemjadallah/Desktop/OptionsTrader
python backend/api/oauth_callback_server.py
```

This will:
1. Open your browser to Tastytrade login
2. Handle the OAuth callback automatically
3. Save your tokens for future use

**Login credentials:**
- Username: `tastyslim`
- Password: `saleemjadallah1986`

### Step 2: Test Connection

After authentication, test your connection:

```bash
python backend/api/tastytrade_client.py
```

You should see your account information and balances.

### Step 3: Start Building

Use the high-level client in your code:

```python
from backend.api.tastytrade_client import TastytradeClient

# Initialize client (auto-loads saved tokens)
client = TastytradeClient()

# Get accounts
accounts = client.get_accounts()
print(f"Found {len(accounts)} accounts")

# Get positions
positions = client.get_positions()
print(f"Current positions: {len(positions)}")

# Get account balances
balances = client.get_account_balances()
print(f"Cash: ${balances['cash-balance']:,.2f}")
```

---

## 📚 What You Can Do Now

### Account Management
```python
client = TastytradeClient()

# Get account summary
summary = client.get_account_summary()

# Get positions
positions = client.get_positions()

# Get transactions
transactions = client.get_transactions(
    start_date='2025-01-01',
    end_date='2025-01-06'
)
```

### Market Data
```python
# Get stock information
equity = client.get_equity('AAPL')

# Get option chain
options = client.get_option_chain('SPY', nested=True)

# Get futures
futures = client.get_futures(product_code='ES')
```

### Order Management
```python
# Create an order
order = {
    'time-in-force': 'Day',
    'order-type': 'Limit',
    'price': 150.00,
    'legs': [{
        'instrument-type': 'Equity',
        'symbol': 'AAPL',
        'action': 'Buy to Open',
        'quantity': 10
    }]
}

# Test the order (dry run)
dry_run = client.dry_run_order(order_data=order)
print(f"Buying power impact: ${dry_run['buying-power-effect']['change-in-buying-power']}")

# Place the order (if dry run looks good)
# result = client.place_order(order_data=order)
```

### Watchlists
```python
# Create a watchlist
watchlist = client.create_watchlist(
    name='Tech Stocks',
    symbols=['AAPL', 'MSFT', 'GOOGL', 'NVDA']
)

# Get all watchlists
watchlists = client.get_watchlists()
```

---

## 🔐 Security & Credentials

### Your OAuth2 Application

```
Name: tastyslim Personal OAuth2 App
Scopes: read, trade, openid
Environment: Sandbox
```

### Credentials Location

```
backend/.env                    # Your credentials (NEVER commit)
backend/credentials/tokens.json # Auto-generated tokens (NEVER commit)
```

Both files are already in `.gitignore` and protected.

### Token Management

- **Session Token**: Auto-refreshes every 15 minutes
- **Refresh Token**: Never expires
- **Quote Token**: Auto-refreshes every 24 hours

The client handles all token management automatically!

---

## 📖 Documentation

### Comprehensive API Reference
See `TASTYTRADE_API_DOCUMENTATION.md` for:
- All available endpoints
- Request/response formats
- Symbol conventions
- Streaming setup (DXLink)
- Error handling
- Best practices

### Detailed Setup Guide
See `TASTYTRADE_SETUP_GUIDE.md` for:
- Step-by-step authentication
- Troubleshooting
- Configuration options
- Testing procedures

---

## 🔧 Configuration

Your environment is configured in `backend/.env`:

```bash
# OAuth2 Settings
TASTYTRADE_CLIENT_ID=87a97b47-1714-4ed5-80e2-9b362e270fb3
TASTYTRADE_CLIENT_SECRET=2f1857deda8d116a16fd090bd2e8321a89bef172
TASTYTRADE_REDIRECT_URI=http://localhost:8080/callback
TASTYTRADE_SCOPES=read,trade,openid

# Environment
TASTYTRADE_SANDBOX=True  # Set to False for production

# Account
TASTYTRADE_USERNAME=tastyslim
TASTYTRADE_EMAIL=saleem86@gmail.com
```

---

## 🎯 Integration Examples

### Example 1: Get Current Portfolio Value

```python
from backend.api.tastytrade_client import TastytradeClient

client = TastytradeClient()
balances = client.get_account_balances()

portfolio_value = balances['net-liquidating-value']
cash_balance = balances['cash-balance']
equity_value = portfolio_value - cash_balance

print(f"Portfolio Value: ${portfolio_value:,.2f}")
print(f"Cash: ${cash_balance:,.2f}")
print(f"Equity: ${equity_value:,.2f}")
```

### Example 2: Monitor All Positions

```python
from backend.api.tastytrade_client import TastytradeClient

client = TastytradeClient()
positions = client.get_positions()

for pos in positions:
    symbol = pos['symbol']
    qty = pos['quantity']
    avg_price = pos.get('average-open-price', 0)
    current_price = pos.get('mark-price', 0)
    pnl = pos.get('unrealized-gain-loss', 0)

    print(f"{symbol}: {qty} @ ${avg_price:.2f} | P&L: ${pnl:.2f}")
```

### Example 3: Scan Option Chains

```python
from backend.api.tastytrade_client import TastytradeClient

client = TastytradeClient()

# Get option chain for SPY
chain = client.get_option_chain('SPY', nested=True)

# Access expirations
expirations = chain.get('items', [])

print(f"Found {len(expirations)} expiration dates for SPY")

for exp in expirations[:3]:  # First 3 expirations
    exp_date = exp['expiration-date']
    strikes = exp.get('strikes', [])
    print(f"\n{exp_date}: {len(strikes)} strikes available")
```

### Example 4: Place a Protective Put

```python
from backend.api.tastytrade_client import TastytradeClient

client = TastytradeClient()

# Define protective put order
order = {
    'time-in-force': 'Day',
    'order-type': 'Market',
    'legs': [
        {
            'instrument-type': 'Equity Option',
            'symbol': 'SPY   250131P00550000',  # SPY Jan 31 550 Put
            'action': 'Buy to Open',
            'quantity': 1
        }
    ]
}

# Test first
dry_run = client.dry_run_order(order_data=order)
print(f"Order impact: {dry_run}")

# If acceptable, place order
# result = client.place_order(order_data=order)
```

---

## 🎓 Next Steps

### 1. Learn the API
- Read `TASTYTRADE_API_DOCUMENTATION.md`
- Understand symbol formats (equities, options, futures)
- Study order types and structures

### 2. Test in Sandbox
- Practice order placement
- Test your strategies
- Monitor positions and P&L

### 3. Implement Features
- Market data streaming (DXLink WebSocket)
- Real-time position monitoring
- Automated order execution
- Risk management rules

### 4. Production Readiness
- Test thoroughly in sandbox
- Implement error handling
- Add logging and monitoring
- Create production OAuth app
- Switch `TASTYTRADE_SANDBOX=False`

---

## 🆘 Troubleshooting

### Token Issues
```bash
# Clear tokens and re-authenticate
rm backend/credentials/tokens.json
python backend/api/oauth_callback_server.py
```

### Configuration Check
```bash
# Validate configuration
python backend/config/tastytrade_config.py
```

### Test Authentication
```bash
# Test auth status
python backend/api/tastytrade_auth.py
```

---

## 📞 Support Resources

- **API Documentation**: https://developer.tastytrade.com/
- **Support Portal**: https://support.tastytrade.com/
- **Your Files**: All documentation in your project directory

---

## ✅ Checklist

- [x] OAuth2 app created
- [x] Credentials configured in `.env`
- [x] Configuration module created
- [x] Authentication handler created
- [x] Callback server created
- [x] High-level client created
- [x] Documentation completed
- [ ] **Run authentication**: `python backend/api/oauth_callback_server.py`
- [ ] **Test connection**: `python backend/api/tastytrade_client.py`
- [ ] **Start building**: Integrate with your trading app

---

## 🎉 You're Ready!

Everything is set up and ready to go. Just run the authentication script and start building your trading application!

```bash
python backend/api/oauth_callback_server.py
```

**Happy Trading! 📈**
