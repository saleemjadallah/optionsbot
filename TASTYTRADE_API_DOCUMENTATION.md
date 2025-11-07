# Tastytrade API Comprehensive Documentation

> **Version:** 2025
> **Last Updated:** January 2025
> **Source:** https://developer.tastytrade.com/

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Authentication & Sessions](#authentication--sessions)
4. [API Conventions](#api-conventions)
5. [Instruments & Symbology](#instruments--symbology)
6. [Account Management](#account-management)
7. [Market Data](#market-data)
8. [Streaming Data (DXLink)](#streaming-data-dxlink)
9. [Order Management](#order-management)
10. [Transactions](#transactions)
11. [Watchlists](#watchlists)
12. [Error Handling](#error-handling)
13. [Best Practices](#best-practices)

---

## Overview

The Tastytrade API is a comprehensive REST/JSON API providing access to:

- **Market Data**: Real-time and historical quotes via DXLink streaming
- **Order Execution**: Equity, options, futures, and cryptocurrency trading
- **Portfolio Management**: Positions, balances, and account analytics
- **Risk Analysis**: Margin requirements and options Greeks

**Supported Asset Classes:**
- Equities (stocks and ETFs)
- Equity Options (American-style)
- Futures
- Future Options
- Cryptocurrencies

**Regulatory Information:**
- FINRA, SIPC, and NFA regulated
- Cryptocurrency services powered by Zero Hash LLC

---

## Getting Started

### Sandbox Environment

Tastytrade provides a sandbox environment for testing:

1. Create a sandbox account at https://developer.tastytrade.com/sandbox/
2. Use sandbox credentials for development
3. Test all workflows before production deployment

### Base URL

```
Production: https://api.tastytrade.com
Sandbox: https://api.cert.tastytrade.com
```

### Required Headers

All requests must include:

```http
User-Agent: <product>/<version>
Content-Type: application/json
Accept: application/json
Authorization: Bearer <session-token>
```

**Example:**
```http
User-Agent: my-trading-app/1.0.0
Content-Type: application/json
Accept: application/json
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Authentication & Sessions

### Important: OAuth2 Migration (2025)

**Session-token authentication will be discontinued on December 1, 2025.**

Tastytrade is migrating to OAuth2.0 as the standard authentication method.

### OAuth2 Authentication (Recommended)

OAuth2 provides secure, long-lived authentication with refresh tokens.

**Benefits:**
- Refresh tokens never expire
- Session tokens valid for 15 minutes
- Can be refreshed indefinitely

**OAuth2 Flow:**

1. **Create OAuth Application** at developer portal
2. **Obtain Refresh Token** using client credentials
3. **Exchange for Session Token** via API
4. **Refresh as Needed** (every 15 minutes)

**Endpoint:**
```
POST /oauth/token
```

**OAuth2 Documentation:**
https://developer.tastytrade.com/api-guides/oauth/

### Legacy Session Authentication (Deprecated)

**Endpoint:**
```
POST /sessions
```

**Request Body:**
```json
{
  "login": "username_or_email",
  "password": "password",
  "remember-me": true
}
```

**Response:**
```json
{
  "data": {
    "session-token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
    "remember-token": "abc123...",
    "user": {
      "email": "user@example.com",
      "username": "trader123"
    }
  }
}
```

**Token Expiration:**
- Session tokens expire after 15 minutes
- Must be refreshed or re-authenticated

**Using Session Token:**
```http
Authorization: Bearer <session-token>
```

---

## API Conventions

### Request/Response Format

**JSON Conventions:**
- All keys use dasherized format (e.g., `account-number`, `order-type`)
- Responses contain a `data` object and `context` key
- Multi-object responses include an `items` array within `data`

**GET Request Parameters:**
```
GET /endpoint?my-key[]=value1&my-key[]=value2
```

**POST/PUT/PATCH/DELETE:**
```json
{
  "account-number": "12345",
  "order-type": "Limit"
}
```

### Response Structure

**Single Object:**
```json
{
  "data": {
    "id": "123",
    "account-number": "12345"
  },
  "context": "/accounts/12345"
}
```

**Multiple Objects:**
```json
{
  "data": {
    "items": [
      { "id": "1" },
      { "id": "2" }
    ]
  },
  "context": "/accounts/12345/positions",
  "pagination": {
    "page-offset": 0,
    "per-page": 250,
    "total-items": 42
  }
}
```

### Pagination

**Default Parameters:**
- `page-offset`: 0
- `per-page`: 250 (range: 1-2000)

**Example:**
```
GET /accounts/12345/transactions?page-offset=0&per-page=100
```

### Rate Limits

- HTTP 429 (Too Many Requests) returned when thresholds exceeded
- Specific limits not publicly documented
- Implement exponential backoff for retries

---

## Instruments & Symbology

### Instrument Types

The API supports five tradeable asset classes with specific symbol formats.

#### 1. Equities

**Format:** Alphanumeric (A-Z, 0-9) with optional forward slash

**Examples:**
- `AAPL` - Apple Inc.
- `TSLA` - Tesla
- `BRK/A` - Berkshire Hathaway Class A

**URL Encoding:** Not required for most tickers

#### 2. Equity Options

**Format:** OCC Standard (Options Clearing Corporation)

```
[ROOT][YYMMDD][P/C][STRIKE]
```

**Components:**
- **ROOT**: 6-character underlying symbol (padded with spaces)
- **YYMMDD**: Expiration date
- **P/C**: Option type (Put or Call)
- **STRIKE**: 8-digit strike price (with 3 decimal places)

**Examples:**
- `AAPL  220617P00150000` - AAPL Put, exp 6/17/22, strike $150
- `SPY   230120C00400000` - SPY Call, exp 1/20/23, strike $400

#### 3. Futures

**Format:** Slash prefix + product code + month code + year

```
/[PRODUCT][MONTH][YEAR]
```

**Month Codes:**
- F=January, G=February, H=March, J=April, K=May, M=June
- N=July, Q=August, U=September, V=October, X=November, Z=December

**Examples:**
- `/ESZ2` - E-mini S&P 500, December 2022
- `/6AM23` - Australian Dollar, June 2023

**URL Encoding:** Required - encode `/` as `%2F`
```
/ESZ2 → %2FESZ2
```

#### 4. Future Options

**Format:** `./` prefix with future contract and option codes

**Examples:**
- `./CLZ2 LO1X2 221104C91` - Crude Oil option
- `./ESZ2 EW4X2 221216P4000`

**URL Encoding:** Encode `./` appropriately

#### 5. Cryptocurrencies

**Format:** Currency pair with forward slash

**Examples:**
- `BTC/USD` - Bitcoin
- `ETH/USD` - Ethereum
- `BCH/USD` - Bitcoin Cash

**URL Encoding:** Required - encode `/` as `%2F`
```
BTC/USD → BTC%2FUSD
```

### Streaming Symbol Format

For DXLink streaming, use the `streamer-symbol` field from instrument endpoints.

**Examples:**
- Futures: `/6AM23:XCME`
- Crypto: `BTC/USD:CXTALP`
- Equities: `AAPL`

### Instrument Endpoints

#### Get Equity Instrument
```
GET /instruments/equities/{symbol}
```

**Query Parameters:**
- `lendability`: Filter by borrow availability
- `is-index`: Filter for index instruments
- `is-etf`: Filter for ETFs

#### Get Option Chains

**Nested Format** (grouped by expiration):
```
GET /option-chains/{symbol}/nested
```

**Detailed Format** (full instrument objects):
```
GET /option-chains/{symbol}
```

**Compact Format** (symbol array):
```
GET /option-chains/{symbol}/compact
```

**Query Parameters:**
- `with-expired`: Include expired options (default: false)

#### Get Futures
```
GET /instruments/futures
```

**Query Parameters:**
- `product-code`: Filter by product family (e.g., "ES", "NQ")

#### Get Future Option Chains
```
GET /futures-option-chains/{product_code}/nested
```

#### Get Cryptocurrencies
```
GET /instruments/cryptocurrencies
```

**Query Parameters:**
- `active`: Filter for tradeable instruments

---

## Account Management

### Customer Placeholder

Use `me` as a placeholder for customer ID across all endpoints. This prevents exposing internal customer IDs.

**Example:**
```
GET /customers/me/accounts
```

### Get Customer Accounts

**Endpoint:**
```
GET /customers/me/accounts
```

**Response:**
```json
{
  "data": {
    "items": [
      {
        "account-number": "12345678",
        "authority-level": "owner",
        "nickname": "Individual",
        "account-type": "Cash"
      }
    ]
  }
}
```

**Authority Levels:**
- **owner**: Full access to all operations
- **trade-only**: Trading operations only
- **read-only**: GET endpoints only

### Get Account Balances

**Endpoint:**
```
GET /accounts/{account_number}/balances
```

**Note:** This endpoint is deprecated. Use balance-snapshots instead.

**Endpoint (Current):**
```
GET /accounts/{account_number}/balances/{currency}
```

**Parameters:**
- `currency`: USD, EUR, etc.

**Response Fields (60+ fields including):**
- `cash-balance`: Available cash
- `equity-buying-power`: Equity purchasing power
- `net-liquidating-value`: Total account value
- `maintenance-requirement`: Margin maintenance
- `margin-equity`: Equity used for margin
- `derivative-buying-power`: Options buying power
- `futures-margin-requirement`: Futures margin
- `cryptocurrency-buying-power`: Crypto purchasing power

### Get Balance Snapshots

**Endpoint:**
```
GET /accounts/{account_number}/balance-snapshots
```

**Query Parameters:**
- `currency`: Default USD
- `snapshot-date`: Specific date
- `time-of-day`: BOD (beginning of day) or EOD (end of day)
- `start-date`: Filter range start
- `end-date`: Filter range end
- `page-offset`: Pagination offset (default: 0)
- `per-page`: Items per page (default: 250, max: 2000)

**Response:**
```json
{
  "data": {
    "items": [
      {
        "account-number": "12345678",
        "snapshot-date": "2025-01-06",
        "cash-balance": 50000.00,
        "net-liquidating-value": 75000.00
      }
    ]
  }
}
```

### Get Account Positions

**Endpoint:**
```
GET /accounts/{account_number}/positions
```

**Query Parameters:**
- `include-closed-positions`: Default false
- `include-marks`: Include mark prices (default: false)
- `instrument-type`: Filter by type (Equity, Equity Option, Future, etc.)
- `net-positions`: Combine long/short (default: false)
- `symbol`: Filter by specific symbol
- `underlying-symbol`: Filter by underlying
- `underlying-product-code`: Filter futures by product

**Response:**
```json
{
  "data": {
    "items": [
      {
        "symbol": "AAPL",
        "instrument-type": "Equity",
        "quantity": 100,
        "quantity-direction": "Long",
        "average-open-price": 150.00,
        "mark-price": 155.00,
        "unrealized-gain-loss": 500.00,
        "realized-gain-loss": 0.00
      }
    ]
  }
}
```

**Important Notes:**
- Positions with quantity 0 are considered closed
- Closed positions are purged overnight
- Mark prices are deprecated; use live DXLink quotes for P/L calculations

---

## Market Data

### Quote Token (Required for Streaming)

**Endpoint:**
```
GET /api-quote-tokens
```

**Response:**
```json
{
  "data": {
    "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
    "dxlink-url": "wss://tasty-openapi-ws.dxfeed.com/realtime",
    "level": "api"
  }
}
```

**Important:** API quote tokens expire after 24 hours.

### Market Data Access

- Must be a registered Tastytrade customer for quote access
- Real-time streaming via DXLink WebSocket protocol
- Historical data available through candle events

---

## Streaming Data (DXLink)

### Overview

Tastytrade provides asynchronous streaming quote data via WebSocket using the DXLink protocol.

**Base URL:**
```
wss://tasty-openapi-ws.dxfeed.com/realtime
```

### Connection Workflow

1. **Obtain API Quote Token** via `GET /api-quote-tokens`
2. **Connect to WebSocket** using `dxlink-url`
3. **Send Messages in Sequence**:
   - SETUP
   - AUTHORIZE
   - CHANNEL_REQUEST
   - FEED_SETUP
   - FEED_SUBSCRIPTION
   - KEEPALIVE (every 30 seconds)

### Message Types

#### 1. SETUP

Initialize connection with version and keepalive timeout.

```json
{
  "type": "SETUP",
  "channel": 0,
  "version": "0.1-DXF-JS/0.3.0",
  "keepaliveTimeout": 60,
  "acceptKeepaliveTimeout": 60
}
```

#### 2. AUTHORIZE

Authenticate with API token after receiving `AUTH_STATE: UNAUTHORIZED`.

```json
{
  "type": "AUTH",
  "channel": 0,
  "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### 3. CHANNEL_REQUEST

Open a virtual data channel (e.g., channel 3).

```json
{
  "type": "CHANNEL_REQUEST",
  "channel": 3,
  "service": "FEED",
  "parameters": {
    "contract": "AUTO"
  }
}
```

#### 4. FEED_SETUP

Configure which data fields to receive per event type.

```json
{
  "type": "FEED_SETUP",
  "channel": 3,
  "acceptAggregationPeriod": 10,
  "acceptDataFormat": "COMPACT",
  "acceptEventFields": {
    "Quote": ["eventSymbol", "bidPrice", "askPrice", "bidSize", "askSize"],
    "Trade": ["eventSymbol", "price", "size", "volume"]
  }
}
```

#### 5. FEED_SUBSCRIPTION

Subscribe to specific symbols and event types.

```json
{
  "type": "FEED_SUBSCRIPTION",
  "channel": 3,
  "add": [
    {
      "symbol": "AAPL",
      "type": "Quote"
    },
    {
      "symbol": "AAPL",
      "type": "Trade"
    }
  ]
}
```

#### 6. KEEPALIVE

Maintain connection every 30 seconds.

```json
{
  "type": "KEEPALIVE",
  "channel": 0
}
```

### Event Types

#### Trade
Real-time price, volume, and size data.

**Fields:**
- `eventSymbol`, `price`, `size`, `volume`, `time`

#### TradeETH
Extended hours trading data.

#### Quote
Bid/ask prices and sizes.

**Fields:**
- `eventSymbol`, `bidPrice`, `askPrice`, `bidSize`, `askSize`, `time`

#### Greeks
Options volatility metrics.

**Fields:**
- `delta`, `gamma`, `theta`, `rho`, `vega`, `volatility`

#### Profile
Security descriptions and trading status.

#### Summary
Daily OHLC and open interest.

**Fields:**
- `open`, `high`, `low`, `close`, `volume`, `openInterest`

#### Candle
Aggregated historical data across time intervals.

### Candle Events (Historical Data)

Subscribe to aggregated price data using special symbol syntax.

**Format:**
```
{symbol}{=<period><type>}
```

**Examples:**
- `AAPL{=1m}` - 1-minute candles
- `SPY{=5m}` - 5-minute candles
- `TSLA{=1h}` - 1-hour candles
- `QQQ{=1d}` - 1-day candles

**Time Units:**
- `m` - minutes
- `h` - hours
- `d` - days

**Parameters:**
- `period`: Multiplier (e.g., 5 for 5-minute)
- `type`: Time unit
- `fromTime`: Unix epoch timestamp for start point

**Recommended Data Ranges:**

| Time Back | Interval | Example |
|-----------|----------|---------|
| 1 day | 1 minute | `AAPL{=1m}` |
| 1 week | 5 minutes | `AAPL{=5m}` |
| 1 month | 30 minutes | `AAPL{=30m}` |
| 3 months | 1 hour | `AAPL{=1h}` |
| 6 months | 2 hours | `AAPL{=2h}` |
| 1 year+ | 1 day | `AAPL{=1d}` |

**Important:** The final candle received is always the "live" current period, which updates continuously.

### Streaming Best Practices

1. **Multiple Channels**: Run separate streams for different asset types
2. **Selective Fields**: Configure only required data fields to reduce bandwidth
3. **Compact Format**: Use compact data format for efficiency
4. **Keepalive Required**: Send keepalive every 30 seconds to maintain connection
5. **Reconnection Logic**: Implement exponential backoff for reconnections

### Testing Tools

- **DXLink Protocol Debugger**: demo.dxfeed.com
- **WebSocket Client**: Browser extensions for testing
- **Sample Configs**: Download from Tastytrade documentation

---

## Order Management

### Order Placement

#### Dry-Run Orders

Test orders without execution to calculate effects on buying power and fees.

**Endpoint:**
```
POST /accounts/{account_number}/orders/dry-run
```

**Purpose:**
- Calculate order impact on account
- Validate order parameters
- Preview commissions and fees
- Provide order confirmation screen data

**Important:** If a dry-run returns warnings and you attempt to submit the order, it will be rejected.

#### Submit Order

**Endpoint:**
```
POST /accounts/{account_number}/orders
```

**Response:**
Returns order ID for tracking and management.

### Order Types

The API supports various order types for different asset classes:

- **Market**: Execute at current market price
- **Limit**: Execute at specified price or better
- **Stop**: Trigger market order when stop price reached
- **Stop Limit**: Trigger limit order when stop price reached

**Complex Orders:**
- **OCO** (One-Cancels-Other): Two orders where execution of one cancels the other
- **OTOCO** (One-Triggers-OCO): Parent order triggers OCO pair

### Time-In-Force

Order duration specifications:

- **Day**: Valid for current trading day
- **GTC** (Good-Till-Canceled): Valid until filled or canceled
- **GTD** (Good-Till-Date): Valid until specific date
- **IOC** (Immediate-or-Cancel): Fill immediately or cancel
- **FOK** (Fill-or-Kill): Fill entirely or cancel

### Order Structure

**Basic Equity Order:**
```json
{
  "time-in-force": "Day",
  "order-type": "Limit",
  "price": 150.50,
  "legs": [
    {
      "instrument-type": "Equity",
      "symbol": "AAPL",
      "action": "Buy to Open",
      "quantity": 100
    }
  ]
}
```

**Option Order:**
```json
{
  "time-in-force": "Day",
  "order-type": "Limit",
  "price": 2.50,
  "legs": [
    {
      "instrument-type": "Equity Option",
      "symbol": "AAPL  250117C00150000",
      "action": "Buy to Open",
      "quantity": 1
    }
  ]
}
```

**Multi-Leg Option Strategy:**
```json
{
  "time-in-force": "Day",
  "order-type": "Limit",
  "price": 1.00,
  "legs": [
    {
      "instrument-type": "Equity Option",
      "symbol": "SPY   250117C00400000",
      "action": "Buy to Open",
      "quantity": 1
    },
    {
      "instrument-type": "Equity Option",
      "symbol": "SPY   250117C00410000",
      "action": "Sell to Open",
      "quantity": 1
    }
  ]
}
```

### Order Actions

- **Buy to Open**: Open long position
- **Buy to Close**: Close short position
- **Sell to Open**: Open short position
- **Sell to Close**: Close long position
- **Buy**: General buy (equities)
- **Sell**: General sell (equities)

### Order Management

#### Get Orders
```
GET /accounts/{account_number}/orders
```

**Query Parameters:**
- `status`: Filter by status (Received, Routed, Filled, Cancelled, etc.)
- `start-date`, `end-date`: Filter by date range

#### Get Order by ID
```
GET /accounts/{account_number}/orders/{order_id}
```

#### Cancel Order
```
DELETE /accounts/{account_number}/orders/{order_id}
```

#### Modify Order
```
PATCH /accounts/{account_number}/orders/{order_id}
```

### Order Status Values

- **Received**: Order received by system
- **Routed**: Order sent to exchange
- **In Flight**: Order being processed
- **Live**: Order active on exchange
- **Filled**: Order completely executed
- **Partially Filled**: Order partially executed
- **Cancelled**: Order cancelled
- **Rejected**: Order rejected by exchange
- **Expired**: Order expired

---

## Transactions

### Get Account Transactions

**Endpoint:**
```
GET /accounts/{account_number}/transactions
```

**Query Parameters:**

**Pagination:**
- `page-offset`: Default 0
- `per-page`: Default 250 (range: 1-2000)
- `sort`: Desc (default) or Asc

**Filtering:**
- `currency`: USD, EUR, etc.
- `type`: Transaction type
- `types`: Multiple types (array)
- `sub-type`: Sub-type filter (array)
- `action`: Allocate, Buy, Buy to Close, Buy to Open, Sell, Sell to Close, Sell to Open
- `symbol`: Specific symbol
- `underlying-symbol`: Filter by underlying
- `futures-symbol`: Futures contract symbol
- `instrument-type`: Bond, Cryptocurrency, Equity, Equity Option, Future, etc.

**Date Filtering:**
- `start-date`, `end-date`: Date format (YYYY-MM-DD)
- `start-at`, `end-at`: Datetime format (ISO 8601)

**Response:**
```json
{
  "data": {
    "items": [
      {
        "id": "12345",
        "account-number": "12345678",
        "transaction-type": "Trade",
        "transaction-sub-type": "Buy",
        "symbol": "AAPL",
        "instrument-type": "Equity",
        "action": "Buy to Open",
        "quantity": 100,
        "price": 150.00,
        "executed-at": "2025-01-06T14:30:00Z",
        "transaction-date": "2025-01-06",
        "value": 15000.00,
        "commission": 0.00,
        "clearing-fees": 0.50,
        "regulatory-fees": 0.10
      }
    ]
  },
  "pagination": {
    "page-offset": 0,
    "per-page": 250,
    "total-items": 1234
  }
}
```

### Get Total Fees

**Endpoint:**
```
GET /accounts/{account_number}/transactions/total-fees
```

**Query Parameters:**
- `date`: Specific date (defaults to current day)

**Response:**
```json
{
  "data": {
    "total-fees": 125.50,
    "commission": 0.00,
    "clearing-fees": 50.25,
    "regulatory-fees": 25.10,
    "proprietary-index-fees": 50.15
  }
}
```

### Get Transaction by ID

**Endpoint:**
```
GET /accounts/{account_number}/transactions/{id}
```

### Fee Components

Tastytrade transactions may include:

- **Commission**: Trading commission (typically $0 for equities/options)
- **Clearing Fees**: Exchange clearing fees
- **Regulatory Fees**: SEC/FINRA fees
- **Currency Conversion Fees**: For multi-currency transactions
- **Proprietary Index Option Fees**: Fees for proprietary index options
- **Other Charges**: Additional fees as applicable

---

## Watchlists

### Public Watchlists

**Get All Public Watchlists:**
```
GET /public-watchlists
```

Returns all Tastyworks-curated watchlists.

**Get Specific Public Watchlist:**
```
GET /public-watchlists/{watchlist_name}
```

### User Watchlists

#### Create Watchlist

**Endpoint:**
```
POST /watchlists
```

**Request Body:**
```json
{
  "name": "My Tech Stocks",
  "watchlist-entries": [
    {
      "symbol": "AAPL",
      "instrument-type": "Equity"
    },
    {
      "symbol": "MSFT",
      "instrument-type": "Equity"
    },
    {
      "symbol": "GOOGL",
      "instrument-type": "Equity"
    }
  ],
  "group-name": "Technology",
  "order-index": 1
}
```

#### Get All Watchlists

**Endpoint:**
```
GET /watchlists
```

Returns all watchlists for the authenticated account.

#### Get Specific Watchlist

**Endpoint:**
```
GET /watchlists/{watchlist_name}
```

#### Update Watchlist

**Endpoint:**
```
PUT /watchlists/{watchlist_name}
```

Replaces all properties of the watchlist.

**Request Body:**
```json
{
  "name": "Updated Tech Stocks",
  "watchlist-entries": [
    {
      "symbol": "AAPL",
      "instrument-type": "Equity"
    },
    {
      "symbol": "NVDA",
      "instrument-type": "Equity"
    }
  ]
}
```

#### Delete Watchlist

**Endpoint:**
```
DELETE /watchlists/{watchlist_name}
```

### Pairs Watchlists

Specialized watchlists for pairs trading.

**Get All Pairs Watchlists:**
```
GET /pairs-watchlists
```

**Get Specific Pairs Watchlist:**
```
GET /pairs-watchlists/{pairs_watchlist_name}
```

### Watchlist Entry Structure

**Required Fields:**
- `symbol`: Instrument symbol

**Optional Fields:**
- `instrument-type`: Equity, Equity Option, Future, etc.
- `order-index`: Position in watchlist

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Expired or invalid authorization token |
| 403 | Forbidden | User lacks access to resource |
| 404 | Not Found | Endpoint or resource not found |
| 422 | Unprocessable | Action invalid in current context |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Server Error | Internal server error |

### Error Response Format

```json
{
  "error": {
    "code": "invalid_request",
    "message": "The account-number parameter is required",
    "errors": [
      {
        "code": "missing_parameter",
        "domain": "account-number"
      }
    ]
  }
}
```

### Common Error Scenarios

#### 401 Unauthorized
**Cause:** Expired or invalid session token

**Solution:**
1. Refresh session token (OAuth2)
2. Re-authenticate with credentials
3. Check Authorization header format

#### 403 Forbidden
**Cause:** Insufficient authority level

**Solution:**
1. Check account authority-level
2. Verify account permissions
3. Use appropriate account

#### 422 Unprocessable
**Cause:** Invalid operation in current context

**Examples:**
- Insufficient buying power
- Order validation failed
- Position already closed

**Solution:**
1. Check account balances
2. Validate order parameters
3. Use dry-run to test

#### 429 Rate Limit
**Cause:** Too many requests

**Solution:**
1. Implement exponential backoff
2. Cache responses when possible
3. Batch requests efficiently
4. Reduce request frequency

### Retry Strategy

**Recommended Approach:**

```python
import time

def exponential_backoff(attempt):
    """Calculate backoff time with jitter"""
    base_delay = 1  # seconds
    max_delay = 60  # seconds
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, 0.1 * delay)
    return delay + jitter

def retry_request(func, max_attempts=5):
    """Retry with exponential backoff"""
    for attempt in range(max_attempts):
        try:
            return func()
        except RateLimitError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(exponential_backoff(attempt))
```

---

## Best Practices

### Authentication

1. **Use OAuth2**: Migrate from session-token auth before December 1, 2025
2. **Secure Storage**: Store tokens securely (never in source code)
3. **Token Refresh**: Implement automatic token refresh (15-minute expiry)
4. **Error Handling**: Handle 401 errors with re-authentication logic

### API Usage

1. **Use `me` Placeholder**: Avoid exposing customer IDs
2. **Validate Authority**: Check authority-level before write operations
3. **Pagination**: Use appropriate per-page limits (default: 250)
4. **Field Selection**: Request only needed data fields

### Market Data

1. **Live Quotes**: Use DXLink streaming for real-time P/L calculations
2. **Token Management**: Refresh API quote tokens before 24-hour expiry
3. **Keepalive**: Send keepalive messages every 30 seconds
4. **Selective Subscriptions**: Subscribe only to needed symbols/events
5. **Multiple Channels**: Separate channels for different asset classes

### Order Management

1. **Dry-Run Testing**: Always test orders with dry-run before submission
2. **Order Validation**: Check warnings from dry-run responses
3. **Status Monitoring**: Track order status changes
4. **Error Recovery**: Implement retry logic for transient failures
5. **Idempotency**: Use order IDs to prevent duplicate submissions

### Performance

1. **Connection Pooling**: Reuse HTTP connections
2. **Compression**: Enable gzip compression for requests/responses
3. **Caching**: Cache instrument data and account info
4. **Parallel Requests**: Make independent requests concurrently
5. **Rate Limiting**: Implement client-side rate limiting

### Risk Management

1. **Position Limits**: Enforce position size limits
2. **Order Limits**: Limit order quantities and values
3. **Balance Checks**: Verify buying power before orders
4. **Stop Losses**: Implement automated stop-loss orders
5. **Real-Time Monitoring**: Monitor positions and P/L continuously

### Testing

1. **Sandbox First**: Test all workflows in sandbox environment
2. **Unit Tests**: Write tests for API interactions
3. **Integration Tests**: Test complete trading workflows
4. **Error Scenarios**: Test error handling and edge cases
5. **Load Testing**: Verify performance under load

### Logging & Monitoring

1. **Request Logging**: Log all API requests and responses
2. **Error Tracking**: Monitor and alert on errors
3. **Performance Metrics**: Track latency and throughput
4. **Audit Trail**: Maintain audit logs for compliance
5. **Order Tracking**: Log all order submissions and modifications

### Security

1. **HTTPS Only**: Use secure connections (TLS 1.2+)
2. **Token Security**: Never log or expose tokens
3. **Input Validation**: Validate all user inputs
4. **Principle of Least Privilege**: Use minimal required permissions
5. **Regular Audits**: Review security practices regularly

---

## Additional Resources

### Official Documentation
- **Developer Portal**: https://developer.tastytrade.com/
- **API Overview**: https://developer.tastytrade.com/api-overview/
- **OAuth2 Guide**: https://developer.tastytrade.com/api-guides/oauth/
- **Streaming Guide**: https://developer.tastytrade.com/streaming-market-data/

### SDK & Tools
- **Python SDK**: https://pypi.org/project/tastytrade/
- **Postman Collection**: https://www.postman.com/tastytradeapi/
- **DXLink Debugger**: demo.dxfeed.com

### Support
- **Support Portal**: https://support.tastytrade.com/
- **API Support**: Contact via support portal
- **Community**: GitHub discussions and forums

### Regulatory
- **FINRA**: Member organization
- **SIPC**: Securities investor protection
- **NFA**: Futures regulation
- **Crypto Custody**: Zero Hash LLC

---

## Changelog & Migration Notes

### 2025 Updates

**OAuth2 Migration (Critical)**
- **Deadline**: December 1, 2025
- **Action Required**: Migrate from session-token to OAuth2
- **Impact**: Session-token authentication will be discontinued

**Deprecated Endpoints**
- `GET /accounts/{account_number}/balances` - Use `/balances/{currency}` or `/balance-snapshots`
- Mark prices in positions - Use live DXLink quotes instead

**New Features**
- Enhanced OAuth2 support with refresh tokens
- Improved DXLink streaming capabilities
- Extended candle data ranges

---

## Quick Reference

### Common Workflows

**1. Initial Setup**
```
1. Authenticate (OAuth2 or session)
2. GET /customers/me/accounts
3. GET /accounts/{account_number}/balances
4. GET /accounts/{account_number}/positions
```

**2. Place Order**
```
1. POST /accounts/{account_number}/orders/dry-run (validate)
2. POST /accounts/{account_number}/orders (submit)
3. GET /accounts/{account_number}/orders/{order_id} (monitor)
```

**3. Market Data Streaming**
```
1. GET /api-quote-tokens
2. Connect to WebSocket (dxlink-url)
3. Send SETUP → AUTHORIZE → CHANNEL_REQUEST → FEED_SETUP → FEED_SUBSCRIPTION
4. Send KEEPALIVE every 30 seconds
```

**4. View Transactions**
```
1. GET /accounts/{account_number}/transactions?start-date=YYYY-MM-DD&end-date=YYYY-MM-DD
2. GET /accounts/{account_number}/transactions/total-fees
```

**5. Manage Watchlists**
```
1. GET /watchlists (view all)
2. POST /watchlists (create)
3. PUT /watchlists/{name} (update)
4. DELETE /watchlists/{name} (remove)
```

---

## Appendices

### A. Month Codes (Futures)

| Code | Month |
|------|-------|
| F | January |
| G | February |
| H | March |
| J | April |
| K | May |
| M | June |
| N | July |
| Q | August |
| U | September |
| V | October |
| X | November |
| Z | December |

### B. Instrument Types

- Bond
- Cryptocurrency
- Currency Pair
- Equity
- Equity Offering
- Equity Option
- Fixed Income Security
- Future
- Future Option
- Index
- Liquidity Pool
- Unknown
- Warrant

### C. Transaction Actions

- Allocate
- Buy
- Buy to Close
- Buy to Open
- Sell
- Sell to Close
- Sell to Open

### D. Authority Levels

- **owner**: Full account access (read/write/trade)
- **trade-only**: Trading operations only
- **read-only**: GET endpoints only

---

**Document Version:** 1.0
**Generated:** January 2025
**Source:** Tastytrade Developer Documentation

For the latest updates and detailed technical specifications, always refer to the official Tastytrade API documentation at https://developer.tastytrade.com/
