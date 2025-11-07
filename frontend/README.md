# Options Trading Bot - Frontend Dashboard

A sophisticated, professional-grade frontend interface for the Advanced Options Trading Bot, built with Streamlit and featuring real-time monitoring, strategy management, and comprehensive analytics.

## 🎯 Features

### Core Dashboard Components
- **📊 Portfolio Monitor** - Real-time portfolio value, P&L, and position tracking
- **🎯 Strategy Management** - Configure and monitor multiple trading strategies
- **🤖 Model Ensemble** - AI model performance tracking and weight management
- **⚖️ Risk Management** - Comprehensive risk monitoring with VaR and Greeks
- **📈 Performance Analytics** - Advanced performance attribution and metrics
- **🔧 Configuration** - Intuitive settings for all system parameters
- **📋 System Logs** - Real-time logging and system health monitoring

### Advanced Features
- **Multi-Model Consensus Trading** - Visual display of 4 pricing models working together
- **Dynamic Risk Adjustment** - Real-time risk scaling based on market conditions
- **Strategy Performance Attribution** - Detailed breakdown of returns by strategy
- **Market Regime Detection** - Visual indicators of current market conditions
- **Greeks Exposure Tracking** - Real-time portfolio Greeks with historical trends
- **Auto-refresh Functionality** - Configurable real-time data updates

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)
- Active Tastyworks account (for live trading)

### Installation

1. **Clone and Setup**
```bash
# Navigate to frontend directory
cd frontend/

# Create virtual environment
python -m venv dashboard_env
source dashboard_env/bin/activate  # Linux/Mac
# or dashboard_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

2. **Run the Dashboard**
```bash
# Basic run
streamlit run multipage_app.py

# Custom configuration
streamlit run multipage_app.py --server.port 8501 --server.address 0.0.0.0
```

3. **Access the Dashboard**
Open your browser and navigate to: `http://localhost:8501`

### Optional: External Ensemble Service

Universe scans can be delegated to the standalone FastAPI service. This keeps
Streamlit lightweight and avoids importing heavy scientific dependencies.

1. Deploy the service (see `backend/service/README.md`), or run it locally:
   ```bash
   cd backend
   python -m service.main
   ```
2. Point the frontend at the service before launching Streamlit:
   ```bash
   export ENSEMBLE_SERVICE_URL="http://localhost:8000"
   export ENSEMBLE_RISK_LEVEL="moderate"
   streamlit run app.py
   ```
3. If the service is unreachable, the frontend gracefully falls back to cached
   data (and the ensemble banner will show a warning).

### Docker Deployment

1. **Build and Run**
```bash
# Build the image
docker build -t options-trading-dashboard .

# Run the container
docker run -p 8501:8501 options-trading-dashboard
```

2. **Using Docker Compose**
```bash
# Start all services
docker-compose up -d

# Start with monitoring (optional)
docker-compose --profile monitoring up -d

# Start production setup with nginx
docker-compose --profile production up -d
```

## 📱 User Interface Overview

### Navigation Structure
The dashboard uses a horizontal navigation menu with 8 main sections:

1. **📊 Dashboard** - Main overview with key metrics and quick stats
2. **💼 Portfolio** - Detailed portfolio analysis and position management
3. **🎯 Strategies** - Strategy configuration and performance monitoring
4. **🤖 Models** - Model ensemble analysis and calibration status
5. **⚖️ Risk** - Comprehensive risk management and scenario analysis
6. **🔧 Config** - System configuration and parameter tuning
7. **📈 Analytics** - Advanced performance metrics and attribution
8. **📋 Logs** - System logs and health monitoring

### Key UI Components

#### Main Dashboard
- **Portfolio Value Card** - Current portfolio value with daily P&L
- **Greeks Cards** - Real-time portfolio Delta, Gamma, Theta, Vega
- **Strategy Status** - Active strategies with performance indicators
- **Risk Gauge** - Visual risk score with color-coded warnings

#### Model Ensemble Section
- **Model Weights Pie Chart** - Current allocation across 4 pricing models
- **Accuracy Trends** - Historical accuracy for each model
- **Predictions Table** - Current model predictions vs market prices
- **Calibration Status** - Last calibration date and quality scores

#### Risk Management
- **Greeks Exposure Charts** - Time series of portfolio Greeks
- **VaR Gauges** - Visual representation of Value at Risk
- **Scenario Analysis** - Stress testing under various market conditions
- **Circuit Breakers** - Real-time status of risk controls

## 🎛️ Configuration Options

### Strategy Settings
```python
# Model Ensemble Configuration
min_edge_magnitude = 3.0%        # Minimum edge to trade
min_confidence_score = 0.7       # Minimum model confidence
max_model_disagreement = 25.0%   # Maximum allowable disagreement

# Gamma Scalping Parameters
rebalance_threshold = 0.10       # Delta threshold for rebalancing
min_gamma_exposure = 0.03        # Minimum gamma to initiate position
transaction_cost_threshold = $50  # Minimum profit vs costs
```

### Risk Management
```python
# Position Limits
max_position_size = 25.0%        # Max % of portfolio per position
max_leverage_ratio = 2.0         # Maximum leverage allowed
daily_loss_limit = 5.0%          # Daily loss circuit breaker

# Greeks Limits
max_portfolio_delta = 0.30       # Maximum net delta exposure
max_portfolio_vega = $5,000      # Maximum vega exposure
```

### Model Ensemble
```python
# Model Weights (dynamically adjusted)
black_scholes_weight = 0.20      # Traditional BS model
merton_jump_weight = 0.25        # Jump diffusion model
heston_weight = 0.30             # Stochastic volatility
ml_neural_weight = 0.25          # Neural network model

# Calibration Schedule
merton_calibration = "weekly"    # Jump model recalibration
heston_calibration = "daily"     # Vol model recalibration  
ml_retraining = "monthly"        # NN retraining frequency
```

## 🔒 Security Features

### Authentication (Production Ready)
- Session-based authentication
- Role-based access control
- API key management
- IP whitelist support

### Data Protection
- Encrypted data transmission
- Secure credential storage
- Audit logging
- Sensitive data masking in logs

### Risk Controls
- Circuit breakers for emergency stops
- Multi-level approval for large trades
- Real-time monitoring alerts
- Automated risk limit enforcement

## 📊 Performance Monitoring

### Key Metrics Tracked
- **Portfolio Performance**: Daily/monthly returns, Sharpe ratio, max drawdown
- **Strategy Attribution**: Individual strategy contributions to returns
- **Model Accuracy**: Real-time tracking of model prediction accuracy
- **Risk Metrics**: VaR, Expected Shortfall, Greeks exposure
- **System Health**: API uptime, latency, error rates

### Alerting System
- Email/SMS alerts for risk breaches
- Real-time notifications for system errors
- Performance degradation warnings
- Model calibration reminders

## 🛠️ Development & Customization

### Adding New Pages
```python
def render_custom_page(self):
    """Add your custom page here"""
    st.title("📊 Custom Analysis")
    
    # Your custom content
    custom_data = self.fetch_custom_data()
    fig = create_custom_chart(custom_data)
    st.plotly_chart(fig, use_container_width=True)

# Register in multipage_app.py navigation
if selected_page == "Custom":
    self.render_custom_page()
```

### Custom Metrics
```python
def add_custom_metric(self, name: str, value: float, delta: float = None):
    """Add custom performance metric"""
    with st.container():
        st.metric(
            label=name,
            value=f"${value:,.2f}",
            delta=f"{delta:+.1%}" if delta else None
        )
```

### Theming & Styling
Modify the CSS in `multipage_app.py` to customize the appearance:

```css
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Port Already in Use**
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run multipage_app.py --server.port 8502
```

2. **Memory Issues**
```bash
# Increase Docker memory limit
docker run --memory=2g -p 8501:8501 options-trading-dashboard
```

3. **WebSocket Connection Errors**
```bash
# Check firewall settings and enable WebSocket support
# Ensure port 8501 is accessible
```

### Performance Optimization

1. **Caching Configuration**
```python
# Add to your functions
@st.cache_data(ttl=30)  # Cache for 30 seconds
def expensive_calculation():
    return complex_computation()
```

2. **Data Loading Optimization**
```python
# Use pagination for large datasets
@st.fragment  # Only rerun this section
def render_large_table():
    page_size = 50
    total_rows = len(data)
    page = st.number_input("Page", 1, (total_rows // page_size) + 1)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    st.dataframe(data[start_idx:end_idx])
```

## 📋 Production Deployment Checklist

### Pre-Deployment
- [ ] Test all functionality in development
- [ ] Configure production database connections
- [ ] Set up SSL certificates
- [ ] Configure monitoring and alerting
- [ ] Test backup and recovery procedures
- [ ] Security audit and penetration testing

### Environment Variables
Create a `.env` file with:
```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://user:pass@host:port

# Trading API
TASTYWORKS_USERNAME=your_username
TASTYWORKS_PASSWORD=your_password
TASTYWORKS_ACCOUNT=your_account

# Security
SECRET_KEY=your_secret_key
JWT_SECRET=your_jwt_secret

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
```

### Scaling Considerations
- Use load balancer for multiple dashboard instances
- Implement session stickiness for real-time features
- Consider CDN for static assets
- Monitor resource usage and scale accordingly

## 🤝 Support & Contributing

### Getting Help
1. Check the troubleshooting section
2. Review system logs in the dashboard
3. Create GitHub issue with:
   - Steps to reproduce
   - Error messages
   - System configuration

### Contributing
1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Submit pull request with detailed description

### Feature Requests
We welcome feature requests! Please include:
- Use case description
- Expected behavior
- UI mockups (if applicable)
- Priority level

## 📄 License

This frontend dashboard is part of the Advanced Options Trading Bot project. See LICENSE file for details.

---

**⚠️ Risk Disclaimer**: This software is for educational purposes only. Options trading involves significant risk and may not be suitable for all investors. Past performance does not guarantee future results.
