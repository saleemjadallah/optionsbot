#!/bin/bash

# Run the Options Trading Bot Dashboard
# ======================================

echo "Starting Options Trading Bot Dashboard..."
echo "========================================="

# Navigate to the frontend directory
cd "$(dirname "$0")"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: Virtual environment not detected"
    echo "Activating options_trader_env..."
    source ../options_trader_env/bin/activate
fi

# Run the Streamlit dashboard
streamlit run dashboard.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.primaryColor="#1f77b4" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f0f2f6" \
    --theme.textColor="#262730"

echo "Dashboard is running at http://localhost:8501"