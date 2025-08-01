#!/bin/bash
# Deployment startup script for Replit
echo "Starting Support Ticket Summarizer..."

# Set deployment environment variables
export STREAMLIT_SERVER_PORT=5000
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start Streamlit with deployment configuration
exec streamlit run streamlit_app.py \
    --server.port $STREAMLIT_SERVER_PORT \
    --server.address $STREAMLIT_SERVER_ADDRESS \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base light