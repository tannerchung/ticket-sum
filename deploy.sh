#!/bin/bash
# Robust deployment script for Replit Autoscale
set -e

echo "🚀 Starting Support Ticket Summarizer deployment..."

# Force use of system Python and pip, avoiding any uv conflicts
export PATH="/nix/store/yaps09f01jp3fd1405qlr0qz6haf6z03-python3.11-pip-25.0.1/bin:/usr/bin:/bin:$PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"

# Clear any problematic environment variables that might cause uv to be used
unset UV_PYTHON
unset UV_INSTALL

# Ensure we have the latest pip
echo "📦 Setting up Python environment..."
python -m pip install --quiet --upgrade pip setuptools wheel

# Install dependencies from requirements.txt using pip only
echo "📦 Installing application dependencies..."
python -m pip install --quiet --no-cache-dir -r requirements.txt

# Verify critical imports work
echo "🔍 Verifying installation..."
python -c "import streamlit, crewai, langfuse; print('✅ Core modules imported successfully')"

# Set deployment environment variables
export PORT=${PORT:-5000}
export STREAMLIT_SERVER_PORT=$PORT
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

echo "🌐 Starting Streamlit application on port $PORT..."
exec python main.py