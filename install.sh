#!/bin/bash
# Installation script that ensures our mock uv is prioritized
set -e

echo "🔧 Setting up deployment environment..."

# Copy our mock uv to a location that will be prioritized
cp uv /tmp/uv
chmod +x /tmp/uv

# Ensure our mock uv is first in PATH
export PATH="/tmp:$PWD:$PATH"

echo "✅ Mock uv setup complete at: $(which uv)"

# Run the actual installation
./uv sync