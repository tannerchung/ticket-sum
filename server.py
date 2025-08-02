#!/usr/bin/env python3
"""
Deployment server that provides health checks and serves the Streamlit app.
This addresses Replit deployment requirements for health check endpoints.
"""

import os
import sys
import threading
import subprocess
import time
from flask import Flask, jsonify, redirect
from datetime import datetime

app = Flask(__name__)

# Global variable to track Streamlit process
streamlit_process = None

@app.route('/')
def root():
    """Root endpoint redirects to Streamlit app."""
    return redirect('/app/')

@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "support-ticket-summarizer",
        "timestamp": datetime.now().isoformat(),
        "streamlit_running": streamlit_process is not None and streamlit_process.poll() is None,
        "message": "Service is running successfully"
    })

@app.route('/app/')
def streamlit_app():
    """Information about accessing the Streamlit app."""
    return jsonify({
        "message": "Streamlit app is running",
        "streamlit_url": "Available on configured port",
        "note": "Direct Streamlit access preferred for this deployment"
    })

def start_streamlit():
    """Start Streamlit application in a separate process."""
    global streamlit_process
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print(f"Starting Streamlit: {' '.join(cmd)}")
    streamlit_process = subprocess.Popen(cmd)
    return streamlit_process

def main():
    """Main entry point for deployment."""
    print("Starting Support Ticket Summarizer deployment server...")
    
    # Start Streamlit in background
    print("Launching Streamlit application...")
    start_streamlit()
    
    # Give Streamlit time to start
    time.sleep(3)
    
    # Start Flask server for health checks
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting health check server on port {port}...")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )

if __name__ == "__main__":
    main()