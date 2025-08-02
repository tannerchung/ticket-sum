#!/usr/bin/env python3
"""
Main entry point for deployment.
Simple and direct Streamlit application launcher.
"""

if __name__ == "__main__":
    import subprocess
    import sys
    import os
    
    # Use environment port or default to 5000
    port = os.environ.get('PORT', '5000')
    
    # Start Streamlit application for deployment
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true", 
        "--browser.gatherUsageStats", "false"
    ]
    
    print(f"Starting Streamlit on port {port}")
    subprocess.run(cmd)