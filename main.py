#!/usr/bin/env python3
"""
Main entry point for deployment.
Robust Streamlit application launcher for Replit Autoscale.
"""

if __name__ == "__main__":
    import subprocess
    import sys
    import os
    import signal
    import time
    
    # Get port from environment (Replit Autoscale sets this)
    port = os.environ.get('PORT', '5000')
    
    print(f"🚀 Support Ticket Summarizer starting on port {port}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    print(f"🌐 Executing: {' '.join(cmd)}")
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        print("🛑 Received shutdown signal, exiting gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run Streamlit
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("🛑 Interrupted by user")
        if 'process' in locals():
            process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)