#!/usr/bin/env python3
"""
Entry point for the Streamlit application deployment.
This file ensures proper startup for Replit deployment.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit application with proper configuration."""
    # Ensure we're in the correct directory
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Set environment variables for deployment
    os.environ.setdefault('STREAMLIT_SERVER_PORT', '5000')
    os.environ.setdefault('STREAMLIT_SERVER_ADDRESS', '0.0.0.0')
    os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
    os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    
    # Command to run Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
        '--server.port', '5000',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ]
    
    print("Starting Streamlit application...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run Streamlit
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)

if __name__ == "__main__":
    main()