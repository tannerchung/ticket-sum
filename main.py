#!/usr/bin/env python3
"""
Main entry point for deployment.
This file serves as the deployment entry point since Replit deployment expects specific file names.
"""

import subprocess
import sys
import os

def main():
    """Start the Streamlit application for deployment."""
    # Set deployment environment variables
    os.environ['STREAMLIT_SERVER_PORT'] = '5000'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    print("Starting Streamlit application for deployment...")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Try direct streamlit command first
    try:
        result = subprocess.run(['streamlit', '--version'], capture_output=True, text=True)
        print(f"Streamlit version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Streamlit command not found, trying python -m streamlit")
    
    # Command to run Streamlit
    cmd = [
        'streamlit', 'run', 'streamlit_app.py',
        '--server.port', '5000',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ]
    
    try:
        # Try direct streamlit command
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Direct streamlit command failed: {e}")
        # Fallback to python -m streamlit
        cmd_fallback = [sys.executable, '-m'] + cmd
        print(f"Trying fallback: {' '.join(cmd_fallback)}")
        try:
            subprocess.run(cmd_fallback, check=True)
        except subprocess.CalledProcessError as e2:
            print(f"Error starting Streamlit: {e2}")
            sys.exit(1)

if __name__ == "__main__":
    main()