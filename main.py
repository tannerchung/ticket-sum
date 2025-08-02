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
    
    # Try direct streamlit command first - secure static command array
    try:
        # Static command for version check - secure against command injection
        version_cmd = ['streamlit', '--version']
        result = subprocess.run(version_cmd, capture_output=True, text=True)
        print(f"Streamlit version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Streamlit command not found, trying python -m streamlit")
    
    # Static command to run Streamlit - secure against command injection
    cmd = [
        'streamlit', 
        'run', 
        'streamlit_app.py',
        '--server.port', 
        '5000',
        '--server.address', 
        '0.0.0.0',
        '--server.headless', 
        'true',
        '--browser.gatherUsageStats', 
        'false'
    ]
    
    try:
        # Try direct streamlit command with static command array
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Direct streamlit command failed: {e}")
        # Fallback to python -m streamlit with validated static command construction
        # Security: validate Python executable path before use
        python_executable = sys.executable
        if not python_executable or not os.path.exists(python_executable):
            print("Error: Invalid Python executable path")
            sys.exit(1)
        
        # Static command array - secure against command injection
        cmd_fallback = [
            python_executable,
            '-m',
            'streamlit', 
            'run', 
            'streamlit_app.py',
            '--server.port', 
            '5000',
            '--server.address', 
            '0.0.0.0',
            '--server.headless', 
            'true',
            '--browser.gatherUsageStats', 
            'false'
        ]
        print(f"Trying fallback: {' '.join(cmd_fallback)}")
        try:
            # Execute validated static command array
            subprocess.run(cmd_fallback, check=True)
        except subprocess.CalledProcessError as e2:
            print(f"Error starting Streamlit: {e2}")
            sys.exit(1)

if __name__ == "__main__":
    main()