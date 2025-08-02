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
    
    # Security-hardened command construction with comprehensive validation
    python_executable = sys.executable
    if not python_executable or not Path(python_executable).exists():
        print("Error: Invalid Python executable path")
        sys.exit(1)
    
    # Validate Python executable is safe
    if not isinstance(python_executable, str) or len(python_executable) == 0:
        print("Error: Invalid Python executable type")
        sys.exit(1)
    
    # Define static constants to prevent any injection
    MODULE_FLAG = '-m'
    STREAMLIT_MODULE = 'streamlit'
    RUN_COMMAND = 'run'
    APP_FILE = 'streamlit_app.py'
    PORT_FLAG = '--server.port'
    PORT_VALUE = '5000'
    ADDRESS_FLAG = '--server.address'
    ADDRESS_VALUE = '0.0.0.0'
    HEADLESS_FLAG = '--server.headless'
    HEADLESS_VALUE = 'true'
    STATS_FLAG = '--browser.gatherUsageStats'
    STATS_VALUE = 'false'
    
    # Validate all static components for security
    static_components = [MODULE_FLAG, STREAMLIT_MODULE, RUN_COMMAND, APP_FILE,
                        PORT_FLAG, PORT_VALUE, ADDRESS_FLAG, ADDRESS_VALUE,
                        HEADLESS_FLAG, HEADLESS_VALUE, STATS_FLAG, STATS_VALUE]
    
    for component in static_components:
        if not isinstance(component, str) or len(component) == 0:
            raise ValueError(f"Invalid command component: {component}")
    
    # Construct validated static command array
    cmd = [python_executable, MODULE_FLAG, STREAMLIT_MODULE, RUN_COMMAND,
           APP_FILE, PORT_FLAG, PORT_VALUE, ADDRESS_FLAG, ADDRESS_VALUE,
           HEADLESS_FLAG, HEADLESS_VALUE, STATS_FLAG, STATS_VALUE]
    
    print("Starting Streamlit application...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Execute comprehensively validated command array with timeout protection
        subprocess.run(cmd, check=True, timeout=30)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)

if __name__ == "__main__":
    main()