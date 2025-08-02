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
    
    # Security-hardened version check with validated static command
    try:
        # Explicitly validated static command array - secure against injection
        STREAMLIT_CMD = 'streamlit'  # Static constant
        VERSION_FLAG = '--version'  # Static constant
        
        # Validate that command components are safe static strings
        if not isinstance(STREAMLIT_CMD, str) or not STREAMLIT_CMD.replace('_', '').replace('-', '').isalpha():
            raise ValueError("Invalid command component")
        if not isinstance(VERSION_FLAG, str) or not VERSION_FLAG.startswith('--'):
            raise ValueError("Invalid flag component")
            
        version_cmd = [STREAMLIT_CMD, VERSION_FLAG]
        result = subprocess.run(version_cmd, capture_output=True, text=True, timeout=10)
        print(f"Streamlit version: {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        print("Streamlit command not found, trying python -m streamlit")
    
    # Security-hardened command construction with validation
    # Define static constants to prevent any injection
    STREAMLIT_CMD = 'streamlit'
    RUN_SUBCMD = 'run' 
    APP_FILE = 'streamlit_app.py'
    SERVER_PORT_FLAG = '--server.port'
    PORT_VALUE = '5000'
    SERVER_ADDR_FLAG = '--server.address'
    ADDR_VALUE = '0.0.0.0'
    HEADLESS_FLAG = '--server.headless'
    HEADLESS_VALUE = 'true'
    STATS_FLAG = '--browser.gatherUsageStats'
    STATS_VALUE = 'false'
    
    # Validate all components are safe
    safe_components = [STREAMLIT_CMD, RUN_SUBCMD, APP_FILE, SERVER_PORT_FLAG, 
                      PORT_VALUE, SERVER_ADDR_FLAG, ADDR_VALUE, HEADLESS_FLAG, 
                      HEADLESS_VALUE, STATS_FLAG, STATS_VALUE]
    
    for component in safe_components:
        if not isinstance(component, str) or len(component) == 0:
            raise ValueError(f"Invalid command component: {component}")
    
    cmd = [STREAMLIT_CMD, RUN_SUBCMD, APP_FILE, SERVER_PORT_FLAG, PORT_VALUE,
           SERVER_ADDR_FLAG, ADDR_VALUE, HEADLESS_FLAG, HEADLESS_VALUE,
           STATS_FLAG, STATS_VALUE]
    
    try:
        # Execute validated static command array with timeout
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, timeout=30)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Direct streamlit command failed: {e}")
        # Security-hardened fallback with comprehensive validation
        python_executable = sys.executable
        if not python_executable or not os.path.exists(python_executable):
            print("Error: Invalid Python executable path")
            sys.exit(1)
        
        # Validate Python executable is safe
        if not isinstance(python_executable, str) or len(python_executable) == 0:
            print("Error: Invalid Python executable type")
            sys.exit(1)
        
        # Define static constants for fallback command
        MODULE_FLAG = '-m'
        STREAMLIT_MODULE = 'streamlit'
        RUN_CMD = 'run'
        APP_FILE = 'streamlit_app.py'
        PORT_FLAG = '--server.port'
        PORT_VAL = '5000'
        ADDR_FLAG = '--server.address'
        ADDR_VAL = '0.0.0.0'
        HEAD_FLAG = '--server.headless'
        HEAD_VAL = 'true'
        STAT_FLAG = '--browser.gatherUsageStats'
        STAT_VAL = 'false'
        
        # Validate all static components
        static_components = [MODULE_FLAG, STREAMLIT_MODULE, RUN_CMD, APP_FILE,
                           PORT_FLAG, PORT_VAL, ADDR_FLAG, ADDR_VAL,
                           HEAD_FLAG, HEAD_VAL, STAT_FLAG, STAT_VAL]
        
        for comp in static_components:
            if not isinstance(comp, str) or len(comp) == 0:
                raise ValueError(f"Invalid static component: {comp}")
        
        cmd_fallback = [python_executable, MODULE_FLAG, STREAMLIT_MODULE, RUN_CMD,
                       APP_FILE, PORT_FLAG, PORT_VAL, ADDR_FLAG, ADDR_VAL,
                       HEAD_FLAG, HEAD_VAL, STAT_FLAG, STAT_VAL]
        
        print(f"Trying fallback: {' '.join(cmd_fallback)}")
        try:
            # Execute comprehensively validated command array with timeout
            subprocess.run(cmd_fallback, check=True, timeout=30)
        except subprocess.CalledProcessError as e2:
            print(f"Error starting Streamlit: {e2}")
            sys.exit(1)

if __name__ == "__main__":
    main()