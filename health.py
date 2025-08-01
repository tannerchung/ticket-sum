#!/usr/bin/env python3
"""
Health check endpoint for deployment verification.
This provides a simple HTTP response for deployment health checks.
"""

import json
import sys
from datetime import datetime

def main():
    """Simple health check response."""
    health_response = {
        "status": "healthy",
        "service": "support-ticket-summarizer",
        "timestamp": datetime.now().isoformat(),
        "message": "Streamlit application is running successfully",
        "python_version": sys.version,
        "deployment": "ready"
    }
    
    print("Content-Type: application/json\n")
    print(json.dumps(health_response, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())