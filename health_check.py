#!/usr/bin/env python3
"""
Health check server that runs alongside Streamlit to provide a simple health endpoint.
This helps with deployment health checks.
"""

import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os

class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple health check handler."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health' or self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_data = {
                'status': 'healthy',
                'service': 'support-ticket-summarizer',
                'timestamp': time.time(),
                'message': 'Streamlit app is running'
            }
            
            self.wfile.write(json.dumps(health_data).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

def start_health_server(port=8080):
    """Start the health check server."""
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"Health check server starting on port {port}")
    server.serve_forever()

def main():
    """Main function to start health check server."""
    port = int(os.environ.get('HEALTH_CHECK_PORT', 8080))
    
    try:
        start_health_server(port)
    except KeyboardInterrupt:
        print("\nHealth check server shutting down...")
    except Exception as e:
        print(f"Error starting health check server: {e}")

if __name__ == "__main__":
    main()