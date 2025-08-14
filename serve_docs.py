#!/usr/bin/env python3
"""
Simple script to serve the ContextManager documentation.
"""

import os
import sys
import http.server
import socketserver
from pathlib import Path

def main():
    # Change to docs directory
    docs_dir = Path(__file__).parent / "docs"
    if not docs_dir.exists():
        print("❌ Error: docs directory not found!")
        sys.exit(1)
    
    os.chdir(docs_dir)
    
    # Set up server
    PORT = 8080
    
    class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers for development
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"🚀 ContextManager Documentation Server")
            print(f"📖 Serving at: http://localhost:{PORT}")
            print(f"📁 Directory: {docs_dir.absolute()}")
            print(f"🌐 Open your browser and navigate to: http://localhost:{PORT}")
            print(f"⏹️  Press Ctrl+C to stop the server")
            print("-" * 50)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Error: Port {PORT} is already in use!")
            print(f"💡 Try a different port: python3 serve_docs.py --port 8081")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 