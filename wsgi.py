"""
WSGI entry point for production deployment.
This file is used by production WSGI servers like Gunicorn, uWSGI, or Waitress.
"""
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Flask app
from app import app as application

# For compatibility with some WSGI servers
app = application

if __name__ == "__main__":
    # For local testing with production server
    # Run with: python wsgi.py
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting production server on http://0.0.0.0:{port}")
    serve(app, host='0.0.0.0', port=port, threads=4)
