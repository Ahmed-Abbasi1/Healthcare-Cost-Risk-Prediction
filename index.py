"""
Vercel Entry Point for Healthcare Cost Prediction API
This file is required by Vercel to identify the FastAPI application
"""
from app.api import app

# Vercel expects an 'app' or 'handler' variable
handler = app
