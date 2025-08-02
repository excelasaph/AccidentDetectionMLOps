#!/usr/bin/env python3
"""
FastAPI Server Startup Script
=============================
Starts the AccidentDetectionMLOps FastAPI server with optimal settings.
"""

import uvicorn
import os

if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 Starting AccidentDetectionMLOps FastAPI Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔄 Auto-reload enabled for development")
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
