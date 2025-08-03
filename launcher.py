#!/usr/bin/env python3
"""
Launch script for Accident Detection MLOps Dashboard
Starts both FastAPI backend and Streamlit frontend
"""

import subprocess
import time
import sys
import os
import signal
import webbrowser
from threading import Thread

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def start_fastapi():
    """Start FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        # Change to project directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Start FastAPI
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n🛑 FastAPI server stopped")
    except Exception as e:
        print(f"❌ Error starting FastAPI: {e}")

def start_streamlit():
    """Start Streamlit dashboard"""
    print("🎨 Starting Streamlit dashboard...")
    try:
        # Wait for FastAPI to start
        time.sleep(3)
        
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py", 
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def main():
    """Main launcher function"""
    print("🚗 Accident Detection MLOps Dashboard Launcher")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Start services
    print("\n🔄 Starting services...")
    
    try:
        # Start FastAPI in a separate thread
        fastapi_thread = Thread(target=start_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Wait a moment then start Streamlit
        time.sleep(3)
        
        print("\n✅ Services starting...")
        print("📊 FastAPI: http://localhost:8000")
        print("🎨 Streamlit: http://localhost:8501")
        print("\nPress Ctrl+C to stop all services")
        
        # Open browser
        time.sleep(2)
        webbrowser.open("http://localhost:8501")
        
        # Start Streamlit (blocking)
        start_streamlit()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
