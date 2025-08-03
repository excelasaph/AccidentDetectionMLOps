"""
Simple startup script for Streamlit dashboard
"""

import subprocess
import sys
import time
import webbrowser
from threading import Thread

def start_streamlit():
    """Start Streamlit dashboard"""
    print("🎨 Starting Streamlit dashboard...")
    print("📍 Dashboard will be available at: http://localhost:8501")
    print("\n🛑 Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Start streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py", 
            "--server.port", "8501",
            "--server.address", "127.0.0.1"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit dashboard: {e}")

def open_browser():
    """Open browser after a short delay"""
    time.sleep(5)  # Wait for streamlit to start
    try:
        print("🌐 Opening browser...")
        webbrowser.open("http://localhost:8501")
    except Exception as e:
        print(f"⚠️ Could not open browser automatically: {e}")

if __name__ == "__main__":
    print("🚗 Accident Detection MLOps - Streamlit Dashboard")
    print("=" * 50)
    
    # Open browser in background
    browser_thread = Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start Streamlit dashboard (blocking)
    start_streamlit()
