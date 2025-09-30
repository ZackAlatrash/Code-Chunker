
#!/usr/bin/env python3
"""
Startup script for the RAG Code Search Web Application
Helps start both backend and provide instructions for frontend
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import requests

def check_opensearch():
    """Check if OpenSearch is running"""
    try:
        response = requests.get("http://localhost:9200", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def check_backend():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def start_backend():
    """Start the FastAPI backend"""
    backend_path = Path(__file__).parent / "backend" / "app.py"
    if not backend_path.exists():
        print("❌ Backend app.py not found!")
        return None
    
    print("🚀 Starting FastAPI backend...")
    try:
        # Start backend in background
        process = subprocess.Popen(
            [sys.executable, str(backend_path)],
            cwd=backend_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if it's running
        if check_backend():
            print("✅ Backend started successfully on http://localhost:8000")
            return process
        else:
            print("❌ Backend failed to start properly")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def open_frontend():
    """Open the frontend in browser"""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if not frontend_path.exists():
        print("❌ Frontend index.html not found!")
        return False
    
    try:
        webbrowser.open(f"file://{frontend_path.absolute()}")
        print("🌐 Frontend opened in your default browser")
        return True
    except Exception as e:
        print(f"❌ Failed to open frontend: {e}")
        print(f"   Manually open: {frontend_path.absolute()}")
        return False

def main():
    print("🔍 RAG Code Search - Web Application Startup")
    print("=" * 50)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    # Check OpenSearch
    if check_opensearch():
        print("✅ OpenSearch is running on http://localhost:9200")
    else:
        print("❌ OpenSearch is not running on http://localhost:9200")
        print("   Please start OpenSearch first:")
        print("   cd ops/ && docker-compose up -d")
        return False
    
    # Check if backend is already running
    if check_backend():
        print("✅ Backend is already running on http://localhost:8000")
        backend_process = None
    else:
        # Start backend
        backend_process = start_backend()
        if not backend_process:
            return False
    
    # Open frontend
    print("\n🌐 Opening frontend...")
    frontend_opened = open_frontend()
    
    # Provide instructions
    print("\n📚 Usage Instructions:")
    print("- Backend API: http://localhost:8000")
    print("- API Docs: http://localhost:8000/docs")
    print("- Health Check: http://localhost:8000/health")
    
    if frontend_opened:
        print("- Frontend: Opened in browser")
    else:
        frontend_path = Path(__file__).parent / "frontend" / "index.html"
        print(f"- Frontend: Open {frontend_path.absolute()} manually")
    
    print("\n🔍 Try searching for:")
    print("  • 'authentication middleware'")
    print("  • 'database connection'")
    print("  • 'error handling'")
    print("  • 'JWT token validation'")
    
    # Keep backend running if we started it
    if backend_process:
        print(f"\n⏳ Backend process running (PID: {backend_process.pid})")
        print("Press Ctrl+C to stop the backend and exit")
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down backend...")
            backend_process.terminate()
            backend_process.wait()
            print("✅ Backend stopped")
    else:
        print("\n✨ All services are running!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
