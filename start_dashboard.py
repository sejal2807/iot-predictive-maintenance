#!/usr/bin/env python3
"""
IoT Predictive Maintenance Dashboard Startup Script
Simple script to start the enhanced dashboard
"""

import subprocess
import sys
import os

def main():
    """Start the IoT Predictive Maintenance Dashboard"""
    print("🔧 Starting IoT Predictive Maintenance Dashboard...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} found")
    except ImportError:
        print("❌ Streamlit not found. Installing dependencies...")
        print("🔧 Using Python 3.13+ compatible versions...")
        
        # Try minimal requirements first
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_minimal.txt"], check=True)
            print("✅ Minimal dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️ Minimal install failed, trying enhanced requirements...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_enhanced.txt"], check=True)
                print("✅ Enhanced dependencies installed successfully")
            except subprocess.CalledProcessError:
                print("❌ Failed to install dependencies. Please install manually:")
                print("   pip install streamlit plotly pandas numpy")
                return
    
    # Start the enhanced dashboard
    print("🚀 Launching Enhanced Dashboard...")
    print("📊 Features:")
    print("   • Real-time sensor monitoring")
    print("   • Interactive visualizations")
    print("   • Live data simulation")
    print("   • Anomaly detection")
    print("   • Predictive maintenance")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "run_app_enhanced.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main()
