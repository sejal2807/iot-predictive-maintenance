#!/usr/bin/env python3
"""
Dependency Installation Script for Python 3.13+
Handles compatibility issues with older package versions
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package with error handling"""
    try:
        print(f"📦 Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install dependencies for Python 3.13+"""
    print("🔧 Installing IoT Dashboard Dependencies for Python 3.13+")
    print("=" * 60)
    
    # Core packages that should work with Python 3.13
    packages = [
        "streamlit",
        "plotly", 
        "pandas",
        "numpy"
    ]
    
    # Optional packages (may fail on some systems)
    optional_packages = [
        "scikit-learn",
        "tensorflow",
        "pyyaml"
    ]
    
    print("📦 Installing core packages...")
    core_success = True
    for package in packages:
        if not install_package(package):
            core_success = False
    
    if not core_success:
        print("❌ Some core packages failed to install")
        print("💡 Try running: pip install --upgrade pip")
        print("💡 Then run: pip install streamlit plotly pandas numpy")
        return False
    
    print("\n📦 Installing optional packages...")
    for package in optional_packages:
        install_package(package)  # Don't fail if optional packages fail
    
    print("\n✅ Installation complete!")
    print("🚀 You can now run: streamlit run run_app_enhanced.py")
    return True

if __name__ == "__main__":
    main()
