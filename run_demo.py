"""
Demo script to generate sample data and run the IoT Predictive Maintenance Dashboard
"""

import os
import sys
import subprocess
from datetime import datetime

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🔧 IoT Predictive Maintenance Dashboard")
    print("=" * 60)
    print("🚀 Starting demo setup...")
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scikit-learn', 'tensorflow', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📥 Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', 'pip'
            ])
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '-r', 'requirements.txt'
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    return True

def generate_sample_data():
    """Generate sample IoT data"""
    print("📊 Generating sample IoT data...")
    
    try:
        # Add src to path
        sys.path.append('src')
        
        from data_generator import create_sample_dataset
        
        # Generate data
        df = create_sample_dataset()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save data
        df.to_csv('data/iot_sensor_data.csv', index=False)
        
        print(f"✅ Generated {len(df)} records for {df['device_id'].nunique()} devices")
        print(f"📈 Anomaly rate: {df['anomaly'].mean():.1%}")
        print(f"🔧 Maintenance required: {df['maintenance_required'].sum()} instances")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Streamlit dashboard...")
    print("🌐 Dashboard will open at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'run_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main demo function"""
    print_banner()
    
    # Check if we're in the right directory
    if not os.path.exists('src') or not os.path.exists('requirements.txt'):
        print("❌ Please run this script from the project root directory")
        print("   Make sure you're in the 'iot-predictive-maintenance' folder")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    print()
    
    # Generate sample data
    if not generate_sample_data():
        print("❌ Failed to generate sample data")
        return
    
    print()
    
    # Run dashboard
    run_dashboard()

if __name__ == "__main__":
    main()
