"""
Simple IoT Predictive Maintenance Dashboard
Minimal version that works without external dependencies
"""

import streamlit as st
import random
import time
import math
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration removed - handled by main entry point

# Header
st.title("🔧 IoT Predictive Maintenance Dashboard")
st.markdown("---")

# Simple data generation
def generate_simple_data():
    random.seed(42)
    data = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(24):
        timestamp = base_time + timedelta(hours=i)
        temperature = 25 + random.uniform(-5, 15)
        vibration = 2 + random.uniform(-1, 3)
        pressure = 10 + random.uniform(-2, 4)
        current = 15 + random.uniform(-5, 10)
        humidity = 50 + random.uniform(-20, 20)
        
        # Add some anomalies
        is_anomaly = random.random() < 0.1  # 10% chance of anomaly
        if is_anomaly:
            temperature += random.uniform(10, 20)
            vibration += random.uniform(2, 4)
        
        data.append({
            'time': timestamp.strftime('%H:%M'),
            'timestamp': timestamp,
            'temp': round(temperature, 1),
            'vib': round(vibration, 2),
            'press': round(pressure, 1),
            'curr': round(current, 1),
            'hum': round(humidity, 1),
            'anomaly': is_anomaly,
            'device': f'device_{i%5+1:03d}',
            'device_type': random.choice(['motor', 'pump', 'compressor', 'generator'])
        })
    
    return data

# Load data
data = generate_simple_data()

# Metrics
col1, col2, col3, col4 = st.columns(4)

total_devices = len(set(d['device'] for d in data))
anomalies = sum(1 for d in data if d['anomaly'])
anomaly_rate = anomalies / len(data) if len(data) > 0 else 0
device_types = set(d['device_type'] for d in data)

with col1:
    st.metric("Active Devices", total_devices)
with col2:
    st.metric("Anomalies (24h)", anomalies)
with col3:
    st.metric("Anomaly Rate", f"{anomaly_rate:.1%}")
with col4:
    st.metric("Device Types", len(device_types))

# Charts
st.subheader("📈 Real-Time Sensor Data")

# Temperature
st.subheader("Temperature (°C)")
temp_values = [d['temp'] for d in data]
st.line_chart(temp_values)

# Vibration
st.subheader("Vibration (mm/s)")
vib_values = [d['vib'] for d in data]
st.line_chart(vib_values)

# Pressure
st.subheader("Pressure (bar)")
press_values = [d['press'] for d in data]
st.line_chart(press_values)

# Current
st.subheader("Current (A)")
curr_values = [d['curr'] for d in data]
st.line_chart(curr_values)

# Humidity
st.subheader("Humidity (%)")
hum_values = [d['hum'] for d in data]
st.line_chart(hum_values)

st.markdown("---")

# Anomaly detection
st.subheader("🔍 Anomaly Detection & Alerts")

anomaly_data = [d for d in data if d['anomaly']]

if anomaly_data:
    st.warning(f"🚨 {len(anomaly_data)} anomalies detected in the last 24 hours!")
    
    for i, anomaly in enumerate(anomaly_data[:5]):
        st.write(f"**Anomaly #{i+1}** - {anomaly['time']}: Temp {anomaly['temp']}°C, Vib {anomaly['vib']} mm/s")
else:
    st.success("✅ No anomalies detected - All systems operating normally")

st.markdown("---")

# Maintenance recommendations
st.subheader("🔧 Predictive Maintenance Recommendations")

recommendations = []
for d in data:
    if d['temp'] > 40:
        recommendations.append(f"🌡️ **{d['device']}**: High temperature {d['temp']}°C at {d['time']}")
    elif d['vib'] > 4:
        recommendations.append(f"📳 **{d['device']}**: High vibration {d['vib']} mm/s at {d['time']}")
    elif d['press'] > 15:
        recommendations.append(f"🔧 **{d['device']}**: High pressure {d['press']} bar at {d['time']}")

if recommendations:
    for rec in recommendations[:5]:
        st.write(rec)
else:
    st.success("✅ All systems operating within normal parameters")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ for IoT Predictive Maintenance**")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
