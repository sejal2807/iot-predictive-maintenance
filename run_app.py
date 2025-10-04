"""
Simple IoT Dashboard
Basic version for demo purposes
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
from data_generator import IoTDataGenerator

# Page configuration
st.set_page_config(
    page_title="IoT Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

# Header
st.title("🔧 IoT Predictive Maintenance Dashboard")
st.markdown("---")

# Generate realistic industrial IoT data
def generate_data():
    # Use the advanced data generator for realistic patterns
    generator = IoTDataGenerator(seed=42)
    
    # Generate data for multiple devices over the last 7 days
    start_time = datetime.now() - timedelta(days=7)
    device_ids = ['device_001', 'device_002', 'device_003', 'device_004', 'device_005']
    
    # Get the full dataset
    full_data = generator.generate_multi_device_data(
        device_ids=device_ids,
        start_time=start_time,
        duration_hours=168  # 7 days
    )
    
    # Get the last 24 hours for dashboard display
    last_24h = datetime.now() - timedelta(hours=24)
    recent_data = full_data[full_data['timestamp'] >= last_24h].copy()
    
    # Convert to dashboard format
    dashboard_data = []
    for _, row in recent_data.iterrows():
        dashboard_data.append({
            'time': row['timestamp'].strftime('%H:%M'),
            'temp': round(row['temperature'], 1),
            'vib': round(row['vibration'], 2),
            'press': round(row['pressure'], 1),
            'curr': round(row['current'], 1),
            'hum': round(row['humidity'], 1),
            'anomaly': row['is_anomaly'],
            'device': row['device_id'],
            'device_type': row.get('device_type', 'motor')
        })
    
    return dashboard_data

# Load data
data = generate_data()

# Metrics
col1, col2, col3, col4 = st.columns(4)

# Calculate realistic metrics
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

# Device breakdown
st.subheader("🏭 Industrial Equipment Status")
device_col1, device_col2 = st.columns(2)

with device_col1:
    st.write("**Equipment Types:**")
    for device_type in device_types:
        count = len([d for d in data if d['device_type'] == device_type])
        st.write(f"• {device_type.title()}: {count} units")

with device_col2:
    st.write("**Device Performance:**")
    for device in set(d['device'] for d in data):
        device_data = [d for d in data if d['device'] == device]
        device_anomalies = sum(1 for d in device_data if d['anomaly'])
        device_rate = device_anomalies / len(device_data) if len(device_data) > 0 else 0
        status = "🔴 Critical" if device_rate > 0.1 else "🟡 Warning" if device_rate > 0.05 else "🟢 Normal"
        st.write(f"• {device}: {status} ({device_rate:.1%} anomalies)")

st.markdown("---")

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
    
    # Group anomalies by device
    anomaly_by_device = {}
    for anomaly in anomaly_data:
        device = anomaly['device']
        if device not in anomaly_by_device:
            anomaly_by_device[device] = []
        anomaly_by_device[device].append(anomaly)
    
    for device, device_anomalies in anomaly_by_device.items():
        st.error(f"**{device}** ({device_anomalies[0]['device_type'].title()}): {len(device_anomalies)} anomalies")
        for i, anomaly in enumerate(device_anomalies[:3]):  # Show first 3
            st.write(f"  • {anomaly['time']}: Temp {anomaly['temp']}°C, Vib {anomaly['vib']} mm/s, Press {anomaly['press']} bar")
        if len(device_anomalies) > 3:
            st.write(f"  ... and {len(device_anomalies) - 3} more")
else:
    st.success("✅ No anomalies detected - All systems operating normally")

st.markdown("---")

# Maintenance
st.subheader("🔧 Predictive Maintenance Recommendations")

recommendations = []
maintenance_priority = {"Critical": [], "High": [], "Medium": [], "Low": []}

for d in data:
    device = d['device']
    device_type = d['device_type']
    time = d['time']
    
    # Critical issues
    if d['temp'] > 50:
        maintenance_priority["Critical"].append(f"🌡️ **{device}** ({device_type}): Critical temperature {d['temp']}°C at {time}")
    elif d['vib'] > 6:
        maintenance_priority["Critical"].append(f"📳 **{device}** ({device_type}): Critical vibration {d['vib']} mm/s at {time}")
    elif d['press'] > 18:
        maintenance_priority["Critical"].append(f"🔧 **{device}** ({device_type}): Critical pressure {d['press']} bar at {time}")
    
    # High priority
    elif d['temp'] > 40:
        maintenance_priority["High"].append(f"🌡️ **{device}** ({device_type}): Elevated temperature {d['temp']}°C at {time}")
    elif d['vib'] > 4:
        maintenance_priority["High"].append(f"📳 **{device}** ({device_type}): High vibration {d['vib']} mm/s at {time}")
    elif d['press'] > 15:
        maintenance_priority["High"].append(f"🔧 **{device}** ({device_type}): High pressure {d['press']} bar at {time}")
    
    # Medium priority
    elif d['temp'] > 35:
        maintenance_priority["Medium"].append(f"🌡️ **{device}** ({device_type}): Monitor temperature {d['temp']}°C at {time}")
    elif d['vib'] > 3:
        maintenance_priority["Medium"].append(f"📳 **{device}** ({device_type}): Monitor vibration {d['vib']} mm/s at {time}")

# Display recommendations by priority
for priority, recs in maintenance_priority.items():
    if recs:
        if priority == "Critical":
            st.error(f"🚨 **{priority} Priority:**")
        elif priority == "High":
            st.warning(f"⚠️ **{priority} Priority:**")
        elif priority == "Medium":
            st.info(f"ℹ️ **{priority} Priority:**")
        else:
            st.write(f"📝 **{priority} Priority:**")
        
        for rec in recs[:5]:  # Show max 5 per priority
            st.write(rec)
        if len(recs) > 5:
            st.write(f"... and {len(recs) - 5} more {priority.lower()} priority items")

if not any(maintenance_priority.values()):
    st.success("✅ All systems operating within normal parameters - No maintenance required")

st.markdown("---")

# Export
st.subheader("📤 Export Data")

if st.button("📊 Download Report"):
    csv_data = "Time,Temperature,Vibration,Pressure,Current,Humidity,Anomaly\n"
    for d in data:
        csv_data += f"{d['time']},{d['temp']},{d['vib']},{d['press']},{d['curr']},{d['hum']},{d['anomaly']}\n"
    
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"iot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ for IoT Predictive Maintenance**")