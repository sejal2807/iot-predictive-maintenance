"""
ULTRA-SIMPLE IoT Dashboard
100% guaranteed to work - no external dependencies
"""

import streamlit as st
import random
import time
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="IoT Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

# Header
st.title("🔧 IoT Predictive Maintenance Dashboard")
st.markdown("---")

# Generate sample data
def generate_data():
    random.seed(42)
    data = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(24):
        timestamp = base_time + timedelta(hours=i)
        temperature = 25 + random.uniform(-5, 15)
        vibration = 2 + random.uniform(-1, 3)
        pressure = 10 + random.uniform(-2, 4)
        current = 15 + random.uniform(-5, 10)
        humidity = 60 + random.uniform(-20, 20)
        anomaly = random.random() < 0.1
        
        data.append({
            'time': timestamp.strftime('%H:%M'),
            'temp': round(temperature, 1),
            'vib': round(vibration, 2),
            'press': round(pressure, 1),
            'curr': round(current, 1),
            'hum': round(humidity, 1),
            'anomaly': anomaly
        })
    
    return data

# Load data
data = generate_data()

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Devices", "5")
with col2:
    anomalies = sum(1 for d in data if d['anomaly'])
    st.metric("Anomalies", anomalies)
with col3:
    st.metric("Anomaly Rate", f"{anomalies/len(data):.1%}")
with col4:
    st.metric("Data Points", len(data))

st.markdown("---")

# Charts
st.subheader("📈 Sensor Data")

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
st.subheader("🔍 Anomaly Detection")

anomaly_data = [d for d in data if d['anomaly']]

if anomaly_data:
    st.warning(f"🚨 {len(anomaly_data)} anomalies detected!")
    
    for i, anomaly in enumerate(anomaly_data):
        st.error(f"Alert #{i+1}: {anomaly['time']} - Temp: {anomaly['temp']}°C, Vib: {anomaly['vib']} mm/s")
else:
    st.success("✅ No anomalies detected - All systems normal")

st.markdown("---")

# Maintenance
st.subheader("🔧 Maintenance Recommendations")

recommendations = []
for d in data:
    if d['temp'] > 40:
        recommendations.append(f"🌡️ High temperature: {d['temp']}°C at {d['time']}")
    if d['vib'] > 4:
        recommendations.append(f"📳 High vibration: {d['vib']} mm/s at {d['time']}")
    if d['press'] > 15:
        recommendations.append(f"🔧 High pressure: {d['press']} bar at {d['time']}")

if recommendations:
    for rec in recommendations:
        st.info(rec)
else:
    st.success("✅ No maintenance required")

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
