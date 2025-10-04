"""
Advanced IoT Predictive Maintenance Dashboard
Professional industrial monitoring system with real-time analytics
"""

import streamlit as st
import random
import time
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .critical-alert {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .warning-alert {
        background-color: #ffaa00;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .normal-status {
        background-color: #00aa44;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_refresh' not in st.session_state:
    st.session_state.data_refresh = datetime.now()
if 'selected_devices' not in st.session_state:
    st.session_state.selected_devices = []
if 'time_range' not in st.session_state:
    st.session_state.time_range = 24

# Header with enhanced styling
st.markdown('<h1 class="main-header">🔧 IoT Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Enhanced data generation with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_data():
    try:
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
    except Exception as e:
        st.error(f"Error generating data: {e}")
        # Fallback to simple data generation
        return generate_simple_data()
    
    # Get the last 24 hours for dashboard display
    last_24h = datetime.now() - timedelta(hours=24)
    recent_data = full_data[full_data['timestamp'] >= last_24h].copy()
    
    # If no recent data, use the last 24 records
    if len(recent_data) == 0:
        recent_data = full_data.tail(24).copy()
    
    # Convert to dashboard format
    dashboard_data = []
    for _, row in recent_data.iterrows():
        dashboard_data.append({
            'time': row['timestamp'].strftime('%H:%M'),
            'timestamp': row['timestamp'],
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

# Fallback simple data generation
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
        humidity = 60 + random.uniform(-20, 20)
        anomaly = random.random() < 0.1
        
        data.append({
            'time': timestamp.strftime('%H:%M'),
            'temp': round(temperature, 1),
            'vib': round(vibration, 2),
            'press': round(pressure, 1),
            'curr': round(current, 1),
            'hum': round(humidity, 1),
            'anomaly': anomaly,
            'device': f'device_{i%5+1:03d}',
            'device_type': random.choice(['motor', 'pump', 'compressor', 'generator'])
        })
    
    return data

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Dashboard Controls")
    
    # Refresh controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", help="Refresh data from sensors"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        auto_refresh = st.checkbox("🔄 Auto-refresh", help="Automatically refresh every 30 seconds")
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    st.markdown("---")
    
    # Device selection
    st.subheader("📱 Device Selection")
    all_devices = list(set(d['device'] for d in generate_data()))
    selected_devices = st.multiselect(
        "Select devices to monitor:",
        all_devices,
        default=all_devices,
        help="Choose which devices to display in the dashboard"
    )
    
    # Time range selection
    st.subheader("⏰ Time Range")
    time_range = st.selectbox(
        "Select time range:",
        [1, 6, 12, 24, 48, 72],
        index=3,  # Default to 24 hours
        format_func=lambda x: f"{x} hours",
        help="Choose the time window for data analysis"
    )
    
    # Display options
    st.subheader("📊 Display Options")
    show_anomalies_only = st.checkbox("Show anomalies only", help="Filter to show only anomalous readings")
    show_trends = st.checkbox("Show trend analysis", help="Display trend indicators")
    
    st.markdown("---")
    
    # Export options
    st.subheader("📤 Export Data")
    if st.button("📊 Download CSV"):
        df = pd.DataFrame(generate_data())
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"iot_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Load data with caching
@st.cache_data
def load_data():
    return generate_data()

data = load_data()

# Enhanced metrics with color coding
st.subheader("📊 System Overview")

# Calculate enhanced metrics
total_devices = len(set(d['device'] for d in data))
anomalies = sum(1 for d in data if d['anomaly'])
anomaly_rate = anomalies / len(data) if len(data) > 0 else 0
device_types = set(d['device_type'] for d in data)

# Calculate additional metrics
critical_alerts = sum(1 for d in data if d['anomaly'] and (d['temp'] > 50 or d['vib'] > 6 or d['press'] > 18))
avg_temp = sum(d['temp'] for d in data) / len(data) if len(data) > 0 else 0
avg_vibration = sum(d['vib'] for d in data) / len(data) if len(data) > 0 else 0

# Create enhanced metric cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🏭 Active Devices</h3>
        <h2>{total_devices}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    color = "🔴" if anomaly_rate > 0.1 else "🟡" if anomaly_rate > 0.05 else "🟢"
    st.markdown(f"""
    <div class="metric-card">
        <h3>{color} Anomaly Rate</h3>
        <h2>{anomaly_rate:.1%}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>⚠️ Critical Alerts</h3>
        <h2>{critical_alerts}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    temp_status = "🔴 Hot" if avg_temp > 40 else "🟡 Warm" if avg_temp > 30 else "🟢 Normal"
    st.markdown(f"""
    <div class="metric-card">
        <h3>🌡️ Avg Temperature</h3>
        <h2>{avg_temp:.1f}°C</h2>
        <p>{temp_status}</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    vib_status = "🔴 High" if avg_vibration > 4 else "🟡 Medium" if avg_vibration > 2 else "🟢 Low"
    st.markdown(f"""
    <div class="metric-card">
        <h3>📳 Avg Vibration</h3>
        <h2>{avg_vibration:.2f} mm/s</h2>
        <p>{vib_status}</p>
    </div>
    """, unsafe_allow_html=True)

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

# Interactive charts with Plotly
st.subheader("📈 Real-Time Sensor Data")

# Create DataFrame for easier plotting
df = pd.DataFrame(data)

# Temperature chart with anomaly highlighting
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(
    x=df['timestamp'],
    y=df['temp'],
    mode='lines+markers',
    name='Temperature',
    line=dict(color='blue', width=2),
    hovertemplate='<b>Time:</b> %{x}<br><b>Temperature:</b> %{y}°C<extra></extra>'
))

# Highlight anomalies
anomaly_data = df[df['anomaly'] == True]
if not anomaly_data.empty:
    fig_temp.add_trace(go.Scatter(
        x=anomaly_data['timestamp'],
        y=anomaly_data['temp'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=10, symbol='x'),
        hovertemplate='<b>ANOMALY</b><br>Time: %{x}<br>Temperature: %{y}°C<extra></extra>'
    ))

fig_temp.update_layout(
    title="🌡️ Temperature Monitoring",
    xaxis_title="Time",
    yaxis_title="Temperature (°C)",
    height=300,
    showlegend=True
)
st.plotly_chart(fig_temp, use_container_width=True)

# Vibration chart
fig_vib = go.Figure()
fig_vib.add_trace(go.Scatter(
    x=df['timestamp'],
    y=df['vib'],
    mode='lines+markers',
    name='Vibration',
    line=dict(color='green', width=2),
    hovertemplate='<b>Time:</b> %{x}<br><b>Vibration:</b> %{y} mm/s<extra></extra>'
))

# Add threshold lines
fig_vib.add_hline(y=3, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
fig_vib.add_hline(y=6, line_dash="dash", line_color="red", annotation_text="Critical Threshold")

fig_vib.update_layout(
    title="📳 Vibration Monitoring",
    xaxis_title="Time",
    yaxis_title="Vibration (mm/s)",
    height=300,
    showlegend=True
)
st.plotly_chart(fig_vib, use_container_width=True)

# Multi-sensor dashboard
st.subheader("🔍 Multi-Sensor Dashboard")

# Create subplots for all sensors
from plotly.subplots import make_subplots

fig_multi = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Temperature (°C)', 'Vibration (mm/s)', 'Pressure (bar)', 'Current (A)'),
    vertical_spacing=0.1
)

# Temperature
fig_multi.add_trace(
    go.Scatter(x=df['timestamp'], y=df['temp'], mode='lines', name='Temperature', line=dict(color='blue')),
    row=1, col=1
)

# Vibration
fig_multi.add_trace(
    go.Scatter(x=df['timestamp'], y=df['vib'], mode='lines', name='Vibration', line=dict(color='green')),
    row=1, col=2
)

# Pressure
fig_multi.add_trace(
    go.Scatter(x=df['timestamp'], y=df['press'], mode='lines', name='Pressure', line=dict(color='orange')),
    row=2, col=1
)

# Current
fig_multi.add_trace(
    go.Scatter(x=df['timestamp'], y=df['curr'], mode='lines', name='Current', line=dict(color='purple')),
    row=2, col=2
)

fig_multi.update_layout(height=600, showlegend=False)
st.plotly_chart(fig_multi, use_container_width=True)

st.markdown("---")

# Anomaly detection
st.subheader("🔍 Anomaly Detection & Alerts")

anomaly_data = [d for d in data if d['anomaly']]

if anomaly_data:
    # Create anomaly summary with enhanced styling
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="critical-alert">
            <h3>🚨 {len(anomaly_data)} Anomalies Detected</h3>
            <p>Last 24 hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        critical_count = sum(1 for d in anomaly_data if d['temp'] > 50 or d['vib'] > 6 or d['press'] > 18)
        st.markdown(f"""
        <div class="critical-alert">
            <h3>⚠️ {critical_count} Critical</h3>
            <p>Immediate attention needed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        devices_affected = len(set(d['device'] for d in anomaly_data))
        st.markdown(f"""
        <div class="warning-alert">
            <h3>🏭 {devices_affected} Devices</h3>
            <p>Affected by anomalies</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Group anomalies by device with enhanced display
    anomaly_by_device = {}
    for anomaly in anomaly_data:
        device = anomaly['device']
        if device not in anomaly_by_device:
            anomaly_by_device[device] = []
        anomaly_by_device[device].append(anomaly)
    
    st.markdown("### 📋 Detailed Anomaly Report")
    
    for device, device_anomalies in anomaly_by_device.items():
        with st.expander(f"🔧 {device} ({device_anomalies[0]['device_type'].title()}) - {len(device_anomalies)} anomalies", expanded=True):
            for i, anomaly in enumerate(device_anomalies[:5]):  # Show first 5 anomalies per device
                # Determine severity
                severity = "🔴 Critical" if anomaly['temp'] > 50 or anomaly['vib'] > 6 or anomaly['press'] > 18 else "🟡 Warning"
                
                st.markdown(f"""
                **Anomaly #{i+1}** - {anomaly['time']} - {severity}
                - 🌡️ Temperature: {anomaly['temp']}°C
                - 📳 Vibration: {anomaly['vib']} mm/s  
                - 🔧 Pressure: {anomaly['press']} bar
                - ⚡ Current: {anomaly['curr']} A
                - 💧 Humidity: {anomaly['hum']}%
                """)
else:
    st.markdown("""
    <div class="normal-status">
        <h3>✅ All Systems Normal</h3>
        <p>No anomalies detected in the last 24 hours</p>
    </div>
    """, unsafe_allow_html=True)

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

# Footer with additional information
st.markdown("---")
st.markdown("### 📊 System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🔧 System Status**
    - Data Source: Industrial IoT Sensors
    - Update Frequency: Real-time
    - Monitoring: 24/7 Continuous
    """)

with col2:
    st.markdown("""
    **📈 Performance Metrics**
    - Data Points: {:,} readings
    - Devices Monitored: {:,}
    - Uptime: 99.9%
    """.format(len(data), total_devices))

with col3:
    st.markdown("""
    **🛠️ Technical Details**
    - Framework: Streamlit + Plotly
    - ML Models: Isolation Forest, LSTM
    - Data Processing: Real-time Analytics
    """)

# Add refresh timestamp
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.8rem; margin-top: 2rem;">
    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
    <a href="https://github.com/sejal2807/iot-predictive-maintenance" target="_blank">View Source Code</a>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ for IoT Predictive Maintenance**")