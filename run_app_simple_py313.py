"""
Simple IoT Predictive Maintenance Dashboard
Python 3.13+ Compatible - Minimal Dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta

# Page configuration - ONLY ONE CALL
st.set_page_config(
    page_title="🔧 IoT Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .status-normal {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .status-critical {
        background: linear-gradient(135deg, #F44336 0%, #D32F2F 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #4CAF50;
        border-radius: 50%;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_refresh' not in st.session_state:
    st.session_state.data_refresh = datetime.now()
if 'live_data' not in st.session_state:
    st.session_state.live_data = []
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False

# Enhanced data generation
@st.cache_data(ttl=60)
def generate_iot_data():
    """Generate realistic IoT sensor data"""
    np.random.seed(42)
    random.seed(42)
    
    base_time = datetime.now() - timedelta(hours=24)
    data = []
    
    device_types = ['Motor', 'Pump', 'Compressor', 'Generator', 'Turbine']
    device_locations = ['Plant A', 'Plant B', 'Warehouse', 'Office', 'Factory']
    
    for i in range(24):
        timestamp = base_time + timedelta(hours=i)
        hour = timestamp.hour
        
        for device_id in range(1, 6):
            device_type = device_types[device_id - 1]
            location = device_locations[device_id - 1]
            
            # Realistic sensor patterns
            temp_base = 25 + 10 * np.sin(2 * np.pi * hour / 24)
            if device_type == 'Motor':
                temp_base += 5 + np.random.normal(0, 2)
            elif device_type == 'Compressor':
                temp_base += 8 + np.random.normal(0, 3)
            else:
                temp_base += np.random.normal(0, 1.5)
            
            # Vibration with working hours pattern
            if 8 <= hour <= 18:
                vib_base = 2.5 + np.random.normal(0, 0.8)
            else:
                vib_base = 1.0 + np.random.normal(0, 0.3)
            
            # Pressure with gradual wear
            pressure_base = 10 + (i / 24) * 3 + np.random.normal(0, 0.5)
            
            # Current with load variations
            current_base = 15 + 5 * np.sin(2 * np.pi * hour / 12) + np.random.normal(0, 1)
            
            # Humidity
            humidity = 60 - (temp_base - 25) * 2 + np.random.normal(0, 5)
            humidity = max(0, min(100, humidity))
            
            # Add anomalies (10% chance)
            is_anomaly = random.random() < 0.1
            if is_anomaly:
                temp_base += random.uniform(10, 25)
                vib_base += random.uniform(2, 5)
                pressure_base += random.uniform(3, 8)
                current_base += random.uniform(5, 15)
            
            # Calculate health score
            health_score = max(0, min(100, 100 - (
                (temp_base - 25) * 2 + 
                vib_base * 10 + 
                (pressure_base - 10) * 3 + 
                (current_base - 15) * 2
            )))
            
            data.append({
                'timestamp': timestamp,
                'device_id': f'device_{device_id:03d}',
                'device_type': device_type,
                'location': location,
                'temperature': round(temp_base, 1),
                'vibration': round(vib_base, 2),
                'pressure': round(pressure_base, 1),
                'current': round(current_base, 1),
                'humidity': round(humidity, 1),
                'health_score': round(health_score, 1),
                'is_anomaly': is_anomaly,
                'status': 'Critical' if health_score < 30 else 'Warning' if health_score < 70 else 'Normal'
            })
    
    return data

# Real-time simulation
def simulate_live_data():
    """Simulate live data updates"""
    if st.session_state.simulation_running:
        current_time = datetime.now()
        device_id = random.randint(1, 5)
        device_types = ['Motor', 'Pump', 'Compressor', 'Generator', 'Turbine']
        
        temp = 25 + random.uniform(-5, 15) + random.uniform(-2, 2)
        vib = 2 + random.uniform(-1, 3) + random.uniform(-0.5, 0.5)
        press = 10 + random.uniform(-2, 4) + random.uniform(-1, 1)
        curr = 15 + random.uniform(-5, 10) + random.uniform(-2, 2)
        hum = 60 + random.uniform(-20, 20) + random.uniform(-5, 5)
        
        health_score = max(0, min(100, 100 - (
            (temp - 25) * 2 + 
            vib * 10 + 
            (press - 10) * 3 + 
            (curr - 15) * 2
        )))
        
        new_data = {
            'timestamp': current_time,
            'device_id': f'device_{device_id:03d}',
            'device_type': device_types[device_id - 1],
            'temperature': round(temp, 1),
            'vibration': round(vib, 2),
            'pressure': round(press, 1),
            'current': round(curr, 1),
            'humidity': round(hum, 1),
            'health_score': round(health_score, 1),
            'is_anomaly': health_score < 50,
            'status': 'Critical' if health_score < 30 else 'Warning' if health_score < 70 else 'Normal'
        }
        
        st.session_state.live_data.append(new_data)
        if len(st.session_state.live_data) > 100:
            st.session_state.live_data = st.session_state.live_data[-100:]

# Main header
st.markdown('<h1 class="main-header">🔧 IoT Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)

# Live indicator
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span class="live-indicator"></span>
        <span style="margin-left: 10px; font-weight: bold; color: #4CAF50;">LIVE MONITORING ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    st.subheader("📡 Live Data Simulation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Live"):
            st.session_state.simulation_running = True
            st.success("Live simulation started!")
    
    with col2:
        if st.button("⏹️ Stop Live"):
            st.session_state.simulation_running = False
            st.info("Live simulation stopped!")
    
    auto_refresh = st.checkbox("🔄 Auto-refresh (5s)")
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    st.markdown("---")
    
    st.subheader("📱 Device Selection")
    all_devices = list(set(d['device_id'] for d in generate_iot_data()))
    selected_devices = st.multiselect(
        "Select devices to monitor:",
        all_devices,
        default=all_devices
    )
    
    st.subheader("⏰ Time Range")
    time_range = st.selectbox(
        "Select time range:",
        [1, 6, 12, 24, 48, 72],
        index=3,
        format_func=lambda x: f"{x} hours"
    )

# Simulate live data
simulate_live_data()

# Load data
data = generate_iot_data()
if st.session_state.live_data:
    data.extend(st.session_state.live_data)

# Filter data
if selected_devices:
    data = [d for d in data if d['device_id'] in selected_devices]

# Metrics dashboard
st.subheader("📊 System Overview")

# Calculate metrics
total_devices = len(set(d['device_id'] for d in data))
anomalies = sum(1 for d in data if d['is_anomaly'])
anomaly_rate = anomalies / len(data) if len(data) > 0 else 0
critical_devices = sum(1 for d in data if d['status'] == 'Critical')
avg_health = sum(d['health_score'] for d in data) / len(data) if len(data) > 0 else 0

# Metric cards
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
        <h3>⚠️ Critical Devices</h3>
        <h2>{critical_devices}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    health_color = "🔴" if avg_health < 50 else "🟡" if avg_health < 80 else "🟢"
    st.markdown(f"""
    <div class="metric-card">
        <h3>{health_color} Avg Health</h3>
        <h2>{avg_health:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col5:
    uptime = 99.9 - (anomaly_rate * 10)
    st.markdown(f"""
    <div class="metric-card">
        <h3>⏱️ System Uptime</h3>
        <h2>{uptime:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

# Real-time charts
st.subheader("📈 Real-Time Sensor Monitoring")

df = pd.DataFrame(data)

# Create charts using Streamlit's built-in charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ Temperature (°C)")
    temp_data = df.set_index('timestamp')['temperature']
    st.line_chart(temp_data)
    
    st.subheader("📳 Vibration (mm/s)")
    vib_data = df.set_index('timestamp')['vibration']
    st.line_chart(vib_data)

with col2:
    st.subheader("🔧 Pressure (bar)")
    press_data = df.set_index('timestamp')['pressure']
    st.line_chart(press_data)
    
    st.subheader("⚡ Current (A)")
    curr_data = df.set_index('timestamp')['current']
    st.line_chart(curr_data)

# Device status
st.subheader("🏭 Device Status Overview")

device_data = {}
for d in data:
    device_id = d['device_id']
    if device_id not in device_data:
        device_data[device_id] = []
    device_data[device_id].append(d)

cols = st.columns(len(device_data))
for i, (device_id, device_readings) in enumerate(device_data.items()):
    with cols[i % len(cols)]:
        latest_reading = device_readings[-1]
        status = latest_reading['status']
        health_score = latest_reading['health_score']
        
        card_class = "status-critical" if status == "Critical" else "status-warning" if status == "Warning" else "status-normal"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4>🔧 {device_id}</h4>
            <p><strong>Type:</strong> {latest_reading['device_type']}</p>
            <p><strong>Location:</strong> {latest_reading['location']}</p>
            <p><strong>Health Score:</strong> {health_score:.1f}%</p>
            <p><strong>Status:</strong> {status}</p>
            <p><strong>Temperature:</strong> {latest_reading['temperature']}°C</p>
            <p><strong>Vibration:</strong> {latest_reading['vibration']} mm/s</p>
        </div>
        """, unsafe_allow_html=True)

# Anomaly detection
st.subheader("🔍 Anomaly Detection & Alerts")

anomaly_data = [d for d in data if d['is_anomaly']]

if anomaly_data:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="status-critical">
            <h3>🚨 {len(anomaly_data)} Anomalies Detected</h3>
            <p>Last {time_range} hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        critical_count = sum(1 for d in anomaly_data if d['status'] == 'Critical')
        st.markdown(f"""
        <div class="status-critical">
            <h3>⚠️ {critical_count} Critical</h3>
            <p>Immediate attention needed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        devices_affected = len(set(d['device_id'] for d in anomaly_data))
        st.markdown(f"""
        <div class="status-warning">
            <h3>🏭 {devices_affected} Devices</h3>
            <p>Affected by anomalies</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show anomalies
    st.markdown("### 📋 Anomaly Details")
    for i, anomaly in enumerate(anomaly_data[:10]):
        severity_class = "status-critical" if anomaly['status'] == 'Critical' else "status-warning"
        
        st.markdown(f"""
        <div class="{severity_class}">
            <h4>🚨 Anomaly #{i+1} - {anomaly['device_id']} ({anomaly['device_type']})</h4>
            <p><strong>Time:</strong> {anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Status:</strong> {anomaly['status']}</p>
            <p><strong>Health Score:</strong> {anomaly['health_score']:.1f}%</p>
            <p><strong>Temperature:</strong> {anomaly['temperature']}°C | <strong>Vibration:</strong> {anomaly['vibration']} mm/s</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-normal">
        <h3>✅ All Systems Normal</h3>
        <p>No anomalies detected in the selected time range</p>
    </div>
    """, unsafe_allow_html=True)

# Maintenance recommendations
st.subheader("🔧 Predictive Maintenance Recommendations")

recommendations = []
for device_id, device_readings in device_data.items():
    latest = device_readings[-1]
    avg_health = sum(d['health_score'] for d in device_readings) / len(device_readings)
    
    if latest['status'] == 'Critical':
        recommendations.append(f"🚨 **{device_id}** ({latest['device_type']}): CRITICAL - Health score {latest['health_score']:.1f}% - Immediate maintenance required")
    elif latest['status'] == 'Warning':
        recommendations.append(f"⚠️ **{device_id}** ({latest['device_type']}): WARNING - Health score {latest['health_score']:.1f}% - Schedule maintenance soon")
    elif avg_health < 80:
        recommendations.append(f"ℹ️ **{device_id}** ({latest['device_type']}): Monitor - Average health {avg_health:.1f}% - Consider preventive maintenance")

if recommendations:
    for rec in recommendations[:10]:
        st.info(rec)
else:
    st.success("✅ All devices operating within normal parameters - No maintenance required")

# Footer
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
    st.markdown(f"""
    **📈 Performance Metrics**
    - Data Points: {len(data):,} readings
    - Devices Monitored: {total_devices:,}
    - Uptime: {99.9 - (anomaly_rate * 10):.1f}%
    - Health Score: {avg_health:.1f}%
    """)

with col3:
    st.markdown("""
    **🛠️ Technical Details**
    - Framework: Streamlit
    - Data Processing: Real-time Analytics
    - Visualization: Interactive Charts
    """)

st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.8rem; margin-top: 2rem;">
    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
    Python 3.13+ Compatible Version
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("**Built with ❤️ for IoT Predictive Maintenance**")
