"""
MINIMAL IoT Predictive Maintenance Dashboard
100% guaranteed to work - uses only Streamlit built-in features
"""

import streamlit as st
import random
import time
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="IoT Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-alert {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-alert {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_sensor_data():
    """Generate sample sensor data using only Python built-ins"""
    random.seed(42)
    
    data = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(24):  # 24 hours of data
        timestamp = base_time + timedelta(hours=i)
        
        # Generate realistic sensor readings
        hour = timestamp.hour
        
        # Temperature with daily cycle
        temp_base = 25 + 10 * (1 if 8 <= hour <= 18 else 0.5)
        temperature = temp_base + random.uniform(-3, 3)
        
        # Vibration (higher during working hours)
        vibration = 2 + random.uniform(-1, 1) if 8 <= hour <= 18 else 1 + random.uniform(-0.5, 0.5)
        
        # Pressure (gradual increase)
        pressure = 10 + (i / 24) * 2 + random.uniform(-1, 1)
        
        # Current (varies with load)
        current = 15 + 5 * (1 if 8 <= hour <= 18 else 0.5) + random.uniform(-2, 2)
        
        # Humidity (inverse of temperature)
        humidity = 60 - (temperature - 25) * 2 + random.uniform(-10, 10)
        humidity = max(0, min(100, humidity))
        
        # Anomaly detection (5% chance)
        anomaly = random.random() < 0.05
        if anomaly:
            temperature *= random.uniform(1.5, 3)
            vibration *= random.uniform(2, 4)
        
        data.append({
            'timestamp': timestamp,
            'temperature': round(temperature, 1),
            'vibration': round(vibration, 2),
            'pressure': round(pressure, 1),
            'current': round(current, 1),
            'humidity': round(humidity, 1),
            'anomaly': anomaly,
            'device_id': f'device_{i % 5 + 1:03d}'
        })
    
    return data

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔧 IoT Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Generate sample data
    with st.spinner("Loading IoT sensor data..."):
        data = generate_sensor_data()
    
    # Calculate metrics
    total_devices = len(set(item['device_id'] for item in data))
    total_anomalies = sum(1 for item in data if item['anomaly'])
    anomaly_rate = total_anomalies / len(data)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Devices", total_devices)
    
    with col2:
        st.metric("Total Anomalies", total_anomalies)
    
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.1%}")
    
    with col4:
        st.metric("Data Points", len(data))
    
    # Main dashboard
    tab1, tab2, tab3 = st.tabs(["📈 Real-time Monitoring", "🔍 Anomaly Detection", "🔧 Maintenance"])
    
    with tab1:
        st.subheader("📈 Real-time Sensor Monitoring")
        
        # Create simple charts using Streamlit's built-in charting
        sensor_data = {
            'Temperature': [item['temperature'] for item in data],
            'Vibration': [item['vibration'] for item in data],
            'Pressure': [item['pressure'] for item in data],
            'Current': [item['current'] for item in data],
            'Humidity': [item['humidity'] for item in data]
        }
        
        # Display charts
        for sensor, values in sensor_data.items():
            st.subheader(f"{sensor} Over Time")
            st.line_chart({sensor: values})
        
        # Live updates simulation
        if st.button("🔄 Simulate Live Updates"):
            placeholder = st.empty()
            
            for i in range(5):
                with placeholder.container():
                    new_data = random.choice(data)
                    st.success(f"📡 New data from {new_data['device_id']}")
                    st.json({
                        "timestamp": str(new_data['timestamp']),
                        "temperature": new_data['temperature'],
                        "vibration": new_data['vibration'],
                        "anomaly": new_data['anomaly']
                    })
                time.sleep(1)
    
    with tab2:
        st.subheader("🔍 Anomaly Detection Analysis")
        
        # Anomaly analysis
        anomaly_data = [item for item in data if item['anomaly']]
        
        if anomaly_data:
            st.warning(f"🚨 {len(anomaly_data)} anomalies detected!")
            
            # Show anomaly details
            for i, anomaly in enumerate(anomaly_data):
                st.markdown(f"""
                <div class="anomaly-alert">
                    <h4>🚨 Anomaly #{i+1} - {anomaly['device_id']}</h4>
                    <p><strong>Time:</strong> {anomaly['timestamp']}</p>
                    <p><strong>Temperature:</strong> {anomaly['temperature']}°C</p>
                    <p><strong>Vibration:</strong> {anomaly['vibration']} mm/s</p>
                    <p><strong>Pressure:</strong> {anomaly['pressure']} bar</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-alert">
                <h4>✅ No Anomalies Detected</h4>
                <p>All devices are operating within normal parameters.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Device status table
        st.subheader("Device Status Summary")
        device_status = {}
        for item in data:
            device_id = item['device_id']
            if device_id not in device_status:
                device_status[device_id] = {
                    'anomalies': 0,
                    'avg_temp': 0,
                    'avg_vibration': 0,
                    'count': 0
                }
            
            device_status[device_id]['anomalies'] += 1 if item['anomaly'] else 0
            device_status[device_id]['avg_temp'] += item['temperature']
            device_status[device_id]['avg_vibration'] += item['vibration']
            device_status[device_id]['count'] += 1
        
        # Calculate averages
        for device_id in device_status:
            count = device_status[device_id]['count']
            device_status[device_id]['avg_temp'] = round(device_status[device_id]['avg_temp'] / count, 1)
            device_status[device_id]['avg_vibration'] = round(device_status[device_id]['avg_vibration'] / count, 2)
        
        # Display table
        st.table(device_status)
    
    with tab3:
        st.subheader("🔧 Maintenance Management")
        
        # Maintenance recommendations
        st.subheader("💡 Maintenance Recommendations")
        
        recommendations = []
        
        # Analyze each device
        for device_id in set(item['device_id'] for item in data):
            device_data = [item for item in data if item['device_id'] == device_id]
            avg_temp = sum(item['temperature'] for item in device_data) / len(device_data)
            avg_vibration = sum(item['vibration'] for item in device_data) / len(device_data)
            anomaly_count = sum(1 for item in device_data if item['anomaly'])
            
            if avg_temp > 40:
                recommendations.append(f"🌡️ {device_id}: High temperature ({avg_temp:.1f}°C) - Check cooling system")
            
            if avg_vibration > 3:
                recommendations.append(f"📳 {device_id}: High vibration ({avg_vibration:.2f} mm/s) - Check bearings")
            
            if anomaly_count > 2:
                recommendations.append(f"⚠️ {device_id}: Multiple anomalies ({anomaly_count}) - Schedule inspection")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ No maintenance recommendations at this time.")
        
        # Export functionality
        st.subheader("📤 Export Data")
        
        if st.button("📊 Download Anomaly Report"):
            anomaly_report = [item for item in data if item['anomaly']]
            if anomaly_report:
                # Create CSV content
                csv_content = "timestamp,device_id,temperature,vibration,pressure,current,humidity\n"
                for item in anomaly_report:
                    csv_content += f"{item['timestamp']},{item['device_id']},{item['temperature']},{item['vibration']},{item['pressure']},{item['current']},{item['humidity']}\n"
                
                st.download_button(
                    label="Download CSV",
                    data=csv_content,
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No anomalies to export.")

if __name__ == "__main__":
    main()
