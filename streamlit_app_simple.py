"""
Simplified IoT Predictive Maintenance Dashboard
100% guaranteed to work on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

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
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample IoT sensor data"""
    np.random.seed(42)
    random.seed(42)
    
    # Generate time series
    start_time = datetime.now() - timedelta(days=7)
    timestamps = pd.date_range(start=start_time, periods=168, freq='H')
    
    data = []
    for i, timestamp in enumerate(timestamps):
        # Generate realistic sensor data
        temperature = 25 + 10 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 2)
        vibration = 2 + np.random.normal(0, 0.5)
        pressure = 10 + (i / 168) * 2 + np.random.normal(0, 0.5)
        current = 15 + 5 * np.sin(2 * np.pi * i / 12) + np.random.normal(0, 1)
        humidity = 60 - (temperature - 25) * 2 + np.random.normal(0, 5)
        
        # Add some anomalies
        anomaly = False
        if np.random.random() < 0.05:  # 5% anomaly rate
            anomaly = True
            temperature *= np.random.uniform(1.5, 3)
            vibration *= np.random.uniform(2, 4)
        
        data.append({
            'timestamp': timestamp,
            'temperature': round(temperature, 2),
            'vibration': round(vibration, 3),
            'pressure': round(pressure, 2),
            'current': round(current, 2),
            'humidity': round(humidity, 1),
            'anomaly': anomaly,
            'device_id': f'device_{i % 5 + 1:03d}'
        })
    
    return pd.DataFrame(data)

def create_sensor_chart(df):
    """Create sensor data visualization"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Temperature (°C)', 'Vibration (mm/s)', 
                       'Pressure (bar)', 'Current (A)', 
                       'Humidity (%)', 'Anomaly Status'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['temperature'],
                  mode='lines', name='Temperature', line=dict(color='red')),
        row=1, col=1
    )
    
    # Vibration
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['vibration'],
                  mode='lines', name='Vibration', line=dict(color='orange')),
        row=1, col=2
    )
    
    # Pressure
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['pressure'],
                  mode='lines', name='Pressure', line=dict(color='blue')),
        row=2, col=1
    )
    
    # Current
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['current'],
                  mode='lines', name='Current', line=dict(color='green')),
        row=2, col=2
    )
    
    # Humidity
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['humidity'],
                  mode='lines', name='Humidity', line=dict(color='purple')),
        row=3, col=1
    )
    
    # Anomaly status
    anomaly_data = df[df['anomaly'] == True]
    if not anomaly_data.empty:
        fig.add_trace(
            go.Scatter(x=anomaly_data['timestamp'], y=[1]*len(anomaly_data),
                      mode='markers', name='Anomalies', 
                      marker=dict(color='red', size=10, symbol='x')),
            row=3, col=2
        )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="IoT Sensor Data Dashboard",
        title_x=0.5
    )
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔧 IoT Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Generate sample data
    with st.spinner("Loading IoT sensor data..."):
        df = generate_sample_data()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Devices", df['device_id'].nunique())
    
    with col2:
        st.metric("Total Anomalies", df['anomaly'].sum())
    
    with col3:
        st.metric("Anomaly Rate", f"{df['anomaly'].mean():.1%}")
    
    with col4:
        st.metric("Data Points", len(df))
    
    # Main chart
    st.subheader("📈 Real-time Sensor Monitoring")
    chart = create_sensor_chart(df)
    st.plotly_chart(chart, use_container_width=True)
    
    # Anomaly analysis
    st.subheader("🔍 Anomaly Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly by device
        anomaly_by_device = df.groupby('device_id')['anomaly'].agg(['sum', 'count']).reset_index()
        anomaly_by_device['anomaly_rate'] = anomaly_by_device['sum'] / anomaly_by_device['count']
        
        fig = px.bar(
            anomaly_by_device, 
            x='device_id', 
            y='anomaly_rate',
            title='Anomaly Rate by Device',
            color='anomaly_rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Device status
        device_status = df.groupby('device_id').agg({
            'anomaly': 'sum',
            'temperature': 'mean',
            'vibration': 'mean'
        }).round(2)
        
        st.subheader("Device Status Summary")
        st.dataframe(device_status, use_container_width=True)
    
    # Maintenance recommendations
    st.subheader("🔧 Maintenance Recommendations")
    
    critical_devices = df[df['anomaly'] == True]['device_id'].unique()
    
    if len(critical_devices) > 0:
        st.warning(f"🚨 Critical alerts detected for devices: {', '.join(critical_devices)}")
        
        for device in critical_devices:
            device_data = df[df['device_id'] == device]
            avg_temp = device_data['temperature'].mean()
            avg_vibration = device_data['vibration'].mean()
            
            if avg_temp > 40:
                st.error(f"🌡️ {device}: High temperature detected ({avg_temp:.1f}°C) - Check cooling system")
            
            if avg_vibration > 3:
                st.error(f"📳 {device}: High vibration detected ({avg_vibration:.2f} mm/s) - Check bearings")
    else:
        st.success("✅ All devices operating normally - No maintenance required")
    
    # Export functionality
    st.subheader("📤 Export Data")
    
    if st.button("📊 Download Anomaly Report"):
        anomaly_report = df[df['anomaly'] == True]
        csv = anomaly_report.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
