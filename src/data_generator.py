"""
IoT Sensor Data Generator for Predictive Maintenance
Simulates realistic sensor data with anomalies for testing ML models
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class IoTDataGenerator:
    """Generate synthetic IoT sensor data with realistic patterns and anomalies"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_normal_data(self, 
                           start_time: datetime, 
                           duration_hours: int = 24,
                           sampling_rate: int = 60) -> pd.DataFrame:
        """Generate normal sensor readings with realistic patterns"""
        
        # Create time series
        end_time = start_time + timedelta(hours=duration_hours)
        timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{sampling_rate}S')
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            # Base patterns for different sensors
            hour = timestamp.hour
            
            # Temperature (Celsius) - daily cycle with some noise
            temp_base = 25 + 10 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
            temperature = temp_base + np.random.normal(0, 2)
            
            # Vibration (mm/s) - higher during "working hours"
            if 8 <= hour <= 18:
                vibration_base = 2.5 + np.random.normal(0, 0.5)
            else:
                vibration_base = 1.0 + np.random.normal(0, 0.3)
            vibration = max(0, vibration_base)
            
            # Pressure (bar) - gradual increase over time (wear pattern)
            pressure_base = 10 + (i / len(timestamps)) * 2  # Gradual increase
            pressure = pressure_base + np.random.normal(0, 0.5)
            
            # Current (A) - varies with load
            current_base = 15 + 5 * np.sin(2 * np.pi * hour / 12)  # 12-hour cycle
            current = current_base + np.random.normal(0, 1)
            
            # Humidity (%) - inverse relationship with temperature
            humidity = 60 - (temperature - 25) * 2 + np.random.normal(0, 5)
            humidity = max(0, min(100, humidity))
            
            data.append({
                'timestamp': timestamp,
                'temperature': round(temperature, 2),
                'vibration': round(vibration, 3),
                'pressure': round(pressure, 2),
                'current': round(current, 2),
                'humidity': round(humidity, 1),
                'anomaly': False
            })
        
        return pd.DataFrame(data)
    
    def inject_anomalies(self, df: pd.DataFrame, anomaly_rate: float = 0.05) -> pd.DataFrame:
        """Inject various types of anomalies into the data"""
        
        df = df.copy()
        n_anomalies = int(len(df) * anomaly_rate)
        anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'drift', 'noise', 'level_shift'])
            
            if anomaly_type == 'spike':
                # Sudden spike in one or more sensors
                sensor = np.random.choice(['temperature', 'vibration', 'pressure', 'current'])
                df.loc[idx, sensor] *= np.random.uniform(2, 5)
                
            elif anomaly_type == 'drift':
                # Gradual drift starting from this point
                drift_duration = min(10, len(df) - idx)
                sensor = np.random.choice(['temperature', 'vibration', 'pressure'])
                drift_factor = np.random.uniform(1.5, 3)
                
                for i in range(drift_duration):
                    if idx + i < len(df):
                        df.loc[idx + i, sensor] *= (1 + (i / drift_duration) * (drift_factor - 1))
                        
            elif anomaly_type == 'noise':
                # High noise in multiple sensors
                for sensor in ['temperature', 'vibration', 'pressure', 'current']:
                    noise_factor = np.random.uniform(3, 8)
                    df.loc[idx, sensor] += np.random.normal(0, noise_factor)
                    
            elif anomaly_type == 'level_shift':
                # Permanent shift in sensor readings
                sensor = np.random.choice(['temperature', 'vibration', 'pressure', 'current'])
                shift_amount = np.random.uniform(1.5, 3)
                df.loc[idx:, sensor] *= shift_amount
            
            df.loc[idx, 'anomaly'] = True
        
        return df
    
    def generate_device_data(self, 
                           device_id: str,
                           start_time: datetime,
                           duration_hours: int = 168,  # 1 week
                           anomaly_rate: float = 0.03) -> pd.DataFrame:
        """Generate complete dataset for a single device"""
        
        # Generate normal data
        df = self.generate_normal_data(start_time, duration_hours)
        
        # Inject anomalies
        df = self.inject_anomalies(df, anomaly_rate)
        
        # Add device metadata
        df['device_id'] = device_id
        df['device_type'] = np.random.choice(['motor', 'pump', 'compressor', 'generator'])
        df['location'] = np.random.choice(['plant_a', 'plant_b', 'warehouse', 'office'])
        
        return df
    
    def generate_multi_device_data(self, 
                                 device_ids: List[str],
                                 start_time: datetime,
                                 duration_hours: int = 168) -> pd.DataFrame:
        """Generate data for multiple devices"""
        
        all_data = []
        
        for device_id in device_ids:
            # Different anomaly rates for different devices
            anomaly_rate = np.random.uniform(0.01, 0.08)
            device_data = self.generate_device_data(
                device_id, start_time, duration_hours, anomaly_rate
            )
            all_data.append(device_data)
        
        return pd.concat(all_data, ignore_index=True)
    
    def create_maintenance_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add maintenance events based on anomaly patterns"""
        
        df = df.copy()
        df['maintenance_required'] = False
        df['maintenance_priority'] = 'none'
        
        # Group by device and find anomaly clusters
        for device_id in df['device_id'].unique():
            device_data = df[df['device_id'] == device_id].copy()
            
            # Find consecutive anomalies
            anomaly_mask = device_data['anomaly']
            anomaly_groups = (anomaly_mask != anomaly_mask.shift()).cumsum()
            
            for group_id in anomaly_groups.unique():
                group_mask = anomaly_groups == group_id
                if device_data.loc[group_mask, 'anomaly'].any():
                    group_size = group_mask.sum()
                    
                    if group_size >= 3:  # 3+ consecutive anomalies
                        df.loc[device_data.index[group_mask], 'maintenance_required'] = True
                        
                        # Determine priority based on sensor values
                        max_temp = device_data.loc[group_mask, 'temperature'].max()
                        max_vibration = device_data.loc[group_mask, 'vibration'].max()
                        
                        if max_temp > 50 or max_vibration > 5:
                            priority = 'critical'
                        elif max_temp > 40 or max_vibration > 3:
                            priority = 'high'
                        else:
                            priority = 'medium'
                        
                        df.loc[device_data.index[group_mask], 'maintenance_priority'] = priority
        
        return df

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    
    generator = IoTDataGenerator()
    
    # Generate data for 5 devices over 1 week
    device_ids = [f'device_{i:03d}' for i in range(1, 6)]
    start_time = datetime.now() - timedelta(days=7)
    
    df = generator.generate_multi_device_data(device_ids, start_time, 168)
    df = generator.create_maintenance_events(df)
    
    return df

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample IoT dataset...")
    df = create_sample_dataset()
    
    print(f"Generated {len(df)} records for {df['device_id'].nunique()} devices")
    print(f"Anomalies detected: {df['anomaly'].sum()}")
    print(f"Maintenance required: {df['maintenance_required'].sum()}")
    
    # Save to CSV
    df.to_csv('data/iot_sensor_data.csv', index=False)
    print("Data saved to data/iot_sensor_data.csv")

