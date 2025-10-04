"""
Helper functions
Stuff that's useful across the project
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yaml
import os

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file"""
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return {}

def save_config(config: Dict, config_path: str = "config/config.yaml"):
    """Save configuration to YAML file"""
    
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def calculate_anomaly_score(values: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """Calculate anomaly scores using various methods"""
    
    if method == 'zscore':
        mean_val = np.mean(values)
        std_val = np.std(values)
        return np.abs((values - mean_val) / (std_val + 1e-6))
    
    elif method == 'iqr':
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        return np.abs(values - np.median(values)) / (IQR + 1e-6)
    
    elif method == 'percentile':
        return 1 - (np.argsort(np.argsort(values)) / len(values))
    
    else:
        raise ValueError(f"Unknown method: {method}")

def detect_trends(data: pd.Series, window: int = 10) -> Dict[str, float]:
    """Detect trends in time series data"""
    
    # Calculate moving average
    ma = data.rolling(window=window).mean()
    
    # Calculate trend slope
    x = np.arange(len(ma.dropna()))
    y = ma.dropna().values
    
    if len(y) > 1:
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0
    
    # Determine trend direction
    if slope > 0.01:
        trend = 'increasing'
    elif slope < -0.01:
        trend = 'decreasing'
    else:
        trend = 'stable'
    
    return {
        'trend': trend,
        'slope': slope,
        'magnitude': abs(slope)
    }

def calculate_device_health_score(device_data: pd.DataFrame) -> float:
    """Calculate overall health score for a device"""
    
    # Normalize sensor values (0-1 scale)
    temp_norm = min(device_data['temperature'].mean() / 50, 1.0)
    vib_norm = min(device_data['vibration'].mean() / 5, 1.0)
    press_norm = min(device_data['pressure'].mean() / 20, 1.0)
    curr_norm = min(device_data['current'].mean() / 30, 1.0)
    
    # Calculate weighted health score
    health_score = (
        (1 - temp_norm) * 0.3 +  # Temperature (lower is better)
        (1 - vib_norm) * 0.3 +   # Vibration (lower is better)
        (1 - press_norm) * 0.2 + # Pressure (lower is better)
        (1 - curr_norm) * 0.2    # Current (lower is better)
    )
    
    return max(0, min(1, health_score))

def generate_maintenance_recommendations(device_data: pd.DataFrame) -> List[str]:
    """Generate maintenance recommendations based on sensor data"""
    
    recommendations = []
    
    # Temperature analysis
    avg_temp = device_data['temperature'].mean()
    max_temp = device_data['temperature'].max()
    
    if avg_temp > 40:
        recommendations.append("High average temperature detected - check cooling system")
    if max_temp > 50:
        recommendations.append("Critical temperature spike - immediate inspection required")
    
    # Vibration analysis
    avg_vibration = device_data['vibration'].mean()
    max_vibration = device_data['vibration'].max()
    
    if avg_vibration > 3:
        recommendations.append("High vibration levels - check bearings and alignment")
    if max_vibration > 5:
        recommendations.append("Critical vibration detected - possible mechanical failure")
    
    # Pressure analysis
    pressure_trend = detect_trends(device_data['pressure'])
    if pressure_trend['trend'] == 'increasing' and pressure_trend['magnitude'] > 0.1:
        recommendations.append("Increasing pressure trend - check for blockages or leaks")
    
    # Current analysis
    current_variance = device_data['current'].var()
    if current_variance > 10:
        recommendations.append("High current variance - check electrical connections")
    
    # Anomaly analysis
    anomaly_rate = device_data['anomaly'].mean()
    if anomaly_rate > 0.1:
        recommendations.append("High anomaly rate - comprehensive inspection recommended")
    
    return recommendations

def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional time series features"""
    
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_working_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def calculate_correlation_matrix(df: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
    """Calculate correlation matrix for sensor data"""
    
    sensor_data = df[sensor_columns]
    correlation_matrix = sensor_data.corr()
    
    return correlation_matrix

def detect_seasonality(data: pd.Series, period: int = 24) -> Dict[str, float]:
    """Detect seasonality in time series data"""
    
    # Calculate autocorrelation
    autocorr = data.autocorr(lag=period)
    
    # Calculate seasonal strength
    seasonal_strength = abs(autocorr)
    
    # Determine if seasonal
    is_seasonal = seasonal_strength > 0.3
    
    return {
        'is_seasonal': is_seasonal,
        'seasonal_strength': seasonal_strength,
        'autocorrelation': autocorr
    }

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def calculate_uptime(device_data: pd.DataFrame, threshold_hours: int = 24) -> float:
    """Calculate device uptime percentage"""
    
    total_hours = len(device_data)
    anomaly_hours = device_data['anomaly'].sum()
    
    uptime = (total_hours - anomaly_hours) / total_hours * 100
    
    return max(0, uptime)

def generate_alert_message(device_id: str, alert_type: str, 
                         sensor_values: Dict[str, float]) -> str:
    """Generate formatted alert message"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    message = f"""
🚨 ALERT - {alert_type.upper()}
Device: {device_id}
Time: {timestamp}

Sensor Values:
"""
    
    for sensor, value in sensor_values.items():
        message += f"  {sensor}: {value}\n"
    
    return message

def validate_sensor_data(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Validate sensor data quality"""
    
    issues = {
        'missing_values': [],
        'out_of_range': [],
        'duplicate_timestamps': [],
        'data_gaps': []
    }
    
    # Check for missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            issues['missing_values'].append(f"{col}: {df[col].isnull().sum()} missing values")
    
    # Check for out-of-range values
    sensor_ranges = {
        'temperature': (-50, 100),
        'vibration': (0, 20),
        'pressure': (0, 30),
        'current': (0, 50),
        'humidity': (0, 100)
    }
    
    for sensor, (min_val, max_val) in sensor_ranges.items():
        if sensor in df.columns:
            out_of_range = ((df[sensor] < min_val) | (df[sensor] > max_val)).sum()
            if out_of_range > 0:
                issues['out_of_range'].append(f"{sensor}: {out_of_range} values out of range")
    
    # Check for duplicate timestamps
    if 'timestamp' in df.columns:
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            issues['duplicate_timestamps'].append(f"timestamp: {duplicates} duplicate timestamps")
    
    return issues

def export_data_summary(df: pd.DataFrame) -> Dict[str, any]:
    """Export data summary statistics"""
    
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
            'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
        },
        'devices': df['device_id'].nunique() if 'device_id' in df.columns else 0,
        'anomaly_rate': df['anomaly'].mean() if 'anomaly' in df.columns else 0,
        'sensor_statistics': {}
    }
    
    # Calculate sensor statistics
    sensor_columns = ['temperature', 'vibration', 'pressure', 'current', 'humidity']
    
    for sensor in sensor_columns:
        if sensor in df.columns:
            summary['sensor_statistics'][sensor] = {
                'mean': float(df[sensor].mean()),
                'std': float(df[sensor].std()),
                'min': float(df[sensor].min()),
                'max': float(df[sensor].max()),
                'median': float(df[sensor].median())
            }
    
    return summary
