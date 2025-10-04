"""
Data Processing for IoT
Feature engineering and data cleaning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class IoTDataProcessor:
    """Process and engineer features from IoT sensor data"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
        self.is_fitted = False
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp"""
        
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_working_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                             window_sizes: List[int] = [5, 10, 30, 60]) -> pd.DataFrame:
        """Create rolling window features for time series"""
        
        df = df.copy()
        sensor_columns = ['temperature', 'vibration', 'pressure', 'current', 'humidity']
        
        for window in window_sizes:
            for sensor in sensor_columns:
                # Rolling statistics
                df[f'{sensor}_mean_{window}'] = df[sensor].rolling(window=window, min_periods=1).mean()
                df[f'{sensor}_std_{window}'] = df[sensor].rolling(window=window, min_periods=1).std()
                df[f'{sensor}_max_{window}'] = df[sensor].rolling(window=window, min_periods=1).max()
                df[f'{sensor}_min_{window}'] = df[sensor].rolling(window=window, min_periods=1).min()
                
                # Change from previous value
                df[f'{sensor}_diff_{window}'] = df[sensor].diff(window)
                
                # Rate of change
                df[f'{sensor}_pct_change_{window}'] = df[sensor].pct_change(window)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                          lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lagged features"""
        
        df = df.copy()
        sensor_columns = ['temperature', 'vibration', 'pressure', 'current', 'humidity']
        
        for lag in lags:
            for sensor in sensor_columns:
                df[f'{sensor}_lag_{lag}'] = df[sensor].shift(lag)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between sensors"""
        
        df = df.copy()
        
        # Temperature-pressure interaction (thermal expansion)
        df['temp_pressure_ratio'] = df['temperature'] / (df['pressure'] + 1e-6)
        
        # Vibration-current interaction (motor load)
        df['vibration_current_product'] = df['vibration'] * df['current']
        
        # Humidity-temperature interaction
        df['humidity_temp_diff'] = df['humidity'] - df['temperature']
        
        # Combined sensor health score
        df['sensor_health_score'] = (
            (df['temperature'] / 50) +  # Normalize temperature
            (df['vibration'] / 5) +     # Normalize vibration
            (df['pressure'] / 20) +     # Normalize pressure
            (df['current'] / 30)        # Normalize current
        ) / 4
        
        return df
    
    def create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for anomaly detection"""
        
        df = df.copy()
        sensor_columns = ['temperature', 'vibration', 'pressure', 'current', 'humidity']
        
        for sensor in sensor_columns:
            # Z-score
            mean_val = df[sensor].mean()
            std_val = df[sensor].std()
            df[f'{sensor}_zscore'] = (df[sensor] - mean_val) / (std_val + 1e-6)
            
            # Percentile rank
            df[f'{sensor}_percentile'] = df[sensor].rank(pct=True)
            
            # Distance from median
            median_val = df[sensor].median()
            df[f'{sensor}_median_distance'] = abs(df[sensor] - median_val)
            
            # Outlier detection (IQR method)
            Q1 = df[sensor].quantile(0.25)
            Q3 = df[sensor].quantile(0.75)
            IQR = Q3 - Q1
            df[f'{sensor}_is_outlier'] = (
                (df[sensor] < (Q1 - 1.5 * IQR)) | 
                (df[sensor] > (Q3 + 1.5 * IQR))
            ).astype(int)
        
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical features for time patterns"""
        
        df = df.copy()
        
        # Hour cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week cyclical features
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def process_data(self, df: pd.DataFrame, 
                    target_column: str = 'anomaly',
                    fit_scalers: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete data processing pipeline"""
        
        print("Processing IoT sensor data...")
        
        # Create all features
        df_processed = self.create_time_features(df)
        df_processed = self.create_rolling_features(df_processed)
        df_processed = self.create_lag_features(df_processed)
        df_processed = self.create_interaction_features(df_processed)
        df_processed = self.create_anomaly_features(df_processed)
        df_processed = self.create_cyclical_features(df_processed)
        
        # Select feature columns (exclude metadata and target)
        exclude_columns = [
            'timestamp', 'device_id', 'device_type', 'location', 
            'anomaly', 'maintenance_required', 'maintenance_priority'
        ]
        
        feature_columns = [col for col in df_processed.columns 
                          if col not in exclude_columns]
        
        # Handle missing values
        df_processed[feature_columns] = df_processed[feature_columns].fillna(method='ffill').fillna(0)
        
        # Split features and target
        X = df_processed[feature_columns].copy()
        y = df_processed[target_column].copy() if target_column in df_processed.columns else None
        
        # Scale features
        if fit_scalers:
            self.scalers['standard'] = StandardScaler()
            self.scalers['minmax'] = MinMaxScaler()
            
            X_scaled = self.scalers['standard'].fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
            
            self.feature_columns = feature_columns
            self.is_fitted = True
        else:
            if self.is_fitted:
                X_scaled = self.scalers['standard'].transform(X)
                X = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        # Add metadata back
        metadata_columns = ['timestamp', 'device_id', 'device_type', 'location']
        for col in metadata_columns:
            if col in df_processed.columns:
                X[col] = df_processed[col]
        
        if y is not None:
            X[target_column] = y
        
        return X, df_processed
    
    def get_feature_importance(self, model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from trained model"""
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
        else:
            return pd.DataFrame()
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance
    
    def reduce_dimensions(self, X: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """Reduce feature dimensions using PCA"""
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X.select_dtypes(include=[np.number]))
        
        pca_columns = [f'PC_{i+1}' for i in range(n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        # Add non-numeric columns back
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X_pca_df[col] = X[col]
        
        return X_pca_df, pca

def load_and_process_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and process IoT data from file"""
    
    # Load data
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Process data
    processor = IoTDataProcessor()
    X, df_processed = processor.process_data(df)
    
    return X, df_processed

if __name__ == "__main__":
    # Test data processing
    from data_generator import create_sample_dataset
    
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    print("Processing data...")
    processor = IoTDataProcessor()
    X, df_processed = processor.process_data(df)
    
    print(f"Original features: {len(df.columns)}")
    print(f"Processed features: {len(X.columns)}")
    print(f"Anomaly rate: {df['anomaly'].mean():.3f}")
    
    # Save processed data
    X.to_csv('data/processed_iot_data.csv', index=False)
    print("Processed data saved to data/processed_iot_data.csv")

