"""
Anomaly detection
Finds weird patterns in sensor data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """Finds anomalies using different ML methods"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_columns = []
        
    def prepare_data(self, df: pd.DataFrame, 
                    target_column: str = 'anomaly',
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training and testing"""
        
        # Select numeric features only
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col != target_column]
        
        X = df[feature_columns].fillna(0).values
        y = df[target_column].values if target_column in df.columns else None
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            return X_train, X_test, y_train, y_test
        else:
            return X, None, None, None
    
    def train_isolation_forest(self, X: np.ndarray, contamination: float = 0.1) -> Dict[str, Any]:
        """Train Isolation Forest model"""
        
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        model.fit(X)
        
        # Predictions
        predictions = model.predict(X)
        scores = model.score_samples(X)
        
        # Convert to binary (1 = normal, -1 = anomaly)
        binary_predictions = (predictions == 1).astype(int)
        
        return {
            'model': model,
            'predictions': binary_predictions,
            'scores': scores,
            'algorithm': 'Isolation Forest'
        }
    
    def train_one_class_svm(self, X: np.ndarray, nu: float = 0.1) -> Dict[str, Any]:
        """Train One-Class SVM model"""
        
        model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        model.fit(X)
        
        # Predictions
        predictions = model.predict(X)
        scores = model.score_samples(X)
        
        # Convert to binary (1 = normal, -1 = anomaly)
        binary_predictions = (predictions == 1).astype(int)
        
        return {
            'model': model,
            'predictions': binary_predictions,
            'scores': scores,
            'algorithm': 'One-Class SVM'
        }
    
    def train_local_outlier_factor(self, X: np.ndarray, n_neighbors: int = 20) -> Dict[str, Any]:
        """Train Local Outlier Factor model"""
        
        model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        model.fit(X)
        
        # Predictions
        predictions = model.predict(X)
        scores = model.score_samples(X)
        
        # Convert to binary (1 = normal, -1 = anomaly)
        binary_predictions = (predictions == 1).astype(int)
        
        return {
            'model': model,
            'predictions': binary_predictions,
            'scores': scores,
            'algorithm': 'Local Outlier Factor'
        }
    
    def train_supervised_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train supervised classification models"""
        
        results = {}
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        
        results['random_forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'probabilities': rf_prob,
            'algorithm': 'Random Forest'
        }
        
        # Logistic Regression
        lr_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_prob = lr_model.predict_proba(X_test)[:, 1]
        
        results['logistic_regression'] = {
            'model': lr_model,
            'predictions': lr_pred,
            'probabilities': lr_prob,
            'algorithm': 'Logistic Regression'
        }
        
        return results
    
    def create_lstm_autoencoder(self, input_shape: Tuple[int, int]) -> Model:
        """Create LSTM Autoencoder for anomaly detection"""
        
        # Encoder
        encoder_input = Input(shape=input_shape, name='encoder_input')
        encoder_lstm1 = LSTM(64, return_sequences=True, name='encoder_lstm1')(encoder_input)
        encoder_lstm2 = LSTM(32, return_sequences=False, name='encoder_lstm2')(encoder_lstm1)
        encoder_dense = Dense(16, name='encoder_dense')(encoder_lstm2)
        
        # Decoder
        decoder_repeat = RepeatVector(input_shape[0], name='decoder_repeat')(encoder_dense)
        decoder_lstm1 = LSTM(32, return_sequences=True, name='decoder_lstm1')(decoder_repeat)
        decoder_lstm2 = LSTM(64, return_sequences=True, name='decoder_lstm2')(decoder_lstm1)
        decoder_output = TimeDistributed(Dense(input_shape[1], name='decoder_output'))(decoder_lstm2)
        
        # Autoencoder
        autoencoder = Model(encoder_input, decoder_output, name='lstm_autoencoder')
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder
    
    def train_lstm_autoencoder(self, X: np.ndarray, 
                             sequence_length: int = 10,
                             epochs: int = 100,
                             batch_size: int = 32) -> Dict[str, Any]:
        """Train LSTM Autoencoder for anomaly detection"""
        
        # Reshape data for LSTM
        X_reshaped = self.create_sequences(X, sequence_length)
        
        # Create and train model
        input_shape = (sequence_length, X.shape[1])
        model = self.create_lstm_autoencoder(input_shape)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_reshaped, X_reshaped,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Predictions and anomaly scores
        predictions = model.predict(X_reshaped)
        mse = np.mean(np.power(X_reshaped - predictions, 2), axis=1)
        
        # Threshold for anomaly detection (using 95th percentile)
        threshold = np.percentile(mse, 95)
        binary_predictions = (mse > threshold).astype(int)
        
        return {
            'model': model,
            'predictions': binary_predictions,
            'scores': mse,
            'threshold': threshold,
            'algorithm': 'LSTM Autoencoder',
            'history': history.history
        }
    
    def create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for LSTM training"""
        
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)
    
    def train_all_models(self, df: pd.DataFrame, 
                        target_column: str = 'anomaly') -> Dict[str, Any]:
        """Train all anomaly detection models"""
        
        print("Training anomaly detection models...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_column)
        
        results = {}
        
        # Unsupervised models
        print("Training Isolation Forest...")
        results['isolation_forest'] = self.train_isolation_forest(X_train)
        
        print("Training One-Class SVM...")
        results['one_class_svm'] = self.train_one_class_svm(X_train)
        
        print("Training Local Outlier Factor...")
        results['local_outlier_factor'] = self.train_local_outlier_factor(X_train)
        
        # Supervised models
        if y_train is not None:
            print("Training supervised models...")
            supervised_results = self.train_supervised_classifier(
                X_train, y_train, X_test, y_test
            )
            results.update(supervised_results)
        
        # LSTM Autoencoder
        print("Training LSTM Autoencoder...")
        results['lstm_autoencoder'] = self.train_lstm_autoencoder(X_train)
        
        self.models = results
        self.is_trained = True
        
        return results
    
    def evaluate_models(self, df: pd.DataFrame, 
                       target_column: str = 'anomaly') -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        X, _, y_true, _ = self.prepare_data(df, target_column)
        evaluations = {}
        
        for name, model_info in self.models.items():
            if 'predictions' in model_info:
                y_pred = model_info['predictions']
                
                if y_true is not None:
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    
                    evaluations[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
                else:
                    # For unsupervised models, use anomaly score statistics
                    scores = model_info.get('scores', [])
                    evaluations[name] = {
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'min_score': np.min(scores),
                        'max_score': np.max(scores)
                    }
        
        return evaluations
    
    def predict_anomalies(self, df: pd.DataFrame, 
                         model_name: str = 'isolation_forest') -> np.ndarray:
        """Predict anomalies using specified model"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        X, _, _, _ = self.prepare_data(df)
        model_info = self.models[model_name]
        
        if hasattr(model_info['model'], 'predict'):
            predictions = model_info['model'].predict(X)
            # Convert to binary (1 = normal, -1 = anomaly)
            binary_predictions = (predictions == 1).astype(int)
        else:
            binary_predictions = model_info['predictions']
        
        return binary_predictions
    
    def get_anomaly_scores(self, df: pd.DataFrame, 
                          model_name: str = 'isolation_forest') -> np.ndarray:
        """Get anomaly scores from specified model"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        X, _, _, _ = self.prepare_data(df)
        model_info = self.models[model_name]
        
        if hasattr(model_info['model'], 'score_samples'):
            scores = model_info['model'].score_samples(X)
        else:
            scores = model_info.get('scores', np.zeros(len(X)))
        
        return scores
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        
        if not self.is_trained:
            raise ValueError("No models to save")
        
        # Save non-Keras models
        model_data = {}
        for name, model_info in self.models.items():
            if name != 'lstm_autoencoder':
                model_data[name] = {
                    'model': model_info['model'],
                    'algorithm': model_info['algorithm']
                }
        
        joblib.dump(model_data, f"{filepath}_sklearn_models.pkl")
        
        # Save LSTM model separately
        if 'lstm_autoencoder' in self.models:
            self.models['lstm_autoencoder']['model'].save(f"{filepath}_lstm_model.h5")
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        
        # Load sklearn models
        model_data = joblib.load(f"{filepath}_sklearn_models.pkl")
        
        # Load LSTM model
        try:
            lstm_model = tf.keras.models.load_model(f"{filepath}_lstm_model.h5")
            model_data['lstm_autoencoder'] = {
                'model': lstm_model,
                'algorithm': 'LSTM Autoencoder'
            }
        except:
            pass
        
        self.models = model_data
        self.is_trained = True

def create_ensemble_predictor(models: Dict[str, Any]) -> Dict[str, Any]:
    """Create ensemble predictor from multiple models"""
    
    def ensemble_predict(X: np.ndarray) -> np.ndarray:
        """Ensemble prediction using voting"""
        
        predictions = []
        for name, model_info in models.items():
            if hasattr(model_info['model'], 'predict'):
                pred = model_info['model'].predict(X)
                # Convert to binary
                binary_pred = (pred == 1).astype(int) if hasattr(pred, '__len__') else int(pred == 1)
                predictions.append(binary_pred)
        
        if predictions:
            # Majority voting
            ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
            return ensemble_pred
        else:
            return np.zeros(len(X))
    
    return {
        'predict': ensemble_predict,
        'algorithm': 'Ensemble',
        'models': list(models.keys())
    }

if __name__ == "__main__":
    # Test anomaly detection
    from data_generator import create_sample_dataset
    from data_processor import IoTDataProcessor
    
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    print("Processing data...")
    processor = IoTDataProcessor()
    X, df_processed = processor.process_data(df)
    
    print("Training anomaly detection models...")
    detector = AnomalyDetector()
    results = detector.train_all_models(df_processed)
    
    print("Evaluating models...")
    evaluations = detector.evaluate_models(df_processed)
    
    for model_name, metrics in evaluations.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nAnomaly detection models trained successfully!")

