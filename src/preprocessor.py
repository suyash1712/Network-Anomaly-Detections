"""
preprocessor.py - Data preprocessing and feature engineering for network traffic.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

FEATURES = [
    'duration', 'bytes_sent', 'bytes_received', 'packets_sent',
    'packets_received', 'port', 'protocol', 'failed_logins', 'num_connections'
]

SCALER_PATH = 'models/scaler.pkl'


def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def preprocess(df: pd.DataFrame, fit_scaler: bool = True) -> tuple:
    """
    Clean, engineer features, and scale.

    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray | None): Labels if present
        scaler (StandardScaler)
    """
    df = df.copy()

    # --- Feature engineering ---
    df['bytes_ratio'] = df['bytes_sent'] / (df['bytes_received'] + 1)
    df['packet_ratio'] = df['packets_sent'] / (df['packets_received'] + 1)
    df['bytes_per_second'] = (df['bytes_sent'] + df['bytes_received']) / (df['duration'] + 1)

    extended_features = FEATURES + ['bytes_ratio', 'packet_ratio', 'bytes_per_second']

    # Handle missing values
    df[extended_features] = df[extended_features].fillna(df[extended_features].median())

    X = df[extended_features].values
    y = df['label'].values if 'label' in df.columns else None

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"Scaler saved to {SCALER_PATH}")
    else:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Train the model first.")
        scaler = joblib.load(SCALER_PATH)
        X = scaler.transform(X)

    return X, y, scaler


def preprocess_single(record: dict) -> np.ndarray:
    """Preprocess a single traffic record dict for real-time prediction."""
    df = pd.DataFrame([record])

    # Defaults for any missing fields
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    df['bytes_ratio'] = df['bytes_sent'] / (df['bytes_received'] + 1)
    df['packet_ratio'] = df['packets_sent'] / (df['packets_received'] + 1)
    df['bytes_per_second'] = (df['bytes_sent'] + df['bytes_received']) / (df['duration'] + 1)

    extended_features = FEATURES + ['bytes_ratio', 'packet_ratio', 'bytes_per_second']
    X = df[extended_features].values

    scaler = joblib.load(SCALER_PATH)
    return scaler.transform(X)
