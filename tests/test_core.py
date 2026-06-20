"""
tests/test_core.py - Unit tests for Network Anomaly Detection core modules.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd

from src.data_generator import generate_network_data
from src.preprocessor   import preprocess, preprocess_single
from src.alert          import classify_anomaly_type, get_severity, create_alert, get_recent_alerts


# ─────────────────────────────────────────────
# Data generator tests
# ─────────────────────────────────────────────

class TestDataGenerator:
    def test_shape(self):
        df = generate_network_data(n_normal=100, n_anomaly=30)
        # n_anomaly//3 * 3 may be slightly less than n_anomaly
        assert len(df) >= 100 + 27  # at minimum 27 anomalies (floor division)

    def test_columns(self):
        df = generate_network_data(n_normal=50, n_anomaly=10)
        required = ['duration','bytes_sent','bytes_received','packets_sent',
                    'packets_received','port','protocol','failed_logins','num_connections','label']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_labels(self):
        df = generate_network_data(n_normal=100, n_anomaly=30)
        assert set(df['label'].unique()).issubset({0, 1})
        assert df['label'].sum() >= 27  # at least floor(30/3)*3 anomalies

    def test_no_negative_bytes(self):
        df = generate_network_data(n_normal=200, n_anomaly=50)
        assert (df['bytes_sent'] >= 0).all()
        assert (df['bytes_received'] >= 0).all()


# ─────────────────────────────────────────────
# Preprocessor tests
# ─────────────────────────────────────────────

class TestPreprocessor:
    def test_preprocess_shape(self):
        df = generate_network_data(n_normal=100, n_anomaly=20)
        X, y, scaler = preprocess(df, fit_scaler=True)
        assert X.shape[0] >= 115    # at least 100 normal + floor(20/3)*3 anomalies
        assert X.shape[1] == 12   # 9 original + 3 engineered
        assert len(y) == X.shape[0]

    def test_preprocess_single_shape(self):
        # Need scaler from previous preprocess
        df = generate_network_data(n_normal=100, n_anomaly=20)
        preprocess(df, fit_scaler=True)

        record = {
            'duration': 2.5, 'bytes_sent': 5000, 'bytes_received': 8000,
            'packets_sent': 20, 'packets_received': 30, 'port': 80,
            'protocol': 0, 'failed_logins': 0, 'num_connections': 10
        }
        X = preprocess_single(record)
        assert X.shape == (1, 12)

    def test_scaled_range(self):
        df = generate_network_data(n_normal=500, n_anomaly=100)
        X, _, _ = preprocess(df, fit_scaler=True)
        # After StandardScaler, mean ≈ 0, std ≈ 1
        assert abs(X.mean()) < 0.5


# ─────────────────────────────────────────────
# Alert tests
# ─────────────────────────────────────────────

class TestAlert:
    def test_brute_force_detection(self):
        record = {'failed_logins': 20, 'bytes_sent': 300, 'num_connections': 10, 'duration': 2}
        assert classify_anomaly_type(record) == 'Brute Force Attack'

    def test_dos_detection(self):
        record = {'failed_logins': 0, 'num_connections': 200, 'bytes_sent': 5000, 'duration': 0.5}
        assert classify_anomaly_type(record) == 'DoS / Flood Attack'

    def test_portscan_detection(self):
        record = {'failed_logins': 0, 'num_connections': 60, 'bytes_sent': 200, 'duration': 0.1}
        assert classify_anomaly_type(record) == 'Port Scan'

    def test_severity_critical(self):
        assert get_severity('Brute Force Attack') == 'CRITICAL'

    def test_severity_high(self):
        assert get_severity('DoS / Flood Attack') == 'HIGH'

    def test_create_alert_structure(self):
        record = {'failed_logins': 15, 'bytes_sent': 300, 'num_connections': 5, 'duration': 1.5}
        alert = create_alert(record, confidence=0.92)
        assert 'severity'     in alert
        assert 'anomaly_type' in alert
        assert 'confidence'   in alert
        assert 'timestamp'    in alert
        assert alert['confidence'] == 92.0

    def test_get_recent_alerts_returns_list(self):
        alerts = get_recent_alerts(5)
        assert isinstance(alerts, list)


# ─────────────────────────────────────────────
# Model integration tests (require trained model)
# ─────────────────────────────────────────────

class TestModelIntegration:
    """These tests run only if a trained model exists."""

    def test_model_loads(self):
        try:
            from src.model import load_best_model
            model = load_best_model()
            assert model is not None
        except FileNotFoundError:
            pytest.skip("Model not trained yet.")

    def test_prediction_output(self):
        try:
            from src.model import load_best_model
            model = load_best_model()
        except FileNotFoundError:
            pytest.skip("Model not trained yet.")

        df = generate_network_data(n_normal=50, n_anomaly=10)
        X, y, _ = preprocess(df, fit_scaler=True)
        preds = model.predict(X)
        assert len(preds) >= 55  # at least 50 normal + floor(10/3)*3 anomalies
        assert set(preds).issubset({0, 1})

    def test_proba_range(self):
        try:
            from src.model import load_best_model
            model = load_best_model()
        except FileNotFoundError:
            pytest.skip("Model not trained yet.")

        df = generate_network_data(n_normal=20, n_anomaly=5)
        X, _, _ = preprocess(df, fit_scaler=True)
        proba = model.predict_proba(X)
        assert proba.shape[1] == 2
        assert ((proba >= 0) & (proba <= 1)).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
