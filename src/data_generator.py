"""
data_generator.py - Generates synthetic network traffic data for training/testing.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_network_data(n_normal=1000, n_anomaly=100, save_path=None):
    """
    Generate synthetic network traffic dataset with normal and anomalous records.

    Features:
        - duration: Connection duration in seconds
        - bytes_sent: Bytes sent in the connection
        - bytes_received: Bytes received
        - packets_sent: Number of packets sent
        - packets_received: Number of packets received
        - port: Destination port number
        - protocol: Network protocol (0=TCP, 1=UDP, 2=ICMP)
        - failed_logins: Number of failed login attempts
        - num_connections: Number of connections from same host
        - label: 0=Normal, 1=Anomaly
    """

    # --- Normal traffic ---
    normal = pd.DataFrame({
        'duration':          np.random.exponential(scale=5, size=n_normal).round(2),
        'bytes_sent':        np.random.normal(5000, 1500, n_normal).clip(100).astype(int),
        'bytes_received':    np.random.normal(8000, 2000, n_normal).clip(100).astype(int),
        'packets_sent':      np.random.normal(20, 5, n_normal).clip(1).astype(int),
        'packets_received':  np.random.normal(30, 8, n_normal).clip(1).astype(int),
        'port':              np.random.choice([80, 443, 22, 53, 8080, 3306], n_normal),
        'protocol':          np.random.choice([0, 1, 2], n_normal, p=[0.7, 0.2, 0.1]),
        'failed_logins':     np.random.choice([0, 1], n_normal, p=[0.97, 0.03]),
        'num_connections':   np.random.normal(10, 3, n_normal).clip(1).astype(int),
        'label':             0
    })

    # --- Anomalous traffic (DoS, Port Scan, Brute Force patterns) ---
    dos = pd.DataFrame({
        'duration':          np.random.exponential(scale=0.5, size=n_anomaly // 3).round(2),
        'bytes_sent':        np.random.normal(100000, 20000, n_anomaly // 3).clip(5000).astype(int),
        'bytes_received':    np.random.normal(500, 100, n_anomaly // 3).clip(0).astype(int),
        'packets_sent':      np.random.normal(500, 50, n_anomaly // 3).clip(100).astype(int),
        'packets_received':  np.random.normal(5, 2, n_anomaly // 3).clip(0).astype(int),
        'port':              np.random.choice([80, 443], n_anomaly // 3),
        'protocol':          np.random.choice([0, 1, 2], n_anomaly // 3),
        'failed_logins':     np.zeros(n_anomaly // 3, dtype=int),
        'num_connections':   np.random.normal(200, 30, n_anomaly // 3).clip(100).astype(int),
        'label':             1
    })

    portscan = pd.DataFrame({
        'duration':          np.random.exponential(scale=0.1, size=n_anomaly // 3).round(2),
        'bytes_sent':        np.random.normal(200, 50, n_anomaly // 3).clip(50).astype(int),
        'bytes_received':    np.random.normal(100, 30, n_anomaly // 3).clip(0).astype(int),
        'packets_sent':      np.random.normal(3, 1, n_anomaly // 3).clip(1).astype(int),
        'packets_received':  np.random.normal(1, 0.5, n_anomaly // 3).clip(0).astype(int),
        'port':              np.random.randint(1, 65535, n_anomaly // 3),
        'protocol':          np.random.choice([0, 1], n_anomaly // 3),
        'failed_logins':     np.zeros(n_anomaly // 3, dtype=int),
        'num_connections':   np.random.normal(150, 20, n_anomaly // 3).clip(50).astype(int),
        'label':             1
    })

    brute = pd.DataFrame({
        'duration':          np.random.normal(2, 0.5, n_anomaly // 3).clip(0.1).round(2),
        'bytes_sent':        np.random.normal(300, 100, n_anomaly // 3).clip(50).astype(int),
        'bytes_received':    np.random.normal(200, 80, n_anomaly // 3).clip(0).astype(int),
        'packets_sent':      np.random.normal(5, 2, n_anomaly // 3).clip(1).astype(int),
        'packets_received':  np.random.normal(4, 1, n_anomaly // 3).clip(1).astype(int),
        'port':              np.full(n_anomaly // 3, 22),
        'protocol':          np.zeros(n_anomaly // 3, dtype=int),
        'failed_logins':     np.random.normal(20, 5, n_anomaly // 3).clip(5).astype(int),
        'num_connections':   np.random.normal(50, 10, n_anomaly // 3).clip(10).astype(int),
        'label':             1
    })

    df = pd.concat([normal, dos, portscan, brute], ignore_index=True).sample(frac=1, random_state=42)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[DataGen] Saved {len(df)} records to {save_path}")

    return df


if __name__ == '__main__':
    df = generate_network_data(save_path='data/network_traffic.csv')
    print(df['label'].value_counts())
    print(df.head())
