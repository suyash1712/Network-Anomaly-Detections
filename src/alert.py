"""
alert.py - Alert generation and logging for detected anomalies.
"""

import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ALERT_LOG_PATH = 'logs/alerts.json'
os.makedirs('logs', exist_ok=True)


SEVERITY_MAP = {
    'high_failed_logins':   'CRITICAL',
    'high_bytes_sent':      'HIGH',
    'high_num_connections': 'HIGH',
    'short_duration':       'MEDIUM',
    'default':              'MEDIUM',
}


def classify_anomaly_type(record: dict) -> str:
    """Heuristic anomaly type classification based on traffic features."""
    fl  = record.get('failed_logins', 0)
    bs  = record.get('bytes_sent', 0)
    nc  = record.get('num_connections', 0)
    dur = record.get('duration', 1)

    if fl >= 10:
        return 'Brute Force Attack'
    if nc >= 100:
        return 'DoS / Flood Attack'
    if dur < 0.5 and nc >= 50:
        return 'Port Scan'
    if bs > 50000:
        return 'Data Exfiltration'
    return 'Unknown Anomaly'


def get_severity(anomaly_type: str) -> str:
    severity_table = {
        'Brute Force Attack':  'CRITICAL',
        'DoS / Flood Attack':  'HIGH',
        'Port Scan':           'MEDIUM',
        'Data Exfiltration':   'CRITICAL',
        'Unknown Anomaly':     'LOW',
    }
    return severity_table.get(anomaly_type, 'LOW')


def create_alert(record: dict, confidence: float) -> dict:
    anomaly_type = classify_anomaly_type(record)
    severity     = get_severity(anomaly_type)

    alert = {
        'timestamp':    datetime.now().isoformat(),
        'severity':     severity,
        'anomaly_type': anomaly_type,
        'confidence':   round(confidence * 100, 1),
        'record':       record,
        'message':      f"[{severity}] {anomaly_type} detected with {round(confidence*100,1)}% confidence."
    }
    _log_alert(alert)
    logger.warning(alert['message'])
    return alert


def _log_alert(alert: dict):
    """Append alert to JSON log file."""
    alerts = []
    if os.path.exists(ALERT_LOG_PATH):
        try:
            with open(ALERT_LOG_PATH) as f:
                alerts = json.load(f)
        except Exception:
            alerts = []
    alerts.append(alert)
    # Keep last 1000 alerts
    alerts = alerts[-1000:]
    with open(ALERT_LOG_PATH, 'w') as f:
        json.dump(alerts, f, indent=2)


def get_recent_alerts(n: int = 20) -> list:
    if not os.path.exists(ALERT_LOG_PATH):
        return []
    try:
        with open(ALERT_LOG_PATH) as f:
            alerts = json.load(f)
        return alerts[-n:][::-1]
    except Exception:
        return []


def get_alert_stats() -> dict:
    alerts = get_recent_alerts(1000)
    if not alerts:
        return {'total': 0, 'by_severity': {}, 'by_type': {}}

    by_severity = {}
    by_type     = {}
    for a in alerts:
        sev  = a.get('severity', 'UNKNOWN')
        atyp = a.get('anomaly_type', 'Unknown')
        by_severity[sev]  = by_severity.get(sev, 0) + 1
        by_type[atyp]     = by_type.get(atyp, 0) + 1

    return {
        'total':       len(alerts),
        'by_severity': by_severity,
        'by_type':     by_type,
    }
