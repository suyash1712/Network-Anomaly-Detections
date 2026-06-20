"""
app.py - Flask web dashboard for Network Anomaly Detection.

Routes:
    GET  /                    → Dashboard
    POST /predict             → Single record prediction
    POST /predict_batch       → CSV batch prediction
    GET  /alerts              → Recent alerts page
    GET  /metrics             → Model performance page
    GET  /api/alerts          → JSON alerts feed
    GET  /api/stats           → JSON dashboard stats
    GET  /api/simulate        → Simulate random traffic record
"""

import os
import sys
import json
import random
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS

from src.preprocessor import preprocess_single, FEATURES
from src.model        import load_best_model
from src.alert        import create_alert, get_recent_alerts, get_alert_stats

app = Flask(__name__)
CORS(app)

# Load model once at startup
try:
    MODEL = load_best_model()
    print("[App] Model loaded successfully.")
except FileNotFoundError:
    MODEL = None
    print("[App] WARNING: Model not found. Run `python train.py` first.")


def _predict(record: dict) -> dict:
    """Core prediction logic. Returns prediction dict."""
    X = preprocess_single(record)
    prediction  = int(MODEL.predict(X)[0])
    probability = float(MODEL.predict_proba(X)[0][1])

    result = {
        'prediction':   prediction,
        'label':        'Anomaly' if prediction == 1 else 'Normal',
        'confidence':   round(probability * 100, 1),
        'probability':  round(probability, 4),
    }

    if prediction == 1:
        alert = create_alert(record, probability)
        result['alert'] = alert

    return result


# ─────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────

@app.route('/')
def dashboard():
    stats = get_alert_stats()
    metrics = {}
    metrics_path = 'models/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return render_template('dashboard.html', stats=stats, metrics=metrics)


@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')


@app.route('/alerts')
def alerts_page():
    alerts = get_recent_alerts(50)
    return render_template('alerts.html', alerts=alerts)


@app.route('/metrics')
def metrics_page():
    metrics = {}
    metrics_path = 'models/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return render_template('metrics.html', metrics=metrics)


# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Run python train.py first.'}), 503

    data = request.get_json(force=True)
    try:
        result = _predict(data)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
        results = []
        for _, row in df.iterrows():
            record = row.to_dict()
            res = _predict(record)
            res['record'] = record
            results.append(res)

        n_anomaly = sum(1 for r in results if r['prediction'] == 1)
        return jsonify({
            'total':    len(results),
            'anomalies': n_anomaly,
            'normal':   len(results) - n_anomaly,
            'results':  results
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.route('/api/alerts')
def api_alerts():
    n = request.args.get('n', 20, type=int)
    return jsonify(get_recent_alerts(n))


@app.route('/api/stats')
def api_stats():
    return jsonify(get_alert_stats())


@app.route('/api/simulate')
def api_simulate():
    """Generate a random network traffic record and predict."""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    # Randomly choose normal or anomalous pattern
    kind = random.choice(['normal', 'dos', 'portscan', 'brute'])
    if kind == 'normal':
        record = {
            'duration':         round(random.expovariate(0.2), 2),
            'bytes_sent':       int(random.gauss(5000, 1500)),
            'bytes_received':   int(random.gauss(8000, 2000)),
            'packets_sent':     max(1, int(random.gauss(20, 5))),
            'packets_received': max(1, int(random.gauss(30, 8))),
            'port':             random.choice([80, 443, 22, 53, 8080]),
            'protocol':         random.choice([0, 1, 2]),
            'failed_logins':    0,
            'num_connections':  max(1, int(random.gauss(10, 3))),
        }
    elif kind == 'dos':
        record = {
            'duration':         round(random.expovariate(2), 2),
            'bytes_sent':       int(random.gauss(100000, 20000)),
            'bytes_received':   int(random.gauss(500, 100)),
            'packets_sent':     int(random.gauss(500, 50)),
            'packets_received': max(0, int(random.gauss(5, 2))),
            'port':             random.choice([80, 443]),
            'protocol':         random.choice([0, 1]),
            'failed_logins':    0,
            'num_connections':  int(random.gauss(200, 30)),
        }
    elif kind == 'portscan':
        record = {
            'duration':         round(random.expovariate(10), 2),
            'bytes_sent':       int(random.gauss(200, 50)),
            'bytes_received':   int(random.gauss(100, 30)),
            'packets_sent':     max(1, int(random.gauss(3, 1))),
            'packets_received': max(0, int(random.gauss(1, 0.5))),
            'port':             random.randint(1, 65535),
            'protocol':         random.choice([0, 1]),
            'failed_logins':    0,
            'num_connections':  int(random.gauss(150, 20)),
        }
    else:  # brute
        record = {
            'duration':         round(random.gauss(2, 0.5), 2),
            'bytes_sent':       int(random.gauss(300, 100)),
            'bytes_received':   int(random.gauss(200, 80)),
            'packets_sent':     max(1, int(random.gauss(5, 2))),
            'packets_received': max(1, int(random.gauss(4, 1))),
            'port':             22,
            'protocol':         0,
            'failed_logins':    max(5, int(random.gauss(20, 5))),
            'num_connections':  int(random.gauss(50, 10)),
        }

    result = _predict(record)
    result['simulated_type'] = kind
    result['record'] = record
    return jsonify(result)


@app.route('/api/model_info')
def api_model_info():
    metrics = {}
    path = 'models/metrics.json'
    if os.path.exists(path):
        with open(path) as f:
            metrics = json.load(f)
    return jsonify({'model': 'Random Forest', 'metrics': metrics.get('Random Forest', {})})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
