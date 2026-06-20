# Network Anomaly Detection System
> AI-powered tool to detect unusual patterns in network traffic, classify threat types, and alert administrators in real time.

---

## Project Structure

```
network_anomaly_detection/
├── app.py                  # Flask web application (dashboard + REST API)
├── train.py                # Model training entry point
├── requirements.txt
├── src/
│   ├── data_generator.py   # Synthetic network traffic generation
│   ├── preprocessor.py     # Feature engineering + scaling
│   ├── model.py            # Isolation Forest, Random Forest, Gradient Boosting
│   └── alert.py            # Alert creation, severity classification, log
├── templates/
│   ├── base.html
│   ├── dashboard.html      # Live dashboard with simulation
│   ├── predict.html        # Single & batch prediction UI
│   ├── alerts.html         # Alert log
│   └── metrics.html        # Model performance + plots
├── tests/
│   └── test_core.py        # Unit tests (pytest)
├── models/                 # Saved models + scaler + metrics (auto-created)
├── data/                   # Generated CSV data (auto-created)
├── logs/                   # Alert log JSON (auto-created)
└── static/
    └── reports/            # Confusion matrix + feature importance plots
```

---

## Features

| Feature | Description |
|---|---|
| **3 ML Models** | Isolation Forest (unsupervised), Random Forest, Gradient Boosting |
| **Feature Engineering** | bytes_ratio, packet_ratio, bytes_per_second derived features |
| **Threat Classification** | DoS, Port Scan, Brute Force, Data Exfiltration |
| **Severity Levels** | CRITICAL / HIGH / MEDIUM / LOW |
| **Web Dashboard** | Live stats, simulate traffic, alert feed |
| **REST API** | Single & batch prediction endpoints |
| **Batch CSV Analysis** | Upload CSV for bulk anomaly detection |
| **Persistent Alert Log** | JSON-based alert history |
| **Model Metrics Page** | Confusion matrices, feature importance charts |
| **Unit Tests** | pytest suite covering all core modules |

---

## Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- pip

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python train.py
```

This will:
- Generate 2400 synthetic network traffic records
- Train Isolation Forest, Random Forest, and Gradient Boosting classifiers
- Save all models to `models/`
- Print a metrics summary

Expected output:
```
✅  Training complete. Models saved to /models/
    Run `python app.py` to start the web dashboard.
```

### 4. Start the web dashboard

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

### 5. Run unit tests

```bash
# From the project root
pytest tests/test_core.py -v
```

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `POST /predict` | POST | Single record JSON prediction |
| `POST /predict_batch` | POST | CSV file batch prediction |
| `GET /api/simulate` | GET | Simulate a random traffic record |
| `GET /api/alerts?n=20` | GET | Last n alerts as JSON |
| `GET /api/stats` | GET | Alert statistics |
| `GET /api/model_info` | GET | Best model metrics |

### Example: Single prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0.1,
    "bytes_sent": 120000,
    "bytes_received": 200,
    "packets_sent": 600,
    "packets_received": 3,
    "port": 80,
    "protocol": 0,
    "failed_logins": 0,
    "num_connections": 250
  }'
```

Response:
```json
{
  "prediction": 1,
  "label": "Anomaly",
  "confidence": 97.3,
  "probability": 0.973,
  "alert": {
    "severity": "HIGH",
    "anomaly_type": "DoS / Flood Attack",
    "timestamp": "2024-01-15T10:23:45.123456"
  }
}
```

---

## ML Models

### Isolation Forest (Unsupervised)
- Detects anomalies without labels by isolating outliers
- `contamination=0.1` — expects ~10% anomalous traffic

### Random Forest (Supervised — Primary)
- 200 trees, class-weight balanced for imbalanced data
- 5-fold cross-validation F1 reported
- Used for all live predictions

### Gradient Boosting (Supervised)
- 150 estimators, learning_rate=0.1
- High precision on known attack patterns

### Feature Set
| Feature | Description |
|---|---|
| duration | Connection duration (seconds) |
| bytes_sent | Bytes transmitted |
| bytes_received | Bytes received |
| packets_sent / received | Packet counts |
| port | Destination port |
| protocol | 0=TCP, 1=UDP, 2=ICMP |
| failed_logins | Failed authentication attempts |
| num_connections | Connections from same host |
| bytes_ratio | bytes_sent / bytes_received |
| packet_ratio | packets_sent / packets_received |
| bytes_per_second | (bytes_sent+received) / duration |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.9+, Flask 3.0 |
| ML Framework | scikit-learn 1.4 |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Frontend | Bootstrap 5, Font Awesome, Vanilla JS |
| Model Persistence | joblib |
| Testing | pytest |

---

## Data Flow

```
Network Traffic → Feature Engineering → StandardScaler →
  ├── Isolation Forest (unsupervised baseline)
  ├── Random Forest (primary classifier) ──→ Prediction
  └── Gradient Boosting (high accuracy)

Prediction = Anomaly → Alert Engine → Severity Classification → Alert Log → Dashboard
```
