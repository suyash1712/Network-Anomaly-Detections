"""
model.py - Train and evaluate anomaly detection models.

Algorithms:
    1. Isolation Forest  (unsupervised baseline)
    2. Random Forest     (supervised, primary model)
    3. Gradient Boosting (supervised, high-accuracy)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR  = 'models'
REPORT_DIR = 'static/reports'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────

def train_isolation_forest(X_train):
    logger.info("Training Isolation Forest …")
    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train)
    joblib.dump(model, f'{MODEL_DIR}/isolation_forest.pkl')
    logger.info("Isolation Forest saved.")
    return model


def train_random_forest(X_train, y_train):
    logger.info("Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, f'{MODEL_DIR}/random_forest.pkl')
    logger.info("Random Forest saved.")
    return model


def train_gradient_boosting(X_train, y_train):
    logger.info("Training Gradient Boosting …")
    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, f'{MODEL_DIR}/gradient_boosting.pkl')
    logger.info("Gradient Boosting saved.")
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute metrics, save confusion matrix plot, return metrics dict."""

    if hasattr(model, 'predict_proba'):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    elif isinstance(model, IsolationForest):
        raw = model.predict(X_test)           # +1 normal, -1 anomaly
        y_pred = np.where(raw == -1, 1, 0)
        scores  = model.score_samples(X_test)  # lower = more anomalous
        y_prob  = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_pred = model.predict(X_test)
        y_prob = y_pred.astype(float)

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
    }
    logger.info(f"{model_name} — {metrics}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'], ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    fig.tight_layout()
    plot_path = f'{REPORT_DIR}/cm_{model_name.lower().replace(" ", "_")}.png'
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)

    return metrics


def save_metrics(all_metrics: dict):
    path = f'{MODEL_DIR}/metrics.json'
    with open(path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")


def plot_feature_importance(model, feature_names: list, model_name: str):
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), importances[indices], align='center', color='steelblue')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title(f'Feature Importance — {model_name}')
    ax.set_ylabel('Importance')
    fig.tight_layout()
    plot_path = f'{REPORT_DIR}/fi_{model_name.lower().replace(" ", "_")}.png'
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    logger.info(f"Feature importance plot saved: {plot_path}")


# ─────────────────────────────────────────────
# Main train pipeline
# ─────────────────────────────────────────────

def train_all(X, y, feature_names: list) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # 1. Isolation Forest
    iso = train_isolation_forest(X_train)
    results['Isolation Forest'] = evaluate_model(iso, X_test, y_test, 'Isolation Forest')

    # 2. Random Forest
    rf = train_random_forest(X_train, y_train)
    results['Random Forest'] = evaluate_model(rf, X_test, y_test, 'Random Forest')
    plot_feature_importance(rf, feature_names, 'Random Forest')

    # 3. Gradient Boosting
    gb = train_gradient_boosting(X_train, y_train)
    results['Gradient Boosting'] = evaluate_model(gb, X_test, y_test, 'Gradient Boosting')
    plot_feature_importance(gb, feature_names, 'Gradient Boosting')

    # Cross-val on best model (RF)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='f1', n_jobs=-1)
    results['Random Forest']['cv_f1_mean'] = round(cv_scores.mean(), 4)
    results['Random Forest']['cv_f1_std']  = round(cv_scores.std(), 4)
    logger.info(f"RF 5-fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    save_metrics(results)
    return results


def load_best_model():
    """Load the best supervised model (Random Forest) for inference."""
    path = f'{MODEL_DIR}/random_forest.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError("Model not trained yet. Run train.py first.")
    return joblib.load(path)
