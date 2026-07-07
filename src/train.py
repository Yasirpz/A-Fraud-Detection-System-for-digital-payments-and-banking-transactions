"""
train.py
--------
Trains the core fraud-detection AI/ML module (the "Actuator-triggering" model
behind the Transaction Blocking/Approval Module described in the PEAS spec).

Model: RandomForestClassifier with class_weight="balanced" to address the
severe class imbalance identified in Task 3 (Critical Analysis) of the
assignment, wrapped in a scikit-learn Pipeline (OneHotEncoder + scaling)
so training and inference always use identical preprocessing.

Usage:
    python src/train.py --data data/transactions.csv --model_out models/fraud_model.joblib
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import (
    load_dataset, split_dataset, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    elapsed_ms = (time.perf_counter() - start) * 1000
    avg_latency_ms = elapsed_ms / max(len(X_test), 1)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall_fraud_detection_rate": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "avg_prediction_latency_ms": round(avg_latency_ms, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_set_size": int(len(X_test)),
        "test_set_fraud_count": int(y_test.sum()),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--model_out", default="models/fraud_model.joblib")
    parser.add_argument("--metrics_out", default="reports/metrics.json")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=args.test_size)

    print(f"Training RandomForest on {len(X_train):,} transactions "
          f"({y_train.sum():,} fraud cases)...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Evaluating on held-out test set...")
    metrics = evaluate(pipeline, X_test, y_test)
    print(json.dumps(metrics, indent=2))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)
    print(f"Model saved -> {args.model_out}")

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {args.metrics_out}")

    print("\nClassification report:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["legitimate", "fraud"]))


if __name__ == "__main__":
    main()
