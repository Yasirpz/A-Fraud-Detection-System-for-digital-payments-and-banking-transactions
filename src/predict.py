"""
predict.py
----------
Bare-bones Command-Line Interface (CLI) for the Fraud Detection AI/ML module.
This is the "functional AI/ML module running via CLI" requested for Lab 1 --
it loads the trained model and scores a single transaction in real time,
then maps the risk score onto the Actuators defined in the PEAS specification
(Approve / Hold for Review / Block + Alert).

Usage examples:
    # 1) Run built-in demo scenarios (no arguments needed)
    python src/predict.py --demo

    # 2) Score a transaction by passing feature values directly
    python src/predict.py --amount 45000 --merchant crypto_exchange --hour 2 \
        --geo_mismatch 1 --new_device 1 --behavioral_score 0.8 \
        --velocity_1h 7 --distance_km 900 --account_age_days 120

    # 3) Score a transaction from a JSON file
    python src/predict.py --json_input sample_transaction.json

    # 4) Interactive prompt mode
    python src/predict.py --interactive
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from data_loader import ALL_FEATURES

# Decision thresholds derived from the assignment's Utility Function:
#   U = 0.7 * Fraud_Caught - 0.3 * False_Positive_Rate
# Higher BLOCK_THRESHOLD favors customer experience (fewer false declines);
# lower values favor stricter fraud prevention. Tune these to explore the
# security-vs-convenience trade-off discussed in Task 3.
HOLD_THRESHOLD = 0.35   # probability above this -> hold for manual review
BLOCK_THRESHOLD = 0.70  # probability above this -> auto-block + alert


def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with:\n"
            f"    python src/train.py"
        )
    return joblib.load(path)


def decide(risk_score: float) -> dict:
    """Maps a fraud probability to the Actuators described in the PEAS spec."""
    if risk_score >= BLOCK_THRESHOLD:
        return {
            "decision": "BLOCK",
            "actuator": "Transaction Blocking Module -> DECLINE",
            "alert": "Alert Generation System -> SMS + push notification sent to cardholder and fraud analyst dashboard",
            "case_management": "Case Management Interface -> new high-priority case opened",
        }
    elif risk_score >= HOLD_THRESHOLD:
        return {
            "decision": "HOLD_FOR_REVIEW",
            "actuator": "Account Restriction Actuator -> transaction held pending manual review",
            "alert": "Alert Generation System -> push notification sent to fraud analyst dashboard",
            "case_management": "Case Management Interface -> case opened with medium priority",
        }
    else:
        return {
            "decision": "APPROVE",
            "actuator": "Transaction Blocking/Approval Module -> APPROVE",
            "alert": "No alert generated",
            "case_management": "No case opened",
        }


def score_transaction(pipeline, transaction: dict) -> dict:
    row = pd.DataFrame([transaction])[ALL_FEATURES]

    start = time.perf_counter()
    proba = pipeline.predict_proba(row)[0][1]
    latency_ms = (time.perf_counter() - start) * 1000

    outcome = decide(proba)
    outcome["risk_score"] = round(float(proba), 4)
    outcome["latency_ms"] = round(latency_ms, 3)
    return outcome


DEMO_SCENARIOS = [
    {
        "label": "Everyday grocery purchase",
        "transaction_amount": 32.50, "merchant_category": "grocery", "hour_of_day": 18,
        "geo_ip_mismatch": 0, "new_device_fingerprint": 0, "behavioral_anomaly_score": 0.08,
        "txn_velocity_last_1h": 1, "distance_from_home_km": 2.3, "account_age_days": 900,
    },
    {
        "label": "Suspicious 3AM crypto purchase, new device, foreign IP",
        "transaction_amount": 48000.00, "merchant_category": "crypto_exchange", "hour_of_day": 3,
        "geo_ip_mismatch": 1, "new_device_fingerprint": 1, "behavioral_anomaly_score": 0.88,
        "txn_velocity_last_1h": 8, "distance_from_home_km": 1450.0, "account_age_days": 60,
    },
    {
        "label": "Mid-risk: unusual jewelry purchase, same device",
        "transaction_amount": 6200.00, "merchant_category": "jewelry", "hour_of_day": 14,
        "geo_ip_mismatch": 0, "new_device_fingerprint": 0, "behavioral_anomaly_score": 0.45,
        "txn_velocity_last_1h": 2, "distance_from_home_km": 60.0, "account_age_days": 500,
    },
]


def run_demo(pipeline):
    print("=" * 70)
    print("FRAUD DETECTION SYSTEM -- DEMO MODE (bare-bones CLI, Lab 1 module)")
    print("=" * 70)
    for scenario in DEMO_SCENARIOS:
        label = scenario.pop("label")
        result = score_transaction(pipeline, scenario)
        print(f"\nScenario: {label}")
        print(f"  Input        : {scenario}")
        print(f"  Risk score   : {result['risk_score']}")
        print(f"  Decision     : {result['decision']}")
        print(f"  Actuator     : {result['actuator']}")
        print(f"  Alert        : {result['alert']}")
        print(f"  Case mgmt    : {result['case_management']}")
        print(f"  Latency (ms) : {result['latency_ms']}")


def run_interactive(pipeline):
    print("Interactive fraud scoring -- enter transaction details:")
    transaction = {
        "transaction_amount": float(input("  Transaction amount: ")),
        "merchant_category": input("  Merchant category (grocery/electronics/fuel/travel/"
                                    "online_retail/utility_bill/restaurant/atm_withdrawal/"
                                    "jewelry/crypto_exchange): ").strip(),
        "hour_of_day": int(input("  Hour of day (0-23): ")),
        "geo_ip_mismatch": int(input("  Geo/IP mismatch? (0/1): ")),
        "new_device_fingerprint": int(input("  New/unrecognized device? (0/1): ")),
        "behavioral_anomaly_score": float(input("  Behavioral anomaly score (0.0-1.0): ")),
        "txn_velocity_last_1h": int(input("  Transactions in last 1 hour: ")),
        "distance_from_home_km": float(input("  Distance from home (km): ")),
        "account_age_days": int(input("  Account age (days): ")),
    }
    result = score_transaction(pipeline, transaction)
    print("\n--- Result ---")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Score a transaction with the fraud detection model.")
    parser.add_argument("--model", default="models/fraud_model.joblib")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo scenarios")
    parser.add_argument("--interactive", action="store_true", help="Prompt for transaction fields")
    parser.add_argument("--json_input", type=str, help="Path to a JSON file with one transaction")

    parser.add_argument("--amount", type=float)
    parser.add_argument("--merchant", type=str)
    parser.add_argument("--hour", type=int)
    parser.add_argument("--geo_mismatch", type=int)
    parser.add_argument("--new_device", type=int)
    parser.add_argument("--behavioral_score", type=float)
    parser.add_argument("--velocity_1h", type=int)
    parser.add_argument("--distance_km", type=float)
    parser.add_argument("--account_age_days", type=int)

    args = parser.parse_args()
    pipeline = load_model(args.model)

    if args.demo:
        run_demo(pipeline)
        return

    if args.interactive:
        run_interactive(pipeline)
        return

    if args.json_input:
        with open(args.json_input) as f:
            transaction = json.load(f)
        result = score_transaction(pipeline, transaction)
        print(json.dumps(result, indent=2))
        return

    if args.amount is not None:
        transaction = {
            "transaction_amount": args.amount,
            "merchant_category": args.merchant,
            "hour_of_day": args.hour,
            "geo_ip_mismatch": args.geo_mismatch,
            "new_device_fingerprint": args.new_device,
            "behavioral_anomaly_score": args.behavioral_score,
            "txn_velocity_last_1h": args.velocity_1h,
            "distance_from_home_km": args.distance_km,
            "account_age_days": args.account_age_days,
        }
        missing = [k for k, v in transaction.items() if v is None]
        if missing:
            parser.error(f"Missing required fields for scoring: {missing}")
        result = score_transaction(pipeline, transaction)
        print(json.dumps(result, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
