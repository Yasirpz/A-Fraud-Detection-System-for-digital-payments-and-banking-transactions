"""
generate_dataset.py
--------------------
Generates a synthetic, realistic banking/payments transaction dataset for the
Fraud Detection System (PEAS System G) assignment.

Why synthetic data?
Real fraud  datasets (e.g. Kaggle's "Credit Card Fraud Detection") cannot be
downloaded automatically in every environment and are heavily PCA-anonymized,
which makes the feature set unreadable for a course demo. This generator
instead builds a labeled dataset whose columns map DIRECTLY onto the Sensors
and Environment variables described in Task 1 of the PEAS specification
(transaction amount, merchant category, geolocation/IP mismatch, device
fingerprint change, behavioral biometrics, time-of-day, account velocity),
so the model, features, and report all tell one consistent story.

Run:
    python data/generate_dataset.py --n 20000 --fraud_ratio 0.02 --out data/transactions.csv
"""

import argparse
import numpy as np
import pandas as pd


MERCHANT_CATEGORIES = [
    "grocery", "electronics", "fuel", "travel", "online_retail",
    "utility_bill", "restaurant", "atm_withdrawal", "jewelry", "crypto_exchange"
]

HIGH_RISK_CATEGORIES = {"jewelry", "crypto_exchange", "electronics", "travel"}


def generate_transactions(n_samples: int, fraud_ratio: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def build_block(n, is_fraud: bool):
        merchant = rng.choice(MERCHANT_CATEGORIES, size=n,
                               p=_merchant_probs(is_fraud))
        hour = _hour_distribution(rng, n, is_fraud)

        if is_fraud:
            amount = rng.lognormal(mean=6.2, sigma=1.1, size=n)          # larger, spikier amounts
            geo_ip_mismatch = rng.binomial(1, 0.65, size=n)              # billing vs IP country mismatch
            new_device = rng.binomial(1, 0.70, size=n)                   # unrecognized device fingerprint
            behavioral_anomaly = np.clip(rng.normal(0.75, 0.15, size=n), 0, 1)  # typing/touch deviation score
            velocity_1h = rng.poisson(6, size=n)                         # transactions in last 1 hour
            distance_from_home_km = rng.exponential(scale=800, size=n)
            account_age_days = rng.integers(1, 4000, size=n)
        else:
            amount = rng.lognormal(mean=3.8, sigma=0.9, size=n)
            geo_ip_mismatch = rng.binomial(1, 0.03, size=n)
            new_device = rng.binomial(1, 0.05, size=n)
            behavioral_anomaly = np.clip(rng.normal(0.15, 0.10, size=n), 0, 1)
            velocity_1h = rng.poisson(0.8, size=n)
            distance_from_home_km = rng.exponential(scale=15, size=n)
            account_age_days = rng.integers(1, 4000, size=n)

        df = pd.DataFrame({
            "transaction_amount": np.round(amount, 2),
            "merchant_category": merchant,
            "hour_of_day": hour,
            "geo_ip_mismatch": geo_ip_mismatch,
            "new_device_fingerprint": new_device,
            "behavioral_anomaly_score": np.round(behavioral_anomaly, 3),
            "txn_velocity_last_1h": velocity_1h,
            "distance_from_home_km": np.round(distance_from_home_km, 2),
            "account_age_days": account_age_days,
        })
        df["is_fraud"] = int(is_fraud)
        return df

    def _merchant_probs(is_fraud):
        weights = np.array([3 if m in HIGH_RISK_CATEGORIES else 1 for m in MERCHANT_CATEGORIES], dtype=float)
        if not is_fraud:
            weights = np.ones(len(MERCHANT_CATEGORIES))
        return weights / weights.sum()

    def _hour_distribution(rng, n, is_fraud):
        if is_fraud:
            # fraud skews toward late night / early morning
            hours = rng.choice(range(24), size=n,
                                p=_night_weighted_probs())
        else:
            hours = rng.choice(range(24), size=n, p=_day_weighted_probs())
        return hours

    def _night_weighted_probs():
        base = np.ones(24)
        for h in [0, 1, 2, 3, 4, 23]:
            base[h] = 4
        return base / base.sum()

    def _day_weighted_probs():
        base = np.ones(24)
        for h in [9, 10, 11, 12, 13, 17, 18, 19, 20]:
            base[h] = 3
        return base / base.sum()

    fraud_df = build_block(n_fraud, True)
    legit_df = build_block(n_legit, False)

    full = pd.concat([fraud_df, legit_df], axis=0, ignore_index=True)
    full = full.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    full.insert(0, "transaction_id", [f"TXN{100000 + i}" for i in range(len(full))])
    return full


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud detection dataset.")
    parser.add_argument("--n", type=int, default=20000, help="Total number of transactions")
    parser.add_argument("--fraud_ratio", type=float, default=0.02, help="Proportion of fraudulent transactions")
    parser.add_argument("--out", type=str, default="data/transactions.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_transactions(args.n, args.fraud_ratio, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df):,} transactions -> {args.out}")
    print(f"Fraud cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
