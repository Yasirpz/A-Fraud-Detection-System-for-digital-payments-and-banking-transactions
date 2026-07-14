# Fraud Detection System for Digital Payments & Banking Transactions

**Course:** Introduction to Artificial Intelligence
**Assignment 1 topic:** PEAS Framework & Task Environment Analysis (Russell & Norvig, Ch. 2)
**System selected:** (G) Fraud Detection System for Digital Payments and Banking Transactions

| | |
|---|---|
| **Leader** | Yasir Parveez — Roll No: `2K23/CSM/146` — GitHub: [@Yasirpz](https://github.com/Yasirpz) |
| **Member** | Abdul Fatah — Roll No: `2K23/CSM/03` |

---

## 1. What this repository is

This repo contains the **working, runnable AI/ML module** that backs  the PEAS
specification from Assignment 1 (see [`PEAS_specification.json`](PEAS_specification.json)).
It was built to satisfy the Lab 1 clarification below (Option A: implementing
the module that will later be used in the FYP): a **bare-bones, functional
AI/ML module runnable via a CLI or simple script** — no frontend/backend
integration required at this stage.

It is a full, self-contained fraud-scoring pipeline:

1. **Synthetic data generator** whose columns map directly onto the Sensors
   and Environment described in the PEAS spec (transaction amount, merchant
   category, geo/IP mismatch, device fingerprint, behavioral biometrics,
   account velocity, etc.).
2. **Training pipeline** — a `RandomForestClassifier` (class-balanced) inside
   a scikit-learn `Pipeline`, evaluated with the same Performance Measures
   named in Task 1 (recall, false-positive rate, latency, etc.).
3. **CLI inference tool** that scores a transaction and maps the risk score
   onto the Actuators from the PEAS spec (Approve / Hold for Review / Block +
   Alert + Case Management), using the Task 3 utility function
   `U = 0.7 * Fraud_Caught − 0.3 * False_Positive_Rate` to set decision
   thresholds.

---

## 2. Project structure

```
fraud-detection-ai/
├── main.py                     # Unified CLI entry point (generate / train / predict)
├── requirements.txt
├── PEAS_specification.json      # Structured PEAS + environment classification (Task 4)
├── sample_transaction.json      # Example transaction for CLI JSON-input mode
├── data/
│   └── generate_dataset.py      # Synthetic transaction dataset generator
├── src/
│   ├── data_loader.py           # Shared feature schema + train/test split
│   ├── train.py                 # Model training + evaluation
│   └── predict.py               # CLI inference (demo / interactive / json / flags)
├── models/                      # Trained model artifact lands here (fraud_model.joblib)
├── reports/                     # metrics.json written after training
└── tests/
    └── test_pipeline.py         # Smoke tests (pytest)
```

---

## 3. Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/Yasirpz/fraud-detection-ai.git
cd fraud-detection-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the synthetic transaction dataset (~20,000 transactions, 2% fraud rate)
python main.py generate

# 4. Train the model
python main.py train

# 5. Score transactions via the CLI
python main.py predict --demo
```

### Other ways to run inference

```bash
# Interactive prompt mode
python main.py predict --interactive

# Score a transaction from a JSON file
python main.py predict --json_input sample_transaction.json

# Score a transaction by passing feature values as flags
python main.py predict --amount 48000 --merchant crypto_exchange --hour 3 \
    --geo_mismatch 1 --new_device 1 --behavioral_score 0.85 \
    --velocity_1h 7 --distance_km 1300 --account_age_days 45
```

### Run the tests

```bash
pytest -q
```

---

## 4. Example output

```
$ python main.py predict --demo
======================================================================
FRAUD DETECTION SYSTEM -- DEMO MODE (bare-bones CLI, Lab 1 module)
======================================================================

Scenario: Everyday grocery purchase
  Risk score   : 0.0
  Decision     : APPROVE
  Actuator     : Transaction Blocking/Approval Module -> APPROVE

Scenario: Suspicious 3AM crypto purchase, new device, foreign IP
  Risk score   : 0.9933
  Decision     : BLOCK
  Actuator     : Transaction Blocking Module -> DECLINE
  Alert        : Alert Generation System -> SMS + push notification sent...
  Case mgmt    : Case Management Interface -> new high-priority case opened
```

---

## 5. Design notes (why these choices)

- **Why synthetic data instead of a public dataset?** Public fraud datasets
  (e.g. Kaggle's credit-card fraud set) are PCA-anonymized, so their columns
  (`V1`...`V28`) can't be tied back to the PEAS Sensors/Environment described
  in Assignment 1. The generator here builds interpretable, labeled features
  (amount, merchant, geo/IP mismatch, device fingerprint, behavioral score,
  velocity) that map one-to-one onto the PEAS write-up, so the report and the
  code tell one consistent story.
- **Why RandomForest?** It handles the mixed numeric/categorical feature set
  well, gives calibrated-enough probabilities for threshold tuning, and
  trains in seconds — appropriate for a bare-bones Lab 1 deliverable that
  will be extended later in the FYP.
- **Why `class_weight="balanced"`?** Task 3 identifies severe class imbalance
  (real fraud is rare) as the central technical challenge; balancing the
  class weights prevents the model from trivially predicting "legitimate"
  for everything.
- **Threshold tuning maps directly to the Utility Function trade-off**
  (`HOLD_THRESHOLD` / `BLOCK_THRESHOLD` in `src/predict.py`). Raising
  `BLOCK_THRESHOLD` favors customer experience (fewer false declines);
  lowering it favors stricter fraud prevention — exactly the trade-off
  analyzed in Task 3.2 of the assignment.

---

## 6. Pushing this repo to GitHub

```bash
cd fraud-detection-ai
git init                     # skip if already a git repo
git add .
git commit -m "Fraud Detection AI System - PEAS Assignment 1 + Lab 1 module"
git branch -M main
git remote add origin https://github.com/Yasirpz/fraud-detection-ai.git
git push -u origin main
```

---

## 7. License

MIT License — see [LICENSE](LICENSE).
