# Credit Card Fraud Detection — Semi-Supervised Learning

## Architecture

This project uses a semi-supervised approach:

1. **Autoencoder (TensorFlow)** — Trained only on legitimate transactions. Generates a reconstruction error score for every transaction. High error means suspicious transaction.
2. **XGBoost Classifier** — Trained on original features plus reconstruction error from autoencoder. Makes the final fraud or legitimate decision.
3. **SMOTE** — Applied to balance the training data before XGBoost training.

---

## Dataset

kartik2112/fraud-detection from Kaggle — real simulated credit card transactions with interpretable features like merchant, category, amount, customer location, and job.

- Training samples: 1,296,675
- Test samples: 555,719
- Training fraud rate: 0.58% (7,506 fraud cases)
- Test fraud rate: 0.39% (2,145 fraud cases)

---

## Features Engineered

- hour — transaction hour extracted from timestamp
- day_of_week — day of week from timestamp
- age — customer age calculated from date of birth
- distance — geographic distance between customer and merchant
- tx_velocity — cumulative transaction count per card
- amt_deviation — deviation of transaction amount from card average
- tx_per_day — number of transactions per card per day
- Target encoding for job, category, merchant, state

---

## Results

| Metric    | CV Set | Test Set |
|-----------|--------|----------|
| F1 Score  | 0.922  | 0.744    |
| Precision | 0.953  | 0.845    |
| Recall    | 0.894  | 0.665    |

---

## Why is there a gap between CV and Test F1?

The CV F1 is 0.922 but Test F1 is 0.744. This gap is caused by concept drift.

The training data covers January 2019 to December 2020 while the test data covers 2021. During this time the fraud patterns changed. Specifically the fraud rate dropped from 0.58% in training to 0.39% in test. This means fraudsters changed their behavior between the two periods.

The model learned patterns from 2019-2020 data but test transactions follow slightly different distributions. This is a well known real-world challenge in fraud detection. Production fraud systems handle this by continuously retraining models with new data as fraud patterns evolve.

This gap is not a model failure — it is an honest reflection of how fraud detection works in the real world.

---

## Project Structure

```
Credit-Card-Fraud-Detection/
├── fraud_detection_final.py   # Training pipeline
├── backend.py                 # FastAPI backend
├── frontend.py                # Streamlit frontend
├── autoencoder_fraud.keras    # Trained autoencoder
├── xgb_model.json             # Trained XGBoost model
├── scaler.pkl                 # StandardScaler
├── target_encoders.pkl        # Target encoding mappings
├── best_threshold.pkl         # Tuned classification threshold
└── README.md
```

---

## Setup and Run

### Install dependencies
```bash
pip install numpy pandas scikit-learn tensorflow xgboost imbalanced-learn fastapi uvicorn streamlit kagglehub
```

### Train models
```bash
python fraud_detection_final.py
```

### Start backend
```bash
uvicorn backend:app --reload
```

### Start frontend (new terminal)
```bash
streamlit run frontend.py
```

---

## Tech Stack

- Python 3.13
- TensorFlow 2.21 — Autoencoder
- XGBoost — Classifier
- SMOTE (imbalanced-learn) — Class balancing
- FastAPI — REST API backend
- Streamlit — Web UI
- Scikit-learn — Preprocessing and evaluation

---

## Key Learnings

- Unsupervised anomaly detection alone achieves 0.47 F1 on this dataset
- Combining autoencoder features with XGBoost pushes F1 to 0.922 on CV
- Concept drift between training and test periods is a real challenge in fraud detection
- SMOTE effectively handles class imbalance
- Behavioral features like velocity, deviation and tx_per_day significantly improve detection




**LinkedIn:** [https://www.linkedin.com/in/rakesh-purbiya-0b7091317/]  
**Live Demo:** [https://huggingface.co/spaces/rakesh9773/credit-card-fraud-detection]

---
