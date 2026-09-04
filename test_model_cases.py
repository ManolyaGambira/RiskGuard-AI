import pandas as pd
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier


# Load trained model
model = XGBClassifier()
model.load_model("risk_model.json")


FEATURES = [
    "amount",
    "historical_avg",
    "amount_ratio",
    "device_changed",
    "location_changed",
    "unusual_hour",
    "velocity_1h",
    "payment_method_changed"
]


# Custom unseen test cases
test_cases = pd.DataFrame([
    {
        "case": "High amount, normal behavior",
        "amount": 50000,
        "historical_avg": 5000,
        "amount_ratio": 10.0,
        "device_changed": 0,
        "location_changed": 0,
        "unusual_hour": 0,
        "velocity_1h": 1,
        "payment_method_changed": 0
    },
    {
        "case": "Small amount, many suspicious signals",
        "amount": 800,
        "historical_avg": 1000,
        "amount_ratio": 0.8,
        "device_changed": 1,
        "location_changed": 1,
        "unusual_hour": 1,
        "velocity_1h": 7,
        "payment_method_changed": 1
    },
    {
        "case": "Moderate amount, new location",
        "amount": 7000,
        "historical_avg": 3000,
        "amount_ratio": 2.33,
        "device_changed": 0,
        "location_changed": 1,
        "unusual_hour": 0,
        "velocity_1h": 1,
        "payment_method_changed": 0
    },
    {
        "case": "Normal transaction",
        "amount": 1200,
        "historical_avg": 1500,
        "amount_ratio": 0.80,
        "device_changed": 0,
        "location_changed": 0,
        "unusual_hour": 0,
        "velocity_1h": 1,
        "payment_method_changed": 0
    }
])


X = test_cases[FEATURES]

probabilities = model.predict_proba(X)[:, 1]


print("\n=== UNSEEN TEST CASES ===")

for i, probability in enumerate(probabilities):

    risk_score = round(probability * 100)

    if risk_score >= 85:
        risk_level = "CRITICAL"
    elif risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    print(f"\nCase: {test_cases.iloc[i]['case']}")
    print(f"Fraud Probability: {probability:.4f}")
    print(f"Risk Score: {risk_score}/100")
    print(f"Risk Level: {risk_level}")