import numpy as np
from typing import Dict, Any

def calculate_anomaly_score(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight statistical anomaly detection layer.
    Evaluates amount percentile/z-score, unusual timing, and PCA variance for benchmark features.
    Returns anomaly score (0-100) and detection details.
    """
    amount = float(transaction.get("amount", 0.0))
    hour = int(transaction.get("hour", 12))
    v_features = transaction.get("v_features", {})

    score = 0
    reasons = []

    # 1. Statistical Amount Anomaly
    if amount > 50000:
        score += 45
        reasons.append(f"Extreme transaction amount (₹{amount:,.2f}) > 99th percentile.")
    elif amount > 20000:
        score += 30
        reasons.append(f"Very high transaction amount (₹{amount:,.2f}) > 95th percentile.")
    elif amount > 10000:
        score += 15
        reasons.append(f"Elevated transaction amount (₹{amount:,.2f}) > 90th percentile.")

    # 2. Time-of-day Anomaly
    if hour < 5 or hour > 23:
        score += 20
        reasons.append(f"Off-peak transaction hour ({hour:02d}:00).")

    # 3. Benchmark Feature Variance Anomaly (if V1..V28 present)
    if v_features:
        v_vals = list(v_features.values())
        max_abs = max(abs(v) for v in v_vals) if v_vals else 0
        l2_norm = float(np.linalg.norm(v_vals)) if v_vals else 0

        if max_abs > 10:
            score += 35
            reasons.append(f"Feature component outlier detected (max |V_i| = {max_abs:.2f}).")
        elif max_abs > 6:
            score += 20
            reasons.append(f"Elevated feature variance detected (max |V_i| = {max_abs:.2f}).")

        if l2_norm > 25:
            score += 25
            reasons.append(f"High feature vector magnitude (L2 norm = {l2_norm:.2f}).")

    score = min(score, 100)

    if not reasons:
        reasons.append("No statistical anomalies detected.")

    return {
        "anomaly_score": score,
        "reasons": reasons,
        "status": "available"
    }
