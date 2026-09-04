from typing import Dict, Any, List

def calculate_velocity_risk(transaction: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate transaction velocity metrics and velocity risk score (0-100).
    Evaluates:
    - Recent transaction frequency (last 1 min, 5 mins, 1 hour)
    - Cumulative transaction amount in short time window
    - Burst payment attempts
    """
    if not history:
        return {
            "velocity_score": 0,
            "count_1m": 1,
            "count_5m": 1,
            "count_1h": 1,
            "amount_1h": float(transaction.get("amount", 0)),
            "reasons": ["Single transaction in window. Velocity normal."],
            "status": "available"
        }

    current_time = float(transaction.get("time", 0.0))
    current_amount = float(transaction.get("amount", 0.0))

    # Calculate time differences if timestamps/times exist
    times = [float(h.get("time", 0.0)) for h in history if "time" in h]
    amounts = [float(h.get("amount", 0.0)) for h in history]

    count_1m = 1
    count_5m = 1
    count_1h = 1
    sum_1h = current_amount

    if times:
        for t, a in zip(times, amounts):
            diff = abs(current_time - t)
            if diff <= 60:
                count_1m += 1
            if diff <= 300:
                count_5m += 1
            if diff <= 3600:
                count_1h += 1
                sum_1h += a
    else:
        # Fallback to order-based velocity if explicit timestamp delta is absent
        count_1h = len(history) + 1
        sum_1h = sum(amounts) + current_amount

    score = 0
    reasons = []

    if count_1m >= 5:
        score += 50
        reasons.append(f"High velocity burst: {count_1m} transactions within 1 minute.")
    elif count_1m >= 3:
        score += 30
        reasons.append(f"Moderate velocity burst: {count_1m} transactions within 1 minute.")

    if count_5m >= 10:
        score += 35
        reasons.append(f"High 5-minute velocity: {count_5m} transactions.")
    elif count_5m >= 5:
        score += 20
        reasons.append(f"Elevated 5-minute velocity: {count_5m} transactions.")

    if count_1h >= 15:
        score += 25
        reasons.append(f"High 1-hour transaction volume: {count_1h} transactions.")

    if sum_1h > current_amount * 10 and count_1h > 3:
        score += 20
        reasons.append(f"Rapid cumulative volume surge: ₹{sum_1h:.2f} in recent window.")

    score = min(score, 100)

    if not reasons:
        reasons.append("Transaction velocity within normal limits.")

    return {
        "velocity_score": score,
        "count_1m": count_1m,
        "count_5m": count_5m,
        "count_1h": count_1h,
        "amount_1h": round(sum_1h, 2),
        "reasons": reasons,
        "status": "available"
    }
