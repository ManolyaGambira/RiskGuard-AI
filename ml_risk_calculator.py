# pyrefly: ignore [missing-import]
from langchain.tools import tool
from transaction_data import (
    get_transaction_by_id,
    get_customer_transactions
)
from risk_fusion import map_score_to_level

@tool
def calculate_behavioral_risk(transaction_id: str) -> str:
    """
    Calculate a deterministic behavioral risk score
    using the customer's historical transaction behavior.
    Handles missing history (insufficient_history) gracefully.
    """
    target = get_transaction_by_id(transaction_id)

    if not target:
        return f"No transaction found with ID {transaction_id}."

    customer_id = target["customer_id"]
    all_history = get_customer_transactions(customer_id)
    previous = [t for t in all_history if t.get("transaction_id") != target["transaction_id"]]

    hour = int(target.get("hour", 12))
    unusual_hour = int(hour < 6 or hour > 23)

    if not previous:
        score = 0
        reasons = ["No previous customer history available (new customer). Insufficient historical evidence."]
        if unusual_hour:
            score += 10
            reasons.append("Transaction occurred during an unusual hour.")

        risk_level = map_score_to_level(score)

        return (
            f"Transaction: {transaction_id}\n"
            f"Behavioral Risk Score: {score}/100\n"
            f"Behavioral Risk Level: {risk_level}\n"
            f"History Status: insufficient_history\n"
            f"Historical Average: N/A\n"
            f"Previous Maximum: N/A\n"
            f"Amount Ratio: N/A\n"
            f"Device Changed: 0 (no baseline)\n"
            f"Location Changed: 0 (no baseline)\n"
            f"Unusual Hour: {unusual_hour}\n"
            f"Payment Method Changed: 0 (no baseline)\n"
            f"Reasons: {' | '.join(reasons)}"
        )

    amount = float(target["amount"])
    prev_amounts = [float(t["amount"]) for t in previous]
    historical_avg = sum(prev_amounts) / len(prev_amounts) if prev_amounts else 0.0
    previous_max = max(prev_amounts) if prev_amounts else 0.0

    amount_ratio = amount / historical_avg if historical_avg > 0 else 0.0

    known_devices = {str(t.get("device", "")).lower() for t in previous}
    device_changed = int(str(target.get("device", "")).lower() not in known_devices)

    known_locations = {str(t.get("location", "")).lower() for t in previous}
    location_changed = int(str(target.get("location", "")).lower() not in known_locations)

    known_payment_methods = {str(t.get("payment_method", "")).lower() for t in previous}
    payment_method_changed = int(str(target.get("payment_method", "")).lower() not in known_payment_methods)

    # DETERMINISTIC BEHAVIORAL SCORE
    score = 0
    reasons = []

    if amount_ratio >= 5:
        score += 35
        reasons.append(f"Transaction amount is {amount_ratio:.1f}x historical average.")
    elif amount_ratio >= 3:
        score += 25
        reasons.append(f"Transaction amount is {amount_ratio:.1f}x historical average.")
    elif amount_ratio >= 2:
        score += 15
        reasons.append(f"Transaction amount is {amount_ratio:.1f}x historical average.")

    if amount > previous_max and previous_max > 0:
        score += 20
        reasons.append("Transaction amount exceeds previous maximum transaction.")

    if device_changed:
        score += 15
        reasons.append("Transaction uses a device not previously observed for this customer.")

    if location_changed:
        score += 15
        reasons.append("Transaction occurs from a location not previously observed for this customer.")

    if unusual_hour:
        score += 10
        reasons.append("Transaction occurred during an unusual hour.")

    if payment_method_changed:
        score += 5
        reasons.append("Transaction uses a payment method not previously observed for this customer.")

    score = min(score, 100)
    risk_level = map_score_to_level(score)

    if not reasons:
        reasons.append("No significant behavioral anomalies were detected.")

    return (
        f"Transaction: {transaction_id}\n"
        f"Behavioral Risk Score: {score}/100\n"
        f"Behavioral Risk Level: {risk_level}\n"
        f"History Status: available\n"
        f"Historical Average: ₹{historical_avg:.2f}\n"
        f"Previous Maximum: ₹{previous_max:.2f}\n"
        f"Amount Ratio: {amount_ratio:.2f}\n"
        f"Device Changed: {device_changed}\n"
        f"Location Changed: {location_changed}\n"
        f"Unusual Hour: {unusual_hour}\n"
        f"Payment Method Changed: {payment_method_changed}\n"
        f"Reasons: {' | '.join(reasons)}"
    )