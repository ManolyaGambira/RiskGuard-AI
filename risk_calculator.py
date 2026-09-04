import csv
from pathlib import Path
# pyrefly: ignore [missing-import]
from langchain.tools import tool


DATA_FILE = Path(__file__).parent / "data" / "transactions.csv"


def load_transactions():
    """Load all transactions from the CSV file."""

    with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@tool
def calculate_risk(transaction_id: str) -> str:
    """Calculate a deterministic risk score for a transaction using customer behavior."""

    transactions = load_transactions()

    # Find the target transaction
    target = None

    for transaction in transactions:
        if transaction["transaction_id"] == transaction_id:
            target = transaction
            break

    if target is None:
        return f"No transaction found with ID {transaction_id}."

    customer_id = target["customer_id"]

    # Get previous transactions for this customer
    previous = [
        t for t in transactions
        if t["customer_id"] == customer_id
        and t["transaction_id"] != transaction_id
    ]

    if not previous:
        target_hour = int(target.get("hour", 12))
        score = 0
        reasons = ["No previous customer history available (new customer). Insufficient historical evidence."]
        if target_hour < 6 or target_hour > 23:
            score += 10
            reasons.append("Transaction occurred during an unusual hour.")
        
        return (
            f"Transaction: {transaction_id}\n"
            f"Risk Score: {score}/100\n"
            f"Risk Level: LOW\n"
            f"History Status: insufficient_history\n"
            f"Historical Average: N/A\n"
            f"Previous Maximum: N/A\n"
            f"Reasons:\n- " + "\n- ".join(reasons)
        )

    amount = float(target["amount"])

    previous_amounts = [
        float(t["amount"])
        for t in previous
    ]

    average_amount = sum(previous_amounts) / len(previous_amounts)
    maximum_amount = max(previous_amounts)

    score = 0
    reasons = []

    # --------------------------------
    # 1. Amount anomaly
    # --------------------------------
    amount_ratio = amount / average_amount

    if amount_ratio >= 10:
        score += 50
        reasons.append(
            f"Transaction amount is {amount_ratio:.1f}× "
            f"the customer's historical average."
        )

    elif amount_ratio >= 5:
        score += 35
        reasons.append(
            f"Transaction amount is {amount_ratio:.1f}× "
            f"the customer's historical average."
        )

    elif amount_ratio >= 3:
        score += 20
        reasons.append(
            f"Transaction amount is {amount_ratio:.1f}× "
            f"the customer's historical average."
        )

    # --------------------------------
    # 2. Previous maximum
    # --------------------------------
    if amount > maximum_amount * 5:
        score += 30
        reasons.append(
            f"Transaction amount is more than 5× "
            f"the customer's previous maximum of ₹{maximum_amount:.0f}."
        )

    elif amount > maximum_amount * 3:
        score += 20
        reasons.append(
            f"Transaction amount is more than 3× "
            f"the customer's previous maximum of ₹{maximum_amount:.0f}."
        )

    # --------------------------------
    # 3. Device anomaly
    # --------------------------------
    known_devices = {
        t["device"].lower()
        for t in previous
    }

    if target["device"].lower() not in known_devices:
        score += 15
        reasons.append("Transaction uses a previously unseen device.")

    # --------------------------------
    # 4. Location anomaly
    # --------------------------------
    known_locations = {
        t["location"].lower()
        for t in previous
    }

    if target["location"].lower() not in known_locations:
        score += 15
        reasons.append("Transaction uses a previously unseen location.")

    # --------------------------------
    # 5. Unusual hour
    # --------------------------------
    target_hour = int(target["hour"])

    previous_hours = [
        int(t["hour"])
        for t in previous
    ]

    if target_hour < 6 or target_hour > 23:
        score += 10
        reasons.append("Transaction occurred during an unusual hour.")

    # --------------------------------
    # Final score
    # --------------------------------
    score = min(score, 100)

    if score >= 85:
        risk_level = "CRITICAL"
    elif score >= 70:
        risk_level = "HIGH"
    elif score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if not reasons:
        reasons.append("No major behavioral anomalies detected.")

    return (
        f"Transaction: {transaction_id}\n"
        f"Risk Score: {score}/100\n"
        f"Risk Level: {risk_level}\n"
        f"Historical Average: ₹{average_amount:.2f}\n"
        f"Previous Maximum: ₹{maximum_amount:.2f}\n"
        f"Reasons:\n- " + "\n- ".join(reasons)
    )