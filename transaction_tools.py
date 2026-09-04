# pyrefly: ignore [missing-import]
from langchain.tools import tool
from transaction_data import (
    get_transaction_by_id,
    get_customer_transactions,
    get_device_transactions
)
from velocity_engine import calculate_velocity_risk as calc_velocity
from anomaly_engine import calculate_anomaly_score as calc_anomaly

@tool
def get_transaction(transaction_id: str) -> str:
    """Retrieve complete transaction information using a transaction ID (TXN100X or BENCH-XXXXXX)."""
    tx = get_transaction_by_id(transaction_id)
    if not tx:
        return f"No transaction found with ID {transaction_id}."
    return str(tx)

@tool
def get_customer_history(customer_id: str) -> str:
    """Retrieve previous transactions belonging to a customer."""
    history = get_customer_transactions(customer_id)
    if not history:
        return f"No transaction history found for customer {customer_id}. Status: insufficient_history."
    return str(history)

@tool
def get_device_history(device: str) -> str:
    """Retrieve previous transactions made using a specific device."""
    history = get_device_transactions(device)
    if not history:
        return f"No transaction history found for device {device}."
    return str(history)

@tool
def calculate_velocity_risk(transaction_id: str) -> str:
    """Calculate transaction velocity metrics and surge risk for a transaction ID."""
    tx = get_transaction_by_id(transaction_id)
    if not tx:
        return f"Transaction {transaction_id} not found."
    
    cust_history = get_customer_transactions(tx["customer_id"])
    past_history = [h for h in cust_history if h.get("transaction_id") != transaction_id]
    
    result = calc_velocity(tx, past_history)
    return (
        f"Transaction: {transaction_id}\n"
        f"Velocity Score: {result['velocity_score']}/100\n"
        f"1-Min Volume: {result['count_1m']}\n"
        f"5-Min Volume: {result['count_5m']}\n"
        f"1-Hour Volume: {result['count_1h']}\n"
        f"1-Hour Cumulative Amount: ₹{result['amount_1h']:.2f}\n"
        f"Reasons: {' | '.join(result['reasons'])}"
    )

@tool
def get_customer_profile(customer_id: str) -> str:
    """Get customer risk profile summary including transaction counts, average amount, common devices, and locations."""
    history = get_customer_transactions(customer_id)
    if not history:
        return (
            f"Customer Profile: {customer_id}\n"
            f"History Status: insufficient_history\n"
            f"Total Transactions: 0\n"
            f"Average Amount: N/A\n"
            f"Known Devices: None\n"
            f"Known Locations: None"
        )
        
    amounts = [float(h.get("amount", 0.0)) for h in history]
    devices = list({h.get("device", "Unknown") for h in history})
    locations = list({h.get("location", "Unknown") for h in history})
    payment_methods = list({h.get("payment_method", "Unknown") for h in history})
    
    avg_amt = sum(amounts) / len(amounts) if amounts else 0.0
    max_amt = max(amounts) if amounts else 0.0
    
    return (
        f"Customer Profile: {customer_id}\n"
        f"History Status: available\n"
        f"Total Transactions: {len(history)}\n"
        f"Average Transaction Amount: ₹{avg_amt:.2f}\n"
        f"Maximum Transaction Amount: ₹{max_amt:.2f}\n"
        f"Known Devices: {', '.join(devices)}\n"
        f"Known Locations: {', '.join(locations)}\n"
        f"Payment Methods: {', '.join(payment_methods)}"
    )

@tool
def get_device_profile(device: str) -> str:
    """Get device intelligence summary including unique customer count and shared usage signals."""
    history = get_device_transactions(device)
    if not history:
        return f"Device Intelligence: {device}\nStatus: Unseen device. No historical transactions."
        
    customers = list({h.get("customer_id") for h in history})
    tx_count = len(history)
    
    shared_signal = "High connectivity cluster signal (shared across multiple customers)" if len(customers) > 2 else "Normal device usage pattern"
    
    return (
        f"Device Intelligence: {device}\n"
        f"Total Device Transactions: {tx_count}\n"
        f"Associated Customers ({len(customers)}): {', '.join(customers[:5])}\n"
        f"Risk Cluster Signal: {shared_signal}"
    )