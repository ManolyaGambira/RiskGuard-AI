import csv
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR = Path(__file__).parent
DEMO_CSV = BASE_DIR / "data" / "transactions.csv"
REAL_CSV = BASE_DIR / "data" / "real" / "creditcard.csv"

_benchmark_df_cache: Optional[pd.DataFrame] = None

def get_benchmark_df() -> pd.DataFrame:
    global _benchmark_df_cache
    if _benchmark_df_cache is None:
        _benchmark_df_cache = pd.read_csv(REAL_CSV)
    return _benchmark_df_cache

def load_demo_transactions() -> List[Dict[str, Any]]:
    if not DEMO_CSV.exists():
        return []
    with open(DEMO_CSV, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def normalize_benchmark_row(row_idx: int, row: pd.Series) -> Dict[str, Any]:
    tx_id = f"BENCH-{row_idx + 1:06d}"
    cust_id = f"CUST-BENCH-{(row_idx % 1000) + 1:04d}"
    amount = float(row.get("Amount", 0.0))
    time_sec = float(row.get("Time", 0.0))
    hour = int((time_sec // 3600) % 24)
    actual_label = int(row.get("Class", 0)) if "Class" in row else None
    
    # Deterministic device assignment for benchmark demonstration
    devices = ["Android", "iPhone", "Windows", "Web", "Mac"]
    device = devices[row_idx % len(devices)]
    
    # Deterministic location assignment
    locations = ["Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai", "Kolkata"]
    location = locations[(row_idx * 7) % len(locations)]
    
    payment_methods = ["Credit Card", "UPI", "NetBanking", "Debit Card"]
    payment_method = payment_methods[(row_idx * 3) % len(payment_methods)]

    v_features = {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29) if f"V{i}" in row}

    return {
        "transaction_id": tx_id,
        "customer_id": cust_id,
        "amount": amount,
        "location": location,
        "device": device,
        "payment_method": payment_method,
        "hour": hour,
        "status": "completed",
        "source": "benchmark",
        "benchmark_row_index": row_idx,
        "actual_label": actual_label,
        "time": time_sec,
        "v_features": v_features
    }

def get_transaction_by_id(tx_id: str) -> Optional[Dict[str, Any]]:
    tx_id_str = str(tx_id).strip()
    
    # Check demo transactions
    demos = load_demo_transactions()
    for d in demos:
        if d["transaction_id"].upper() == tx_id_str.upper():
            return {
                "transaction_id": d["transaction_id"],
                "customer_id": d["customer_id"],
                "amount": float(d["amount"]),
                "location": d["location"],
                "device": d["device"],
                "payment_method": d["payment_method"],
                "hour": int(d["hour"]),
                "status": d.get("status", "completed"),
                "source": "demo",
                "benchmark_row_index": None,
                "actual_label": None
            }
            
    # Check benchmark transactions (format: BENCH-000542 or 541)
    df = get_benchmark_df()
    row_idx = None
    
    if tx_id_str.upper().startswith("BENCH-"):
        try:
            num = int(tx_id_str.split("-")[1])
            row_idx = num - 1
        except ValueError:
            pass
    elif tx_id_str.isdigit():
        row_idx = int(tx_id_str)
        
    if row_idx is not None and 0 <= row_idx < len(df):
        return normalize_benchmark_row(row_idx, df.iloc[row_idx])
        
    return None

def get_customer_transactions(customer_id: str) -> List[Dict[str, Any]]:
    results = []
    demos = load_demo_transactions()
    for d in demos:
        if d["customer_id"].upper() == customer_id.upper():
            results.append({
                "transaction_id": d["transaction_id"],
                "customer_id": d["customer_id"],
                "amount": float(d["amount"]),
                "location": d["location"],
                "device": d["device"],
                "payment_method": d["payment_method"],
                "hour": int(d["hour"]),
                "source": "demo"
            })
    
    if customer_id.upper().startswith("CUST-BENCH-"):
        df = get_benchmark_df()
        try:
            cust_num = int(customer_id.split("-")[-1]) - 1
            # Return matching benchmark rows
            for idx in range(cust_num, min(cust_num + 5000, len(df)), 1000):
                if idx < len(df):
                    results.append(normalize_benchmark_row(idx, df.iloc[idx]))
        except ValueError:
            pass
            
    return results

def get_device_transactions(device: str) -> List[Dict[str, Any]]:
    results = []
    demos = load_demo_transactions()
    for d in demos:
        if d["device"].lower() == device.lower():
            results.append({
                "transaction_id": d["transaction_id"],
                "customer_id": d["customer_id"],
                "amount": float(d["amount"]),
                "location": d["location"],
                "device": d["device"],
                "payment_method": d["payment_method"],
                "hour": int(d["hour"]),
                "source": "demo"
            })
    return results

def get_benchmark_batch(limit: int = 1000, start_index: int = 0) -> List[Dict[str, Any]]:
    df = get_benchmark_df()
    end_index = min(start_index + limit, len(df))
    batch = []
    for idx in range(start_index, end_index):
        batch.append(normalize_benchmark_row(idx, df.iloc[idx]))
    return batch
