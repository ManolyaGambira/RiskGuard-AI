import pandas as pd
from pathlib import Path
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
# pyrefly: ignore [missing-import]
from langchain.tools import tool
from risk_fusion import map_score_to_level
from transaction_data import get_benchmark_df, get_transaction_by_id

MODEL_FILE = Path(__file__).parent / "risk_model.json"
DATA_FILE = Path(__file__).parent / "data" / "real" / "creditcard.csv"

model = XGBClassifier()
if MODEL_FILE.exists():
    model.load_model(MODEL_FILE)

FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

@tool
def calculate_real_ml_risk(row_index: str) -> str:
    """
    Calculate fraud risk for a transaction from the real benchmark dataset using XGBoost ML model.
    Accepts row index (e.g. '541') or benchmark ID (e.g. 'BENCH-000542').
    """
    idx_str = str(row_index).strip()
    index = None

    if idx_str.upper().startswith("BENCH-"):
        try:
            index = int(idx_str.split("-")[1]) - 1
        except ValueError:
            pass
    elif idx_str.isdigit():
        index = int(idx_str)

    if index is None:
        # Check if transaction_id lookup can resolve a benchmark_row_index
        tx = get_transaction_by_id(idx_str)
        if tx and tx.get("benchmark_row_index") is not None:
            index = tx["benchmark_row_index"]

    if index is None:
        return f"Invalid benchmark row index or ID '{row_index}'. Please provide an integer row index or valid BENCH-XXXXXX ID."

    df = get_benchmark_df()

    if index < 0 or index >= len(df):
        return f"No transaction found at benchmark row index {index}."

    transaction = df.iloc[index]

    features = pd.DataFrame(
        [[transaction[column] for column in FEATURES]],
        columns=FEATURES
    )

    fraud_probability = float(model.predict_proba(features)[0][1])
    risk_score = round(fraud_probability * 100)
    risk_level = map_score_to_level(risk_score)
    actual_class = int(transaction["Class"])

    return (
        f"Benchmark Transaction: BENCH-{index+1:06d} (Row {index})\n"
        f"Amount: ₹{float(transaction['Amount']):.2f}\n"
        f"Fraud Probability: {fraud_probability:.6f}\n"
        f"ML Risk Score: {risk_score}/100\n"
        f"ML Risk Level: {risk_level}\n"
        f"Actual Dataset Label: {actual_class}"
    )