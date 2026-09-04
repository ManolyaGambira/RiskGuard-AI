import pandas as pd
# pyrefly: ignore [missing-import]
import shap
from pathlib import Path
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
# pyrefly: ignore [missing-import]
from langchain.tools import tool
from transaction_data import get_benchmark_df, get_transaction_by_id

MODEL_FILE = Path(__file__).parent / "risk_model.json"

FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

model = XGBClassifier()
if MODEL_FILE.exists():
    model.load_model(MODEL_FILE)

explainer = shap.TreeExplainer(model)

@tool
def get_shap_explanation(row_index: str) -> str:
    """
    Explain the main feature contributions for a transaction
    using SHAP and the trained XGBoost fraud detection model.
    Accepts row index or BENCH-XXXXXX ID.
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
        tx = get_transaction_by_id(idx_str)
        if tx and tx.get("benchmark_row_index") is not None:
            index = tx["benchmark_row_index"]

    if index is None:
        return f"Invalid benchmark row index or ID '{row_index}' for SHAP calculation."

    df = get_benchmark_df()

    if index < 0 or index >= len(df):
        return f"No transaction found at benchmark row index {index}."

    transaction = df.iloc[index]

    X = pd.DataFrame(
        [[transaction[column] for column in FEATURES]],
        columns=FEATURES
    )

    shap_values = explainer.shap_values(X)

    contributions = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP Value": shap_values[0]
    })

    contributions["Absolute SHAP"] = contributions["SHAP Value"].abs()
    contributions = contributions.sort_values("Absolute SHAP", ascending=False)
    top_features = contributions.head(5)

    explanation_lines = []
    for _, row in top_features.iterrows():
        feature = row["Feature"]
        value = row["SHAP Value"]
        direction = "increased the fraud prediction" if value > 0 else "decreased the fraud prediction"
        explanation_lines.append(f"{feature}: SHAP={value:.4f} ({direction})")

    return (
        f"Benchmark Row Index: {index} (ID: BENCH-{index+1:06d})\n"
        f"Actual Dataset Label: {int(transaction['Class'])}\n"
        f"Amount: ₹{float(transaction['Amount']):.2f}\n"
        f"Top SHAP Feature Contributors:\n"
        + "\n".join(explanation_lines)
        + "\n\n"
        "Note: V1-V28 are anonymized PCA model features from the public benchmark dataset. "
        "Their exact business meanings are anonymized."
    )