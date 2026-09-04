import pandas as pd
# pyrefly: ignore [missing-import]
import shap
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier


MODEL_FILE = "risk_model.json"
DATA_FILE = "data/real/creditcard.csv"


# Load model
model = XGBClassifier()
model.load_model(MODEL_FILE)


# Load dataset
df = pd.read_csv(DATA_FILE)


FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


# Select fraud transaction
row_index = 541

transaction = df.iloc[row_index]

X = pd.DataFrame(
    [[transaction[column] for column in FEATURES]],
    columns=FEATURES
)


# Create SHAP explainer
explainer = shap.TreeExplainer(model)


# Calculate SHAP values
shap_values = explainer.shap_values(X)


print("\n==========================================")
print("             SHAP EXPLANATION")
print("==========================================")

print("Dataset Row:", row_index)
print("Actual Class:", int(transaction["Class"]))
print("Amount:", transaction["Amount"])

print("\nTop contributing features:")

contributions = pd.DataFrame({
    "Feature": FEATURES,
    "SHAP Value": shap_values[0]
})

contributions["Absolute SHAP"] = (
    contributions["SHAP Value"].abs()
)

contributions = contributions.sort_values(
    "Absolute SHAP",
    ascending=False
)

print(
    contributions[
        ["Feature", "SHAP Value"]
    ].head(10).to_string(index=False)
)