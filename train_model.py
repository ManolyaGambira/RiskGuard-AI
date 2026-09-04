import pandas as pd
import json
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. Load real fraud dataset
# ==========================================

DATA_FILE = "data/real/creditcard.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. Separate features and target
# ==========================================

# Everything except Class is used as a feature.
features = [
    column
    for column in df.columns
    if column != "Class"
]

X = df[features]
y = df["Class"]


# ==========================================
# 3. Train/test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining fraud cases:", y_train.sum())
print("Testing fraud cases:", y_test.sum())


# ==========================================
# 4. Handle severe class imbalance
# ==========================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nScale Pos Weight:", scale_pos_weight)


# ==========================================
# 5. Create XGBoost model
# ==========================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    # Important for highly imbalanced fraud data
    scale_pos_weight=scale_pos_weight,

    random_state=42,
    eval_metric="aucpr"
)


# ==========================================
# 6. Train model
# ==========================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)


# ==========================================
# 7. Predictions
# ==========================================

predictions = model.predict(X_test)

fraud_probabilities = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 8. Save model
# ==========================================

model.save_model(
    "risk_model.json"
)

print("\nModel saved successfully!")


# ==========================================
# 9. Evaluation metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    fraud_probabilities
)

pr_auc = average_precision_score(
    y_test,
    fraud_probabilities
)


# ==========================================
# 10. Display performance
# ==========================================

print("\n==========================================")
print("        MODEL PERFORMANCE")
print("==========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


# ==========================================
# 11. Confusion matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions
)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "true_negatives": int(cm[0][0]),
    "false_positives": int(cm[0][1]),
    "false_negatives": int(cm[1][0]),
    "true_positives": int(cm[1][1])
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel metrics saved successfully!")

print("\n==========================================")
print("        CONFUSION MATRIX")
print("==========================================")

print(cm)


# ==========================================
# 12. Classification report
# ==========================================

print("\n==========================================")
print("        CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 13. Sample risk scores
# ==========================================

print("\n==========================================")
print("        SAMPLE RISK SCORES")
print("==========================================")

for probability in fraud_probabilities[:10]:

    risk_score = round(
        probability * 100
    )

    if risk_score >= 85:
        risk_level = "CRITICAL"

    elif risk_score >= 70:
        risk_level = "HIGH"

    elif risk_score >= 40:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    print(
        f"Fraud Probability: {probability:.4f} | "
        f"Risk Score: {risk_score}/100 | "
        f"Risk Level: {risk_level}"
    )