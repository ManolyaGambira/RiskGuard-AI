import time
import pandas as pd
from typing import Dict, Any, List
from transaction_data import get_benchmark_df, normalize_benchmark_row
from real_ml_risk_calculator import model, FEATURES
from risk_fusion import map_score_to_level
from sklearn.metrics import precision_score, recall_score, f1_score

def run_batch_risk_screening(batch_size: int = 1000, start_index: int = 0) -> Dict[str, Any]:
    """
    Stage 1 Fast ML Screening & Risk Prioritization.
    Evaluates up to 10,000 benchmark transactions efficiently using vector inference.
    Returns comprehensive batch analytics, execution timing, and performance metrics.
    """
    start_time = time.time()
    df = get_benchmark_df()
    
    end_index = min(start_index + batch_size, len(df))
    batch_df = df.iloc[start_index:end_index].copy()
    
    if batch_df.empty:
        return {"error": "Empty batch dataset"}
        
    X_batch = batch_df[FEATURES]
    y_true = batch_df["Class"].astype(int).values
    
    # Vectorized XGBoost prediction
    probabilities = model.predict_proba(X_batch)[:, 1]
    scores = (probabilities * 100).round().astype(int)
    
    batch_df["fraud_probability"] = probabilities
    batch_df["risk_score"] = scores
    batch_df["risk_level"] = [map_score_to_level(s) for s in scores]
    batch_df["predicted_fraud"] = (scores >= 70).astype(int)  # HIGH / CRITICAL classified as predicted fraud
    
    elapsed_sec = time.time() - start_time
    throughput = len(batch_df) / elapsed_sec if elapsed_sec > 0 else 0.0
    
    # Count risk distributions
    low_count = int((batch_df["risk_level"] == "LOW").sum())
    medium_count = int((batch_df["risk_level"] == "MEDIUM").sum())
    high_count = int((batch_df["risk_level"] == "HIGH").sum())
    critical_count = int((batch_df["risk_level"] == "CRITICAL").sum())
    
    suspicious_count = high_count + critical_count
    
    # Calculate classification metrics
    tp = int(((batch_df["predicted_fraud"] == 1) & (y_true == 1)).sum())
    fp = int(((batch_df["predicted_fraud"] == 1) & (y_true == 0)).sum())
    tn = int(((batch_df["predicted_fraud"] == 0) & (y_true == 0)).sum())
    fn = int(((batch_df["predicted_fraud"] == 0) & (y_true == 1)).sum())
    
    prec = float(precision_score(y_true, batch_df["predicted_fraud"], zero_division=0))
    rec = float(recall_score(y_true, batch_df["predicted_fraud"], zero_division=0))
    f1 = float(f1_score(y_true, batch_df["predicted_fraud"], zero_division=0))
    
    # Top suspicious transactions formatted for risk queue
    top_suspicious_df = batch_df[batch_df["risk_score"] >= 40].sort_values("risk_score", ascending=False).head(50)
    top_suspicious_list = []
    for idx, row in top_suspicious_df.iterrows():
        tx_dict = normalize_benchmark_row(int(idx), row)
        tx_dict["ml_risk_score"] = int(row["risk_score"])
        tx_dict["risk_level"] = row["risk_level"]
        tx_dict["fraud_probability"] = float(row["fraud_probability"])
        top_suspicious_list.append(tx_dict)

    return {
        "total_transactions": len(batch_df),
        "start_index": start_index,
        "end_index": end_index,
        "elapsed_seconds": round(elapsed_sec, 4),
        "throughput_tx_per_sec": round(throughput, 2),
        "risk_distribution": {
            "LOW": low_count,
            "MEDIUM": medium_count,
            "HIGH": high_count,
            "CRITICAL": critical_count
        },
        "suspicious_count": suspicious_count,
        "actual_fraud_count": int(y_true.sum()),
        "average_risk_score": round(float(scores.mean()), 2),
        "maximum_risk_score": int(scores.max()),
        "metrics": {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        },
        "top_suspicious_transactions": top_suspicious_list
    }
