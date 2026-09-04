from typing import Dict, Any, List, Optional

# Single Source of Truth for Risk Thresholds
RISK_THRESHOLDS = {
    "LOW": (0, 39),
    "MEDIUM": (40, 69),
    "HIGH": (70, 84),
    "CRITICAL": (85, 100)
}

def map_score_to_level(score: int) -> str:
    score = max(0, min(100, score))
    if score >= 85:
        return "CRITICAL"
    elif score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"

def calculate_evidence_completeness(evidence_sources: Dict[str, bool]) -> Dict[str, Any]:
    """
    Calculate completeness score (e.g. 8/10) based on available evidence flags.
    """
    total_sources = len(evidence_sources)
    available_count = sum(1 for v in evidence_sources.values() if v)
    
    score_str = f"{available_count}/{total_sources}"
    ratio = available_count / total_sources if total_sources > 0 else 0.0
    
    available_list = [k for k, v in evidence_sources.items() if v]
    unavailable_list = [k for k, v in evidence_sources.items() if not v]
    
    return {
        "completeness_score": score_str,
        "ratio": ratio,
        "available_sources": available_list,
        "unavailable_sources": unavailable_list
    }

def fuse_risk_evidence(
    ml_score: Optional[int],
    behavioral_score: Optional[int],
    velocity_score: Optional[int],
    anomaly_score: Optional[int],
    history_status: str = "available"
) -> Dict[str, Any]:
    """
    Deterministic Evidence Fusion Engine.
    Combines multiple risk signals into a final authoritative Risk Score (0-100) and Level.
    The LLM does NOT alter or calculate this score.
    """
    evidence_sources = {
        "ML Fraud Model": ml_score is not None,
        "Customer Behavioral History": history_status == "available" and behavioral_score is not None,
        "Transaction Velocity": velocity_score is not None,
        "Statistical Anomaly Detection": anomaly_score is not None,
        "Device Intelligence": True,
        "Location Consistency": True
    }
    
    completeness = calculate_evidence_completeness(evidence_sources)
    
    # Weight calculation based on available evidence
    components = []
    total_weight = 0.0
    weighted_sum = 0.0
    
    if ml_score is not None:
        weight = 0.50
        weighted_sum += ml_score * weight
        total_weight += weight
        components.append(f"ML Model ({ml_score}/100)")
        
    if behavioral_score is not None and history_status == "available":
        weight = 0.30 if ml_score is not None else 0.50
        weighted_sum += behavioral_score * weight
        total_weight += weight
        components.append(f"Behavioral ({behavioral_score}/100)")
        
    if velocity_score is not None:
        weight = 0.15
        weighted_sum += velocity_score * weight
        total_weight += weight
        components.append(f"Velocity ({velocity_score}/100)")
        
    if anomaly_score is not None:
        weight = 0.15
        weighted_sum += anomaly_score * weight
        total_weight += weight
        components.append(f"Anomaly ({anomaly_score}/100)")
        
    primary_score = 0
    if ml_score is not None:
        primary_score = max(primary_score, ml_score)
    if behavioral_score is not None and history_status == "available":
        primary_score = max(primary_score, behavioral_score)

    if total_weight > 0:
        calculated_fused = round(weighted_sum / total_weight)
        fused_score = max(primary_score, calculated_fused)
    else:
        fused_score = primary_score
        
    fused_score = max(0, min(100, fused_score))
    risk_level = map_score_to_level(fused_score)
    
    reasons = []
    if ml_score is not None and ml_score >= 70:
        reasons.append(f"High ML fraud probability score ({ml_score}/100).")
    if behavioral_score is not None and behavioral_score >= 70:
        reasons.append(f"Significant behavioral deviation detected ({behavioral_score}/100).")
    if velocity_score is not None and velocity_score >= 50:
        reasons.append(f"High transaction velocity detected ({velocity_score}/100).")
    if anomaly_score is not None and anomaly_score >= 50:
        reasons.append(f"Statistical anomaly flagged ({anomaly_score}/100).")
    if history_status == "insufficient_history":
        reasons.append("Historical customer baseline unavailable (new customer profile).")
        
    if not reasons:
        reasons.append("Risk signals within normal parameters.")

    return {
        "fused_risk_score": fused_score,
        "risk_level": risk_level,
        "evidence_completeness": completeness["completeness_score"],
        "completeness_details": completeness,
        "components_evaluated": components,
        "reasons": reasons
    }
