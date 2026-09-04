from risk_schema import RiskAssessment


def determine_action(risk_assessment: RiskAssessment) -> str:
    """
    Determine the required action based on the deterministic risk level.

    The LLM does not make this decision.
    """

    if risk_assessment.risk_level == "LOW":
        return "APPROVE"

    elif risk_assessment.risk_level == "MEDIUM":
        return "MONITOR"

    elif risk_assessment.risk_level == "HIGH":
        return "MANUAL_REVIEW"

    elif risk_assessment.risk_level == "CRITICAL":
        return "HOLD_AND_ESCALATE"

    else:
        return "UNKNOWN"


if __name__ == "__main__":

    assessment = RiskAssessment(
        risk_score=80,
        risk_level="HIGH",
        recommendation="Manual review.",
        reasons=[
            "Transaction amount is significantly above historical average.",
            "Transaction amount exceeds previous maximum."
        ]
    )

    action = determine_action(assessment)

    print("Risk Level:", assessment.risk_level)
    print("Risk Score:", assessment.risk_score)
    print("System Action:", action)