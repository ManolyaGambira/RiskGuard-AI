from pydantic import BaseModel, Field
from typing import List


class RiskAssessment(BaseModel):
    risk_score: int = Field(
        description="The deterministic risk score from 0 to 100."
    )

    risk_level: str = Field(
        description="Risk level: LOW, MEDIUM, HIGH, or CRITICAL."
    )

    recommendation: str = Field(
        description="Recommended action based on the risk level."
    )

    reasons: List[str] = Field(
        description="Evidence-based reasons supporting the risk assessment."
    )