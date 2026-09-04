from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from risk_schema import RiskAssessment

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

structured_llm = llm.with_structured_output(RiskAssessment)

result = structured_llm.invoke(
    """
    Create a risk assessment for this transaction.

    Deterministic risk score: 80
    Risk level: HIGH

    Verified reasons:
    - Transaction amount is 21.2x the customer's historical average.
    - Transaction amount is more than 5x the customer's previous maximum.

    Recommended action:
    Manual review.
    """
)

print(result)
print()
print("Risk Score:", result.risk_score)
print("Risk Level:", result.risk_level)
print("Recommendation:", result.recommendation)
print("Reasons:", result.reasons)