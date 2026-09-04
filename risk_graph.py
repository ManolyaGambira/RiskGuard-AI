from typing import TypedDict, Optional, List
from risk_schema import RiskAssessment
from policy_engine import determine_action
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain.agents import create_agent
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END
# pyrefly: ignore [missing-import]
from langgraph.types import interrupt, Command
# pyrefly: ignore [missing-import]
from langgraph.checkpoint.memory import MemorySaver

from audit_storage import (
    initialize_audit_db,
    save_audit_event
)

from transaction_tools import (
    get_transaction,
    get_customer_history,
    get_device_history,
    calculate_velocity_risk,
    get_customer_profile,
    get_device_profile
)

from ml_risk_calculator import calculate_behavioral_risk
from real_ml_risk_calculator import calculate_real_ml_risk
from shap_tools import get_shap_explanation
from risk_fusion import fuse_risk_evidence, map_score_to_level
from transaction_data import get_transaction_by_id

class RiskGraphState(TypedDict, total=False):
    transaction_id: str
    investigation: str
    risk_score: int
    risk_level: str
    system_action: str
    human_decision: Optional[str]
    audit_log: List[str]
    consistency_warning: Optional[str]

def add_audit_event(state: RiskGraphState, event: str) -> List[str]:
    """Append an event to the transaction audit trail cleanly."""
    current = state.get("audit_log", [])
    if current and current[-1] == event:
        return current
    return [*current, event]

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

SYSTEM_PROMPT = """
You are RiskGuard's Autonomous Risk Investigation Agent.

Your job is to investigate financial transactions using verified evidence tools and provide a concise, evidence-based investigation summary.

IMPORTANT RULES:

1. Always use get_transaction to obtain verified transaction facts before investigating.
2. Use get_customer_history or get_customer_profile when customer historical spending patterns are relevant.
3. Use get_device_history or get_device_profile when device behavior is relevant.
4. Use calculate_velocity_risk when evaluating short-term burst activity or velocity.
5. Use calculate_behavioral_risk when investigating behavioral transaction risk.
6. Use calculate_real_ml_risk when investigating a benchmark transaction (BENCH-XXXXXX) or when a benchmark row index is available.
7. Use get_shap_explanation when evaluating feature contributions for a benchmark transaction.
8. NEVER invent transaction details, customer history, device history, locations, payment information, risk scores, probabilities, or SHAP values.
9. Tool outputs are the authoritative source for numerical risk values and verified transaction evidence.
10. Never override deterministic risk scores or risk levels returned by risk tools.
11. V1-V28 are anonymized model features from the public benchmark dataset. Never invent business meanings for them.
12. If evidence is missing (insufficient_history or unavailable), explicitly state that history is insufficient or unavailable. Do not assume fraud solely due to missing history.
13. Clearly separate:
    - Verified transaction facts
    - Behavioral & Velocity evidence
    - ML & SHAP evidence
    - Agent interpretation & concise recommendation
14. The LLM is an investigator and explainer. The deterministic system owns the score, level, and policy action.
"""

agent = create_agent(
    model=llm,
    tools=[
        get_transaction,
        get_customer_history,
        get_device_history,
        calculate_velocity_risk,
        get_customer_profile,
        get_device_profile,
        calculate_behavioral_risk,
        calculate_real_ml_risk,
        get_shap_explanation
    ],
    system_prompt=SYSTEM_PROMPT
)

def investigation_agent_node(state: RiskGraphState) -> RiskGraphState:
    transaction_id = state["transaction_id"]

    audit_log = add_audit_event(state, f"Investigation started for transaction {transaction_id}")
    save_audit_event(transaction_id, f"Investigation started for transaction {transaction_id}")

    try:
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    f"Perform a complete risk investigation of transaction {transaction_id}. "
                    f"Gather transaction facts, check customer profile/history, assess behavioral risk, "
                    f"check velocity risk, and if it is a benchmark transaction (BENCH-XXXXXX), "
                    f"assess ML risk and SHAP explainability. "
                    f"End with a concise textual investigation summary."
                )
            }]
        })

        investigation = ""
        for message in reversed(result["messages"]):
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                investigation = content.strip()
                break
            elif isinstance(content, list):
                parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                if parts:
                    investigation = "\n".join(parts).strip()
                    break

        if not investigation:
            investigation = "Investigation completed. No anomaly detected."
    except Exception:
        investigation = (
            "AI investigation temporarily unavailable. Deterministic risk analysis remains active.\n\n"
            "System-Generated Deterministic Evidence Summary:\n"
            f"- Transaction ID: {transaction_id}\n"
            "- Authoritative deterministic calculations and policy rules remain fully operational below.\n"
            "- Risk score, risk level, and policy decisions are governed by deterministic rules and are unaffected by LLM availability."
        )

    return {
        **state,
        "investigation": investigation,
        "audit_log": audit_log
    }

def risk_assessment_node(state: RiskGraphState) -> RiskGraphState:
    transaction_id = state["transaction_id"]
    tx = get_transaction_by_id(transaction_id)

    beh_output = calculate_behavioral_risk.invoke(transaction_id)
    
    beh_score = 0
    beh_level = "LOW"
    for line in beh_output.splitlines():
        if line.startswith("Behavioral Risk Score:"):
            beh_score = int(line.split(":")[1].split("/")[0].strip())
        elif line.startswith("Behavioral Risk Level:"):
            beh_level = line.split(":")[1].strip()

    ml_score = None
    if tx and (tx.get("source") == "benchmark" or str(transaction_id).upper().startswith("BENCH-")):
        ml_output = calculate_real_ml_risk.invoke(transaction_id)
        for line in ml_output.splitlines():
            if line.startswith("ML Risk Score:"):
                ml_score = int(line.split(":")[1].split("/")[0].strip())

    history_status = "insufficient_history" if "insufficient_history" in beh_output else "available"
    fusion = fuse_risk_evidence(
        ml_score=ml_score,
        behavioral_score=beh_score,
        velocity_score=0,
        anomaly_score=0,
        history_status=history_status
    )

    final_score = fusion["fused_risk_score"]
    final_level = fusion["risk_level"]

    audit_log = add_audit_event(
        state,
        f"Risk calculated: {final_score}/100 ({final_level})"
    )
    save_audit_event(
        transaction_id,
        f"Risk calculated: {final_score}/100 ({final_level})"
    )

    return {
        **state,
        "risk_score": final_score,
        "risk_level": final_level,
        "audit_log": audit_log
    }

def policy_decision_node(state: RiskGraphState) -> RiskGraphState:
    risk_assessment = RiskAssessment(
        risk_score=state["risk_score"],
        risk_level=state["risk_level"],
        recommendation="",
        reasons=[]
    )

    action = determine_action(risk_assessment)

    audit_log = add_audit_event(state, f"Policy decision: {action}")
    save_audit_event(state["transaction_id"], f"Policy decision: {action}")

    if action in ["MANUAL_REVIEW", "HOLD_AND_ESCALATE"]:
        save_audit_event(
            state["transaction_id"],
            f"Human review requested: {state['risk_level']} risk"
        )

    return {
        **state,
        "system_action": action,
        "audit_log": audit_log
    }

def route_after_policy(state: RiskGraphState) -> str:
    if state["risk_level"] in ["HIGH", "CRITICAL"]:
        return "human_review"
    return "end"

def human_review_node(state: RiskGraphState) -> RiskGraphState:
    audit_log = add_audit_event(
        state,
        f"Human review requested: {state['risk_level']} risk"
    )

    decision = interrupt({
        "message": "Human review required.",
        "transaction_id": state["transaction_id"],
        "risk_score": state["risk_score"],
        "risk_level": state["risk_level"],
        "system_action": state["system_action"],
        "options": ["APPROVE", "REJECT"]
    })

    audit_log = add_audit_event(
        {**state, "audit_log": audit_log},
        f"Human decision recorded: {decision}"
    )
    save_audit_event(state["transaction_id"], f"Human decision recorded: {decision}")

    return {
        **state,
        "human_decision": decision,
        "audit_log": audit_log
    }

# Build LangGraph workflow
builder = StateGraph(RiskGraphState)
builder.add_node("investigation_agent", investigation_agent_node)
builder.add_node("risk_assessment", risk_assessment_node)
builder.add_node("policy_decision", policy_decision_node)
builder.add_node("human_review", human_review_node)

builder.add_edge(START, "investigation_agent")
builder.add_edge("investigation_agent", "risk_assessment")
builder.add_edge("risk_assessment", "policy_decision")

builder.add_conditional_edges(
    "policy_decision",
    route_after_policy,
    {
        "human_review": "human_review",
        "end": END
    }
)

builder.add_edge("human_review", END)

initialize_audit_db()
memory = MemorySaver()
risk_graph = builder.compile(checkpointer=memory)

print("RiskGraph compiled successfully!")