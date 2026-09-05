import os
from typing import TypedDict, Optional, List
from risk_schema import RiskAssessment
from policy_engine import determine_action
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
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
from transaction_data import get_transaction_by_id, get_customer_transactions
from velocity_engine import calculate_velocity_risk as calc_velocity
from anomaly_engine import calculate_anomaly_score as calc_anomaly

class RiskGraphState(TypedDict, total=False):
    transaction_id: str
    investigation: str
    risk_score: int
    risk_level: str
    system_action: str
    human_decision: Optional[str]
    final_disposition: Optional[str]
    audit_log: List[str]
    consistency_warning: Optional[str]

def add_audit_event(state: RiskGraphState, event: str) -> List[str]:
    """Append an event to the transaction audit trail cleanly."""
    current = state.get("audit_log", [])
    if current and current[-1] == event:
        return current
    return [*current, event]

def get_google_api_key() -> Optional[str]:
    """Retrieve Gemini API key from environment variables or Streamlit secrets safely."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            elif "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass
    return None

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

class DummyAgent:
    def invoke(self, *args, **kwargs):
        raise ValueError("GEMINI_API_KEY is not configured in environment or Streamlit secrets.")

def create_gemini_agent():
    api_key = get_google_api_key()
    if not api_key:
        return None, DummyAgent()
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key,
            max_retries=2
        )
        ag = create_agent(
            model=model,
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
        return model, ag
    except Exception as e:
        print(f"[RiskGuard Agent Warning] Could not initialize ChatGoogleGenerativeAI: [{type(e).__name__}] {e}")
        return None, DummyAgent()

llm, agent = create_gemini_agent()

def build_deterministic_investigation_report(transaction_id: str) -> str:
    tx_data = get_transaction_by_id(transaction_id)
    if not tx_data:
        return f"Transaction ID '{transaction_id}' not found in RiskGuard record index."

    cust_id = tx_data.get("customer_id", "UNKNOWN")
    amount = float(tx_data.get("amount", 0.0))
    location = tx_data.get("location", "UNKNOWN")
    device = tx_data.get("device", "UNKNOWN")
    payment_method = tx_data.get("payment_method", "UNKNOWN")
    source = tx_data.get("source", "demo")

    cust_history = get_customer_transactions(cust_id)
    past_history = [h for h in cust_history if h.get("transaction_id") != transaction_id]

    beh_output = calculate_behavioral_risk.invoke(transaction_id)
    beh_score = 0
    beh_level = "LOW"
    for line in beh_output.splitlines():
        if line.startswith("Behavioral Risk Score:"):
            beh_score = int(line.split(":")[1].split("/")[0].strip())
        elif line.startswith("Behavioral Risk Level:"):
            beh_level = line.split(":")[1].strip()

    vel_res = calc_velocity(tx_data, past_history)
    anom_res = calc_anomaly(tx_data)

    ml_score = None
    ml_level = None
    shap_text = None
    if source == "benchmark" or str(transaction_id).upper().startswith("BENCH-"):
        ml_output = calculate_real_ml_risk.invoke(transaction_id)
        for line in ml_output.splitlines():
            if line.startswith("ML Risk Score:"):
                ml_score = int(line.split(":")[1].split("/")[0].strip())
            elif line.startswith("ML Risk Level:"):
                ml_level = line.split(":")[1].strip()

        shap_text = get_shap_explanation.invoke(transaction_id)

    history_status = "insufficient_history" if "insufficient_history" in beh_output else "available"
    fusion = fuse_risk_evidence(
        ml_score=ml_score,
        behavioral_score=beh_score,
        velocity_score=vel_res.get("velocity_score", 0),
        anomaly_score=anom_res.get("anomaly_score", 0),
        history_status=history_status
    )
    score = fusion["fused_risk_score"]
    level = fusion["risk_level"]
    action = determine_action(RiskAssessment(risk_score=score, risk_level=level, recommendation="", reasons=[]))

    report_lines = [
        "### INVESTIGATION SUMMARY",
        f"Verified RiskGuard deterministic risk analysis performed for transaction **{transaction_id}**. "
        f"Transaction amount **₹{amount:,.2f}** initiated by customer **{cust_id}** via **{device}** from **{location}**.",
        "",
        "### VERIFIED SIGNALS",
        f"• **Transaction ID**: {transaction_id}",
        f"• **Customer ID**: {cust_id} ({len(past_history)} past transactions in record)",
        f"• **Amount**: ₹{amount:,.2f} | **Device**: {device} | **Location**: {location} | **Payment Method**: {payment_method}",
        f"• **Behavioral Risk Score**: {beh_score}/100 ({beh_level})",
        f"• **Velocity Score**: {vel_res.get('velocity_score', 0)}/100 | **Statistical Anomaly Score**: {anom_res.get('anomaly_score', 0)}/100",
        ""
    ]

    if ml_score is not None:
        first_shap_line = shap_text.splitlines()[0] if shap_text else 'Anonymized model features V1-V28 evaluated'
        report_lines.extend([
            "### ML EVIDENCE",
            f"• **XGBoost Fraud Risk Score**: {ml_score}/100 ({ml_level})",
            f"• **SHAP Feature Attribution**: {first_shap_line}",
            ""
        ])
    else:
        report_lines.extend([
            "### ML EVIDENCE",
            "• **Demo Transaction**: Rule-based behavioral & velocity risk engines active",
            ""
        ])

    report_lines.extend([
        "### RISK ASSESSMENT",
        f"**{level} — {score}/100**",
        "",
        "### RECOMMENDED ACTION",
        f"**{action}**"
    ])

    return "\n".join(report_lines)

def investigation_agent_node(state: RiskGraphState) -> RiskGraphState:
    global llm, agent
    transaction_id = state["transaction_id"]

    audit_log = add_audit_event(state, f"Investigation started for transaction {transaction_id}")
    save_audit_event(transaction_id, f"Investigation started for transaction {transaction_id}")

    is_llm_fallback = False
    investigation = ""

    try:
        current_agent = agent
        if current_agent is None or isinstance(current_agent, DummyAgent):
            _, real_agent = create_gemini_agent()
            if real_agent is not None and not isinstance(real_agent, DummyAgent):
                agent = real_agent
                current_agent = real_agent

        result = current_agent.invoke({
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

        for message in reversed(result.get("messages", [])):
            content = getattr(message, "content", None)
            # Skip messages that only contain tool calls without final response text
            if hasattr(message, "tool_calls") and message.tool_calls and not content:
                continue

            if isinstance(content, str) and content.strip():
                investigation = content.strip()
                break
            elif isinstance(content, list):
                parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                text_content = "\n".join([p for p in parts if p.strip()]).strip()
                if text_content:
                    investigation = text_content
                    break

        if not investigation or "temporarily unavailable" in investigation.lower():
            print(f"[RiskGuard Agent] Gemini response unavailable or empty for {transaction_id}. Using deterministic report.")
            investigation = build_deterministic_investigation_report(transaction_id)
            is_llm_fallback = True

    except Exception as e:
        print(f"[RiskGuard Agent Error] Gemini investigation failed for {transaction_id}: [{type(e).__name__}] {e}")
        investigation = build_deterministic_investigation_report(transaction_id)
        is_llm_fallback = True

    return {
        **state,
        "investigation": investigation,
        "is_llm_fallback": is_llm_fallback,
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

    human_dec_str = f"APPROVED BY HUMAN" if decision == "APPROVE" else f"REJECTED BY HUMAN"
    final_disp_str = f"APPROVED AFTER HUMAN REVIEW" if decision == "APPROVE" else f"REJECTED AFTER HUMAN REVIEW"

    audit_log = add_audit_event(
        {**state, "audit_log": audit_log},
        f"Human decision recorded: {human_dec_str}"
    )
    save_audit_event(state["transaction_id"], f"Human decision recorded: {human_dec_str}")

    audit_log = add_audit_event(
        {**state, "audit_log": audit_log},
        f"Final disposition: {final_disp_str}"
    )
    save_audit_event(state["transaction_id"], f"Final disposition: {final_disp_str}")

    return {
        **state,
        "human_decision": decision,
        "final_disposition": final_disp_str,
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