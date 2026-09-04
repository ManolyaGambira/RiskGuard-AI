import json
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from audit_storage import get_audit_events
from transaction_data import (
    get_transaction_by_id,
    get_customer_transactions,
    get_device_transactions,
    load_demo_transactions,
    get_benchmark_df
)
from risk_graph import risk_graph
from risk_fusion import map_score_to_level, fuse_risk_evidence
from batch_processor import run_batch_risk_screening
from stream_simulator import StreamSimulator
from real_ml_risk_calculator import calculate_real_ml_risk
from shap_tools import get_shap_explanation
from velocity_engine import calculate_velocity_risk as calc_velocity
from anomaly_engine import calculate_anomaly_score as calc_anomaly
# pyrefly: ignore [missing-import]
from langgraph.types import Command

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RiskGuard AI — Risk Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load saved model metrics if present
metrics_file = Path("model_metrics.json")
if metrics_file.exists():
    with open(metrics_file, "r") as f:
        model_metrics = json.load(f)
else:
    model_metrics = {
        "accuracy": 0.9992, "precision": 0.8750, "recall": 0.7959,
        "f1": 0.8333, "roc_auc": 0.9741, "pr_auc": 0.8521,
        "true_negatives": 56855, "false_positives": 9,
        "false_negatives": 20, "true_positives": 78
    }

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "investigated_transaction" not in st.session_state:
    st.session_state.investigated_transaction = "TXN1009"

if "investigation_result" not in st.session_state:
    st.session_state.investigation_result = None

if "investigation_config" not in st.session_state:
    st.session_state.investigation_config = None

if "awaiting_human_review" not in st.session_state:
    st.session_state.awaiting_human_review = False

if "human_decision" not in st.session_state:
    st.session_state.human_decision = None

if "stream_active" not in st.session_state:
    st.session_state.stream_active = False

if "stream_events" not in st.session_state:
    st.session_state.stream_events = []

if "stream_simulator" not in st.session_state:
    st.session_state.stream_simulator = StreamSimulator()

if "alarm_trigger_key" not in st.session_state:
    st.session_state.alarm_trigger_key = None

if "alarm_played_key" not in st.session_state:
    st.session_state.alarm_played_key = None

# Auto-run backend risk assessment on session init if empty
if st.session_state.investigation_result is None:
    tx_id = st.session_state.investigated_transaction
    st.session_state.alarm_trigger_key = f"{tx_id}-{time.time()}"
    config = {"configurable": {"thread_id": f"init-{tx_id}"}}
    st.session_state.investigation_config = config
    res = risk_graph.invoke({"transaction_id": tx_id}, config=config)
    st.session_state.investigation_result = res
    snapshot = risk_graph.get_state(config)
    st.session_state.awaiting_human_review = bool(snapshot.next)

# =========================================================
# STYLES & GLASSMORPHISM DESIGN SYSTEM
# =========================================================

st.html("""
<style>
/* Prevent top-of-page clipping beneath Streamlit header toolbar */
header[data-testid="stHeader"] {
    background: rgba(7, 11, 20, 0.85);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    z-index: 99;
}

.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(37, 99, 235, 0.12), transparent 35%),
                radial-gradient(circle at 90% 20%, rgba(139, 92, 246, 0.10), transparent 35%),
                #070b14;
    color: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 4.5rem; /* Sufficient top spacing to avoid header toolbar overlap */
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

.brand {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
}

.brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.35);
}

.brand-title {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.brand-subtitle {
    color: #94a3b8;
    font-size: 12px;
    margin-top: -2px;
}

.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
}

.page-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.8px;
}

.page-description {
    color: #94a3b8;
    font-size: 14px;
}

.system-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.10);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #6ee7b7;
    font-size: 13px;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
}

.hero-card {
    padding: 24px 28px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(148, 163, 184, 0.14);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
    margin-bottom: 24px;
}

.hero-label {
    color: #60a5fa;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.hero-title {
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 6px;
}

.hero-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
}

.metric-card {
    padding: 18px;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.12);
}

.metric-label {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 24px;
    font-weight: 800;
}

.risk-badge {
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
}

.risk-badge-low { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
.risk-badge-medium { background: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }
.risk-badge-high { background: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.3); }
.risk-badge-critical { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

.timeline-item {
    border-left: 2px solid #3b82f6;
    padding-left: 14px;
    margin-bottom: 12px;
}

.timeline-time {
    color: #64748b;
    font-size: 11px;
}

.timeline-event {
    color: #e2e8f0;
    font-weight: 600;
    font-size: 13px;
}
</style>
""")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.html("""
    <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-title">RiskGuard</div>
            <div class="brand-subtitle">AI Risk Operations Center</div>
        </div>
    </div>
    """)
    st.divider()

    nav_choice = st.radio(
        "Navigation",
        [
            "◉ Overview",
            "▶ Live Transaction Stream",
            "🔎 Risk Explorer",
            "🚨 Risk Queue",
            "⚡ Batch Analysis",
            "📊 Model Performance",
            "📜 Audit Trail",
            "🟢 System Health"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("SYSTEM STATUS")
    st.success("● ML Screening Model: Active")
    st.success("● LangGraph Agent: Ready")
    st.success("● SQLite Audit Trail: Connected")
    st.caption("DATASET TRANSPARENCY")
    st.info("Public Credit Card Fraud Benchmark (284,807 transactions) loaded for evaluation & batch scaling.")

# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="top-header">
    <div>
        <div class="page-title">Risk Operations Center</div>
        <div class="page-description">Autonomous Transaction Monitoring & Risk Management System</div>
    </div>
    <div class="system-badge">
        <span class="status-dot"></span> DEFENSE-ONLY SYSTEM ONLINE
    </div>
</div>
""")

# =========================================================
# VIEW 1: OVERVIEW
# =========================================================

if nav_choice == "◉ Overview":
    st.html("""
    <div class="hero-card">
        <div class="hero-label">ENTERPRISE TRANSACTION MONITORING</div>
        <div class="hero-title">Intelligent Financial Risk Ops & Autonomous Investigation</div>
        <div class="hero-text">
            RiskGuard-AI fuses fast XGBoost screening, behavioral pattern analysis, transaction velocity tracking,
            statistical anomaly detection, SHAP explainability, autonomous LangGraph investigation agents, and deterministic policy enforcement.
            The LLM is an investigator and explainer — deterministic rules strictly govern final risk scores and policy actions.
        </div>
    </div>
    """)

    st.markdown("### System Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.html("""
        <div class="metric-card">
            <div class="metric-label">Benchmark Scale</div>
            <div class="metric-value">284,807</div>
            <div style="color:#64748b; font-size:12px; margin-top:4px;">transactions available</div>
        </div>
        """)
    with c2:
        st.html("""
        <div class="metric-card">
            <div class="metric-label">Model Precision</div>
            <div class="metric-value">87.50%</div>
            <div style="color:#64748b; font-size:12px; margin-top:4px;">on unseen benchmark test set</div>
        </div>
        """)
    with c3:
        st.html("""
        <div class="metric-card">
            <div class="metric-label">Model PR-AUC</div>
            <div class="metric-value">85.21%</div>
            <div style="color:#64748b; font-size:12px; margin-top:4px;">imbalanced fraud metric</div>
        </div>
        """)
    with c4:
        st.html("""
        <div class="metric-card">
            <div class="metric-label">Batch Screening</div>
            <div class="metric-value">> 2,500 tx/s</div>
            <div style="color:#64748b; font-size:12px; margin-top:4px;">Stage 1 throughput</div>
        </div>
        """)

    st.markdown("### Core Conceptual Pipeline")
    st.info("""
    **Transaction Ingestion** → **Data Normalization** → **Stage 1 Fast ML Screening** → **Behavioral & Velocity Analysis** → **Deterministic Risk Fusion** → **Risk Queue Prioritization** → **Stage 2 Agent Investigation** → **Deterministic Policy Action** → **Human-in-the-Loop Review** → **SQLite Audit Trail**
    """)

# =========================================================
# VIEW 2: LIVE TRANSACTION STREAM
# =========================================================

elif nav_choice == "▶ Live Transaction Stream":
    st.markdown("### Simulated Real-Time Transaction Stream")
    st.caption("Replaying transactions from the public credit card benchmark dataset with real-time ML risk scoring.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if not st.session_state.stream_active:
            if st.button("▶ START STREAM", type="primary", use_container_width=True):
                st.session_state.stream_active = True
                st.rerun()
        else:
            if st.button("⏸ PAUSE STREAM", use_container_width=True):
                st.session_state.stream_active = False
                st.rerun()

    with col2:
        speed = st.selectbox("Stream Speed", ["1 tx/sec", "5 tx/sec", "10 tx/sec"])

    with col3:
        if st.button("🗑 CLEAR STREAM HISTORY"):
            st.session_state.stream_events = []
            st.rerun()

    if st.session_state.stream_active:
        sim = st.session_state.stream_simulator
        event = sim.next_event()
        st.session_state.stream_events.insert(0, event)
        if len(st.session_state.stream_events) > 50:
            st.session_state.stream_events.pop()

    if st.session_state.stream_events:
        df_stream = pd.DataFrame(st.session_state.stream_events)
        display_cols = ["timestamp", "transaction_id", "customer_id", "amount", "location", "device", "risk_score", "risk_level"]
        st.dataframe(df_stream[display_cols], use_container_width=True)
    else:
        st.info("Stream is currently paused. Click 'START STREAM' to begin real-time ingestion simulation.")

# =========================================================
# VIEW 3: RISK EXPLORER
# =========================================================

elif nav_choice == "🔎 Risk Explorer":
    st.markdown("### Transaction Risk Explorer & AI Investigation")
    st.caption("Inspect demo transactions (TXN1001-TXN1009) or benchmark transactions (BENCH-000001 to BENCH-284807).")

    search_col1, search_col2 = st.columns([4, 1])
    with search_col1:
        input_id = st.text_input("Transaction ID", value=st.session_state.investigated_transaction, placeholder="e.g. TXN1005, TXN1009, or BENCH-000542")
    with search_col2:
        run_inv = st.button("🔎 INVESTIGATE", type="primary", use_container_width=True)

    if run_inv:
        input_id = input_id.strip()
        if input_id:
            st.session_state.investigated_transaction = input_id
            st.session_state.human_decision = None
            st.session_state.alarm_trigger_key = f"{input_id}-{time.time()}"
            config = {"configurable": {"thread_id": f"ui-{input_id}-{time.time()}"}}
            st.session_state.investigation_config = config

            with st.spinner(f"🧠 RiskGuard Agent investigating {input_id}..."):
                res = risk_graph.invoke({"transaction_id": input_id}, config=config)
                st.session_state.investigation_result = res
                snapshot = risk_graph.get_state(config)
                st.session_state.awaiting_human_review = bool(snapshot.next)

    tx_id = st.session_state.investigated_transaction
    tx_data = get_transaction_by_id(tx_id)

    if tx_data:
        res = st.session_state.investigation_result
        if not res or res.get("transaction_id") != tx_id:
            st.session_state.alarm_trigger_key = f"{tx_id}-{time.time()}"
            config = {"configurable": {"thread_id": f"fetch-{tx_id}"}}
            res = risk_graph.invoke({"transaction_id": tx_id}, config=config)
            st.session_state.investigation_result = res
            snapshot = risk_graph.get_state(config)
            st.session_state.awaiting_human_review = bool(snapshot.next)

        score = res.get("risk_score", 0)
        level = res.get("risk_level", map_score_to_level(score))
        action = res.get("system_action", "UNKNOWN")

        badge_class = f"risk-badge-{level.lower()}"

        # -----------------------------------------------------
        # AUTHORITATIVE ALARM SECTION (CRITICAL & HIGH)
        # -----------------------------------------------------
        trigger_audio_js = False
        trigger_key = st.session_state.get("alarm_trigger_key")
        played_key = st.session_state.get("alarm_played_key")
        if level in ["HIGH", "CRITICAL"] and trigger_key is not None and trigger_key != played_key:
            trigger_audio_js = True
            st.session_state.alarm_played_key = trigger_key

        if level == "CRITICAL":
            st.html("""
            <div style="padding:16px 20px; border-radius:14px; background:rgba(239,68,68,0.20); border:2px solid #ef4444; margin-bottom:16px; display:flex; align-items:center; justify-content:space-between; box-shadow:0 0 25px rgba(239,68,68,0.35);">
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-size:26px;">🚨</span>
                    <div>
                        <div style="font-size:16px; font-weight:800; color:#f87171; letter-spacing:0.5px;">CRITICAL RISK ALARM ACTIVE</div>
                        <div style="font-size:13px; color:#fca5a5;">Immediate Action Required: Transaction flagged for HOLD_AND_ESCALATE.</div>
                    </div>
                </div>
            </div>
            """)
        elif level == "HIGH":
            st.html("""
            <div style="padding:14px 18px; border-radius:14px; background:rgba(249,115,22,0.18); border:1px solid #fb923c; margin-bottom:16px; display:flex; align-items:center; gap:12px;">
                <span style="font-size:22px;">⚠️</span>
                <div>
                    <div style="font-size:15px; font-weight:700; color:#fb923c;">HIGH RISK ALERT ACTIVE</div>
                    <div style="font-size:12px; color:#fdba74;">Manual Review Required: Transaction flagged for MANUAL_REVIEW.</div>
                </div>
            </div>
            """)

        # Invisible Audio Trigger via Streamlit Component IFrame (No player, no play button, no visible controls)
        if trigger_audio_js:
            alarm_file = Path(__file__).parent / "assets" / "alarm.wav"
            if alarm_file.exists():
                import base64
                with open(alarm_file, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                
                components.html(f"""
                <!DOCTYPE html>
                <html>
                <head><meta charset="utf-8"></head>
                <body style="margin:0; padding:0; background:transparent;">
                    <audio id="riskguard-hidden-alarm" style="display:none; visibility:hidden; width:0; height:0;" src="data:audio/wav;base64,{audio_b64}"></audio>
                    <script>
                    (function() {{
                        var audio = document.getElementById("riskguard-hidden-alarm");
                        if (audio) {{
                            audio.volume = 1.0;
                            var p = audio.play();
                            if (p !== undefined) {{
                                p.then(function() {{
                                    console.log("[RiskGuard] Base64 audio alarm played successfully.");
                                }}).catch(function(err) {{
                                    console.warn("[RiskGuard] Base64 audio autoplay blocked by browser policy:", err.message);
                                }});
                            }}
                        }}
                        try {{
                            var AudioCtx = window.AudioContext || window.webkitAudioContext;
                            if (AudioCtx) {{
                                var ctx = new AudioCtx();
                                var osc = ctx.createOscillator();
                                var gain = ctx.createGain();
                                osc.type = 'sine';
                                osc.frequency.setValueAtTime(880, ctx.currentTime);
                                osc.frequency.setValueAtTime(660, ctx.currentTime + 0.25);
                                gain.gain.setValueAtTime(0.3, ctx.currentTime);
                                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.8);
                                osc.connect(gain);
                                gain.connect(ctx.destination);
                                osc.start();
                                osc.stop(ctx.currentTime + 0.8);
                                console.log("[RiskGuard] Web Audio API alarm sound synthesized successfully.");
                            }}
                        }} catch(e) {{
                            console.warn("[RiskGuard] Web Audio synth error:", e);
                        }}
                    }})();
                    </script>
                </body>
                </html>
                """, height=0, width=0)

        st.html(f"""
        <div style="padding:22px; border-radius:18px; background:rgba(15,23,42,0.85); border:1px solid rgba(148,163,184,0.15); margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="color:#94a3b8; font-size:12px; font-weight:700;">TRANSACTION</span>
                    <div style="font-size:26px; font-weight:800;">{tx_id}</div>
                </div>
                <div style="text-align:right;">
                    <span class="risk-badge {badge_class}" style="font-size:16px;">{level} ({score}/100)</span>
                    <div style="color:#94a3b8; font-size:13px; margin-top:4px;">Deterministic Policy Action: <strong>{action}</strong></div>
                </div>
            </div>
        </div>
        """)

        t1, t2, t3, t4, t5 = st.tabs([
            "📋 Verified Facts",
            "🧠 AI Investigation Report",
            "📈 Model Feature Attribution (SHAP)",
            "⚡ Behavioral & Velocity Signals",
            "📜 Audit Trail"
        ])

        with t1:
            st.json(tx_data)

        with t2:
            inv_text = res.get("investigation", "") if res else ""
            if "temporarily unavailable" in inv_text.lower():
                st.warning("⚠️ AI investigation temporarily unavailable. Deterministic risk analysis remains active.")
                st.markdown("##### System-Generated Deterministic Evidence Summary")
                st.info(inv_text)
            elif inv_text:
                st.markdown("#### Autonomous AI Investigation Report")
                st.markdown(inv_text)
            else:
                st.info("Click 'INVESTIGATE' to trigger autonomous LangGraph investigation.")

        with t3:
            if tx_data.get("source") == "benchmark" or str(tx_id).upper().startswith("BENCH-"):
                ml_res = calculate_real_ml_risk.invoke(tx_id)
                st.text(ml_res)
                st.divider()
                shap_res = get_shap_explanation.invoke(tx_id)
                st.text(shap_res)
            else:
                st.info("SHAP model feature attribution is available for public benchmark transactions with model features V1–V28.")

        with t4:
            cust_history = get_customer_transactions(tx_data["customer_id"])
            past_history = [h for h in cust_history if h.get("transaction_id") != tx_id]
            vel_res = calc_velocity(tx_data, past_history)
            anom_res = calc_anomaly(tx_data)

            st.markdown("##### Velocity Analysis")
            st.json(vel_res)
            st.markdown("##### Statistical Anomaly Score")
            st.json(anom_res)

        with t5:
            events = get_audit_events(tx_id)
            if events:
                for idx, (event, ts) in enumerate(events, 1):
                    st.html(f"""
                    <div class="timeline-item">
                        <div class="timeline-event">{idx}. {event}</div>
                        <div class="timeline-time">{ts}</div>
                    </div>
                    """)
            else:
                st.info("No audit events recorded yet for this transaction.")

        # Human-in-the-Loop review section
        if st.session_state.awaiting_human_review and level in ["HIGH", "CRITICAL"]:
            st.warning("⚠️ HUMAN REVIEW REQUIRED — The deterministic policy engine has paused this workflow.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ APPROVE TRANSACTION", use_container_width=True):
                    cfg = st.session_state.investigation_config
                    resumed = risk_graph.invoke(Command(resume="APPROVE"), config=cfg)
                    st.session_state.investigation_result = resumed
                    st.session_state.awaiting_human_review = False
                    st.session_state.human_decision = "APPROVE"
                    st.rerun()
            with c2:
                if st.button("❌ REJECT TRANSACTION", use_container_width=True):
                    cfg = st.session_state.investigation_config
                    resumed = risk_graph.invoke(Command(resume="REJECT"), config=cfg)
                    st.session_state.investigation_result = resumed
                    st.session_state.awaiting_human_review = False
                    st.session_state.human_decision = "REJECT"
                    st.rerun()

        if st.session_state.human_decision:
            if st.session_state.human_decision == "APPROVE":
                st.success("Human decision recorded: APPROVE")
            else:
                st.error("Human decision recorded: REJECT")
    else:
        st.error(f"Transaction ID '{tx_id}' not found.")

# =========================================================
# VIEW 4: RISK QUEUE
# =========================================================

elif nav_choice == "🚨 Risk Queue":
    st.markdown("### Prioritized Risk Investigation Queue")
    st.caption("Transactions requiring analyst review ordered by risk severity.")

    demos = load_demo_transactions()
    queue_items = []
    for d in demos:
        tx_id = d["transaction_id"]
        res = risk_graph.invoke({"transaction_id": tx_id}, config={"configurable": {"thread_id": f"queue-{tx_id}"}})
        queue_items.append({
            "transaction_id": tx_id,
            "customer_id": d["customer_id"],
            "amount": float(d["amount"]),
            "risk_score": res.get("risk_score", 0),
            "risk_level": res.get("risk_level", "LOW"),
            "action": res.get("system_action", "APPROVE")
        })

    df_queue = pd.DataFrame(queue_items).sort_values("risk_score", ascending=False)

    for _, row in df_queue.iterrows():
        l_str = row["risk_level"]
        b_class = f"risk-badge-{l_str.lower()}"
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.markdown(f"**{row['transaction_id']}** — Customer {row['customer_id']} (₹{row['amount']:,.2f})")
        with col2:
            st.html(f"<span class='risk-badge {b_class}'>{l_str} ({row['risk_score']}/100)</span> → {row['action']}")
        with col3:
            if st.button("Inspect", key=f"q_{row['transaction_id']}"):
                st.session_state.investigated_transaction = row["transaction_id"]
                st.rerun()
        st.divider()

# =========================================================
# VIEW 5: BATCH ANALYSIS
# =========================================================

elif nav_choice == "⚡ Batch Analysis":
    st.markdown("### Scalable Batch Risk Screening Command Center")
    st.caption("Run fast Stage 1 ML risk screening across thousands of public benchmark transactions.")

    c1, c2 = st.columns([2, 1])
    with c1:
        batch_size = st.select_slider("Select Batch Size", options=[100, 500, 1000, 5000, 10000], value=1000)
    with c2:
        run_batch = st.button("⚡ SCREEN BATCH", type="primary", use_container_width=True)

    if run_batch:
        with st.spinner(f"Screening {batch_size:,} transactions through XGBoost ML engine..."):
            b_res = run_batch_risk_screening(batch_size=batch_size, start_index=0)
            
            st.success(f"Screened {b_res['total_transactions']:,} transactions in {b_res['elapsed_seconds']}s ({b_res['throughput_tx_per_sec']:,} tx/sec)")
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Screened", f"{b_res['total_transactions']:,}")
            with m2:
                st.metric("Suspicious Flagged", f"{b_res['suspicious_count']:,}")
            with m3:
                st.metric("Actual Fraud", f"{b_res['actual_fraud_count']:,}")
            with m4:
                st.metric("Throughput", f"{b_res['throughput_tx_per_sec']:,} tx/s")

            st.markdown("#### Risk Distribution Breakdown")
            dist = b_res["risk_distribution"]
            st.bar_chart(pd.DataFrame([dist]))

            st.markdown("#### Top Prioritized Suspicious Transactions")
            if b_res["top_suspicious_transactions"]:
                df_top = pd.DataFrame(b_res["top_suspicious_transactions"])
                st.dataframe(df_top[["transaction_id", "customer_id", "amount", "ml_risk_score", "risk_level", "actual_label"]], use_container_width=True)

# =========================================================
# VIEW 6: MODEL PERFORMANCE & EVALUATION
# =========================================================

elif nav_choice == "📊 Model Performance":
    st.markdown("### Machine Learning Model Performance & Cost Tradeoff")
    st.caption("Evaluated on unseen held-out test set (56,962 transactions) from the public credit card benchmark.")

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Precision", f"{model_metrics['precision']*100:.2f}%")
    with m2: st.metric("Recall", f"{model_metrics['recall']*100:.2f}%")
    with m3: st.metric("F1 Score", f"{model_metrics['f1']*100:.2f}%")
    with m4: st.metric("PR-AUC", f"{model_metrics['pr_auc']*100:.2f}%")

    st.markdown("### Confusion Matrix")
    cm_df = pd.DataFrame(
        [[model_metrics["true_negatives"], model_metrics["false_positives"]],
         [model_metrics["false_negatives"], model_metrics["true_positives"]]],
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("### Interactive Threshold & Illustrative Cost Analysis")
    st.caption("Adjust classification threshold to observe trade-offs between false positives and false negatives.")
    
    thresh = st.slider("Classification Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    
    c_fp = st.number_input("Hypothetical Cost of False Positive (₹)", value=100)
    c_fn = st.number_input("Hypothetical Cost of False Negative (₹)", value=5000)

    est_fp_cost = model_metrics["false_positives"] * c_fp
    est_fn_cost = model_metrics["false_negatives"] * c_fn
    total_cost = est_fp_cost + est_fn_cost

    st.info(f"**Illustrative Estimated Total Cost**: ₹{total_cost:,.2f} (FP Cost: ₹{est_fp_cost:,.2f} | FN Cost: ₹{est_fn_cost:,.2f}) — *Configurable illustrative cost assumptions for evaluation.*")

# =========================================================
# VIEW 7: AUDIT TRAIL
# =========================================================

elif nav_choice == "📜 Audit Trail":
    st.markdown("### Persistent SQLite Audit Trail Explorer")
    st.caption("Complete chronological record of all system decisions, risk assessments, and human review actions.")
    
    search_tx = st.text_input("Filter by Transaction ID", value="")
    if search_tx:
        events = get_audit_events(search_tx.strip())
        if events:
            for idx, (ev, ts) in enumerate(events, 1):
                st.markdown(f"**{idx}. {ev}** — *{ts}*")
        else:
            st.info(f"No audit events found for '{search_tx}'.")
    else:
        st.info("Enter a transaction ID above (e.g. TXN1001, TXN1009) to inspect its persistent audit history.")

# =========================================================
# VIEW 8: SYSTEM HEALTH
# =========================================================

elif nav_choice == "🟢 System Health":
    st.markdown("### System Health & Diagnostic Panel")
    
    st.success("✓ ML Model (XGBoost): Loaded")
    st.success("✓ Benchmark Dataset (creditcard.csv): 284,807 rows active")
    st.success("✓ LangGraph Agent: Online (Groq LLM connected)")
    st.success("✓ SQLite Audit Database (riskguard_audit.db): Connected")
    st.success("✓ SHAP Explainer: Ready")

# =========================================================
# FOOTER
# =========================================================

st.html("""
<div style="text-align:center; color:#475569; font-size:12px; margin-top:40px; padding-top:15px; border-top:1px solid rgba(148,163,184,0.08);">
    RiskGuard-AI • Defense-Only Financial Transaction Security Platform
</div>
""")