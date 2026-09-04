import time
from typing import Dict, Any, Generator
from transaction_data import get_benchmark_batch, load_demo_transactions
from real_ml_risk_calculator import calculate_real_ml_risk
from risk_fusion import map_score_to_level
from audit_storage import save_audit_event

class StreamSimulator:
    def __init__(self, source: str = "benchmark", start_index: int = 0):
        self.source = source
        self.current_index = start_index

    def next_event(self) -> Dict[str, Any]:
        if self.source == "demo":
            demos = load_demo_transactions()
            idx = self.current_index % len(demos)
            d = demos[idx]
            self.current_index += 1
            return {
                "transaction_id": d["transaction_id"],
                "customer_id": d["customer_id"],
                "amount": float(d["amount"]),
                "location": d["location"],
                "device": d["device"],
                "payment_method": d["payment_method"],
                "hour": int(d["hour"]),
                "source": "demo",
                "risk_score": 15,
                "risk_level": "LOW",
                "timestamp": time.strftime("%H:%M:%S")
            }
        else:
            batch = get_benchmark_batch(limit=1, start_index=self.current_index)
            if not batch:
                self.current_index = 0
                batch = get_benchmark_batch(limit=1, start_index=0)
            tx = batch[0]
            self.current_index += 1
            
            # Fast ML score
            ml_out = calculate_real_ml_risk.invoke(tx["transaction_id"])
            score = 0
            level = "LOW"
            for line in ml_out.splitlines():
                if line.startswith("ML Risk Score:"):
                    score = int(line.split(":")[1].split("/")[0].strip())
                elif line.startswith("ML Risk Level:"):
                    level = line.split(":")[1].strip()

            tx["risk_score"] = score
            tx["risk_level"] = level
            tx["timestamp"] = time.strftime("%H:%M:%S")
            
            save_audit_event(tx["transaction_id"], f"Ingested in Stream: Score {score} ({level})")
            return tx
