import unittest
from ml_risk_calculator import calculate_behavioral_risk
from velocity_engine import calculate_velocity_risk
from anomaly_engine import calculate_anomaly_score
from risk_fusion import fuse_risk_evidence, map_score_to_level
from policy_engine import determine_action
from risk_schema import RiskAssessment

class TestRiskEngine(unittest.TestCase):
    def test_txn1005_no_history_regression(self):
        """Verify TXN1005 (customer with no history) produces structured score without raising ValueError."""
        res = calculate_behavioral_risk.invoke("TXN1005")
        self.assertIn("Behavioral Risk Score:", res)
        self.assertIn("Behavioral Risk Level:", res)
        self.assertIn("insufficient_history", res)
        self.assertIn("TXN1005", res)

    def test_map_score_to_level(self):
        self.assertEqual(map_score_to_level(10), "LOW")
        self.assertEqual(map_score_to_level(45), "MEDIUM")
        self.assertEqual(map_score_to_level(75), "HIGH")
        self.assertEqual(map_score_to_level(90), "CRITICAL")

    def test_policy_engine(self):
        low = determine_action(RiskAssessment(risk_score=10, risk_level="LOW", recommendation="", reasons=[]))
        med = determine_action(RiskAssessment(risk_score=50, risk_level="MEDIUM", recommendation="", reasons=[]))
        high = determine_action(RiskAssessment(risk_score=75, risk_level="HIGH", recommendation="", reasons=[]))
        crit = determine_action(RiskAssessment(risk_score=90, risk_level="CRITICAL", recommendation="", reasons=[]))
        
        self.assertEqual(low, "APPROVE")
        self.assertEqual(med, "MONITOR")
        self.assertEqual(high, "MANUAL_REVIEW")
        self.assertEqual(crit, "HOLD_AND_ESCALATE")

    def test_risk_fusion(self):
        fusion = fuse_risk_evidence(
            ml_score=85,
            behavioral_score=70,
            velocity_score=50,
            anomaly_score=60,
            history_status="available"
        )
        self.assertGreater(fusion["fused_risk_score"], 70)
        self.assertIn(fusion["risk_level"], ["HIGH", "CRITICAL"])
        self.assertIsNotNone(fusion["evidence_completeness"])

    def test_velocity_engine(self):
        tx = {"amount": 5000, "time": 100}
        history = [{"amount": 1000, "time": 90}, {"amount": 1000, "time": 85}, {"amount": 1000, "time": 80}]
        vel = calculate_velocity_risk(tx, history)
        self.assertGreaterEqual(vel["velocity_score"], 30)
        self.assertEqual(vel["count_1m"], 4)

if __name__ == "__main__":
    unittest.main()
