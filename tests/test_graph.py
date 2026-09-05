import unittest
from pathlib import Path
from unittest.mock import patch
from risk_graph import risk_assessment_node, policy_decision_node, investigation_agent_node

class TestRiskGraph(unittest.TestCase):
    def test_risk_graph_assessment_demo(self):
        state = {"transaction_id": "TXN1001"}
        res = risk_assessment_node(state)
        self.assertIsNotNone(res["risk_score"])
        self.assertIn(res["risk_level"], ["LOW", "MEDIUM", "HIGH", "CRITICAL"])

    def test_risk_graph_assessment_benchmark(self):
        state = {"transaction_id": "BENCH-000542"}
        res = risk_assessment_node(state)
        self.assertIsNotNone(res["risk_score"])
        self.assertIn(res["risk_level"], ["LOW", "MEDIUM", "HIGH", "CRITICAL"])

    def test_policy_decision(self):
        state = {"transaction_id": "TXN1009", "risk_score": 100, "risk_level": "CRITICAL"}
        res = policy_decision_node(state)
        self.assertEqual(res["system_action"], "HOLD_AND_ESCALATE")

    def test_alarm_asset_exists_and_critical_trigger(self):
        alarm_file = Path(__file__).parent.parent / "assets" / "alarm.wav"
        self.assertTrue(alarm_file.exists(), "Local alarm.wav audio file must exist in assets/ directory.")
        
        state = {"transaction_id": "TXN1009"}
        res = risk_assessment_node(state)
        self.assertEqual(res["risk_level"], "CRITICAL")
        self.assertEqual(res["risk_score"], 100)

    @patch("risk_graph.agent.invoke")
    def test_investigation_fallback_on_api_error(self, mock_agent_invoke):
        mock_agent_invoke.side_effect = Exception("Rate limit reached for model groq 429 TPD limit 200000")
        state = {"transaction_id": "TXN1009"}
        res = investigation_agent_node(state)
        inv = res.get("investigation", "")
        self.assertIn("INVESTIGATION SUMMARY", inv)
        self.assertIn("VERIFIED SIGNALS", inv)
        self.assertNotIn("Rate limit reached", inv)
        self.assertNotIn("200000", inv)

if __name__ == "__main__":
    unittest.main()
