import unittest
from unittest.mock import patch
from stream_simulator import StreamSimulator
from risk_graph import (
    build_deterministic_investigation_report,
    investigation_agent_node
)

class TestStreamAndFallback(unittest.TestCase):
    def test_stream_speed_mapping(self):
        speed_map = {
            "1 tx/sec": 1.0,
            "2 tx/sec": 0.5,
            "5 tx/sec": 0.2
        }
        self.assertEqual(speed_map["1 tx/sec"], 1.0)
        self.assertEqual(speed_map["2 tx/sec"], 0.5)
        self.assertEqual(speed_map["5 tx/sec"], 0.2)

    def test_stream_simulator_next_event(self):
        sim = StreamSimulator(source="benchmark", start_index=0)
        ev1 = sim.next_event()
        self.assertIsNotNone(ev1)
        self.assertIn("transaction_id", ev1)
        self.assertIn("risk_score", ev1)
        self.assertIn("risk_level", ev1)

        ev2 = sim.next_event()
        self.assertNotEqual(ev1["transaction_id"], ev2["transaction_id"])

    def test_deterministic_investigation_fallback_report_structure(self):
        report = build_deterministic_investigation_report("TXN1009")
        self.assertIn("INVESTIGATION SUMMARY", report)
        self.assertIn("VERIFIED SIGNALS", report)
        self.assertIn("ML EVIDENCE", report)
        self.assertIn("RISK ASSESSMENT", report)
        self.assertIn("RECOMMENDED ACTION", report)
        self.assertIn("TXN1009", report)
        self.assertIn("CRITICAL — 100/100", report)
        self.assertIn("HOLD_AND_ESCALATE", report)

    @patch("risk_graph.agent.invoke")
    def test_groq_failure_fallback_preserves_deterministic_results(self, mock_agent_invoke):
        mock_agent_invoke.side_effect = Exception("Groq HTTP 429 rate limit exceeded or connection timeout")
        
        state = {"transaction_id": "TXN1009"}
        res = investigation_agent_node(state)
        
        self.assertTrue(res.get("is_llm_fallback"), "State must flag is_llm_fallback when LLM call fails.")
        inv_text = res.get("investigation", "")
        
        self.assertIn("INVESTIGATION SUMMARY", inv_text)
        self.assertIn("VERIFIED SIGNALS", inv_text)
        self.assertIn("RISK ASSESSMENT", inv_text)
        self.assertIn("RECOMMENDED ACTION", inv_text)

if __name__ == "__main__":
    unittest.main()
