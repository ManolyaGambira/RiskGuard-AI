import unittest
from transaction_data import get_operational_risk_queue

class TestQueueInspect(unittest.TestCase):
    def test_operational_risk_queue_composition(self):
        queue = get_operational_risk_queue()
        self.assertGreater(len(queue), 1, "Operational risk queue should contain multiple candidate cases.")

        queued_ids = [q["transaction_id"] for q in queue]
        self.assertIn("TXN1009", queued_ids, "TXN1009 must remain in operational queue as demo CRITICAL item.")

        levels = {q["risk_level"] for q in queue}
        self.assertNotIn("LOW", levels, "LOW risk transactions must be excluded from analyst review queue.")

        for item in queue:
            self.assertIn(item["risk_level"], ["MEDIUM", "HIGH", "CRITICAL"])
            self.assertNotEqual(item["action"], "APPROVE")
            # Ensure ground truth Class label was not required or used for queue membership definition
            self.assertNotIn("actual_label", item)

    def test_operational_risk_queue_contains_critical_and_mixed(self):
        queue = get_operational_risk_queue()
        critical_items = [q for q in queue if q["risk_level"] == "CRITICAL"]
        self.assertGreaterEqual(len(critical_items), 2, "Operational queue must contain at least 2 CRITICAL transactions.")
        
        # Verify queue is sorted by severity and descending risk score
        severity_map = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
        for i in range(len(queue) - 1):
            sev_curr = severity_map[queue[i]["risk_level"]]
            sev_next = severity_map[queue[i+1]["risk_level"]]
            self.assertGreaterEqual(sev_curr, sev_next)
            if sev_curr == sev_next:
                self.assertGreaterEqual(queue[i]["risk_score"], queue[i+1]["risk_score"])

if __name__ == "__main__":
    unittest.main()

