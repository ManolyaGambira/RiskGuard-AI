import unittest
from transaction_data import get_operational_risk_queue, get_transaction_by_id
from batch_processor import run_batch_risk_screening
from audit_storage import get_audit_events

class TestFinal3Fixes(unittest.TestCase):

    def test_risk_queue_mixed_distribution_and_sorting(self):
        queue = get_operational_risk_queue()
        self.assertGreater(len(queue), 1, "Operational queue must contain multiple items.")

        levels = {item["risk_level"] for item in queue}
        self.assertIn("CRITICAL", levels, "Queue must contain CRITICAL transactions.")
        self.assertIn("HIGH", levels, "Queue must contain HIGH transactions.")
        self.assertIn("MEDIUM", levels, "Queue must contain MEDIUM transactions.")
        self.assertNotIn("LOW", levels, "LOW risk transactions must be excluded from analyst review queue.")

        queued_ids = [item["transaction_id"] for item in queue]
        self.assertIn("TXN1009", queued_ids, "TXN1009 must remain in operational queue.")

        # Verify strict severity and descending risk score sorting
        severity_order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
        for i in range(len(queue) - 1):
            curr_sev = severity_order[queue[i]["risk_level"]]
            next_sev = severity_order[queue[i+1]["risk_level"]]
            self.assertGreaterEqual(curr_sev, next_sev)
            if curr_sev == next_sev:
                self.assertGreaterEqual(queue[i]["risk_score"], queue[i+1]["risk_score"])

    def test_batch_chart_data_consistency(self):
        b_res = run_batch_risk_screening(batch_size=100, start_index=0)
        dist = b_res["risk_distribution"]
        
        categories = list(dist.keys())
        self.assertEqual(categories, ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        
        sum_counts = sum(dist.values())
        self.assertEqual(sum_counts, b_res["total_transactions"])

    def test_batch_audit_trail_persistence(self):
        b_res = run_batch_risk_screening(batch_size=10, start_index=500)
        
        # Test searching for a batch transaction ID
        target_bench_id = "BENCH-000501"
        target_batch_id = "BATCH-000501"
        
        events_bench = get_audit_events(target_bench_id)
        events_batch = get_audit_events(target_batch_id)
        
        self.assertGreater(len(events_bench), 0, f"Audit events must exist for {target_bench_id}")
        self.assertGreater(len(events_batch), 0, f"Audit events must exist for {target_batch_id}")
        
        bench_texts = [e[0] for e in events_bench]
        self.assertTrue(any("Batch Screening Completed" in t for t in bench_texts))
        self.assertTrue(any("Policy decision" in t for t in bench_texts))
        self.assertFalse(any("Human decision" in t for t in bench_texts), "No fake human decision events should be generated for normal batch screening.")

if __name__ == "__main__":
    unittest.main()
