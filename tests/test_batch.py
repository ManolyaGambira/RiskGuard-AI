import unittest
from batch_processor import run_batch_risk_screening

class TestBatchProcessor(unittest.TestCase):
    def test_batch_risk_screening_1000(self):
        result = run_batch_risk_screening(batch_size=1000, start_index=0)
        self.assertEqual(result["total_transactions"], 1000)
        self.assertGreater(result["throughput_tx_per_sec"], 50)
        self.assertIn("risk_distribution", result)
        self.assertIn("metrics", result)
        self.assertGreaterEqual(result["metrics"]["precision"], 0.0)

if __name__ == "__main__":
    unittest.main()
