import unittest
from transaction_data import (
    get_transaction_by_id,
    get_customer_transactions,
    load_demo_transactions,
    get_benchmark_batch
)

class TestTransactionData(unittest.TestCase):
    def test_load_demo_transactions(self):
        demos = load_demo_transactions()
        self.assertEqual(len(demos), 9)
        self.assertEqual(demos[0]["transaction_id"], "TXN1001")

    def test_get_transaction_by_id_demo(self):
        tx = get_transaction_by_id("TXN1005")
        self.assertIsNotNone(tx)
        self.assertEqual(tx["transaction_id"], "TXN1005")
        self.assertEqual(tx["customer_id"], "CUST003")
        self.assertEqual(tx["source"], "demo")

    def test_get_transaction_by_id_benchmark(self):
        tx = get_transaction_by_id("BENCH-000001")
        self.assertIsNotNone(tx)
        self.assertEqual(tx["transaction_id"], "BENCH-000001")
        self.assertEqual(tx["benchmark_row_index"], 0)
        self.assertEqual(tx["source"], "benchmark")

    def test_get_customer_transactions_no_history(self):
        history = get_customer_transactions("CUST_NON_EXISTENT")
        self.assertEqual(len(history), 0)

    def test_get_benchmark_batch(self):
        batch = get_benchmark_batch(limit=10, start_index=0)
        self.assertEqual(len(batch), 10)
        self.assertEqual(batch[0]["transaction_id"], "BENCH-000001")
        self.assertEqual(batch[9]["transaction_id"], "BENCH-000010")

if __name__ == "__main__":
    unittest.main()
