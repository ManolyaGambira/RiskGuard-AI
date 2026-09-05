import unittest
import numpy as np
from pathlib import Path

class TestThresholdDynamics(unittest.TestCase):
    def setUp(self):
        self.npz_path = Path(__file__).parent.parent / "data" / "test_set_eval.npz"
        self.assertTrue(self.npz_path.exists(), "test_set_eval.npz dataset must exist.")
        data = np.load(self.npz_path)
        self.y_true = data["y_test"]
        self.probs = data["probs"]

    def _eval_at(self, thresh):
        preds = (self.probs >= thresh).astype(int)
        tp = int(np.sum((preds == 1) & (self.y_true == 1)))
        tn = int(np.sum((preds == 0) & (self.y_true == 0)))
        fp = int(np.sum((preds == 1) & (self.y_true == 0)))
        fn = int(np.sum((preds == 0) & (self.y_true == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(self.y_true)
        cost = fp * 100 + fn * 5000
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "f1": f1, "acc": acc, "cost": cost}

    def test_dynamic_metrics_change_with_threshold(self):
        res_30 = self._eval_at(0.30)
        res_50 = self._eval_at(0.50)
        res_70 = self._eval_at(0.70)

        # Baseline at 0.50 must match expected metrics
        self.assertEqual(res_50["tp"], 82)
        self.assertEqual(res_50["tn"], 56841)
        self.assertEqual(res_50["fp"], 23)
        self.assertEqual(res_50["fn"], 16)

        # Lower threshold (0.30) increases FP and Recall
        self.assertGreater(res_30["fp"], res_50["fp"])
        self.assertGreater(res_30["rec"], res_50["rec"])

        # Higher threshold (0.70) decreases FP and increases Precision
        self.assertLess(res_70["fp"], res_50["fp"])
        self.assertGreater(res_70["prec"], res_50["prec"])

        # Costs change dynamically
        self.assertNotEqual(res_30["cost"], res_50["cost"])
        self.assertNotEqual(res_70["cost"], res_50["cost"])

if __name__ == "__main__":
    unittest.main()
