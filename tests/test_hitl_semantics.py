import unittest
# pyrefly: ignore [missing-import]
from langgraph.types import Command
from risk_graph import risk_graph
from audit_storage import get_audit_events

class TestHITLSemantics(unittest.TestCase):
    def test_human_review_preserves_risk_and_records_disposition(self):
        tx_id = "TXN1009"
        config = {"configurable": {"thread_id": "test-hitl-1009"}}

        # Initial run pauses at interrupt for CRITICAL transaction
        res_initial = risk_graph.invoke({"transaction_id": tx_id}, config=config)
        self.assertEqual(res_initial.get("risk_score"), 100)
        self.assertEqual(res_initial.get("risk_level"), "CRITICAL")
        self.assertEqual(res_initial.get("system_action"), "HOLD_AND_ESCALATE")

        # Resume with human approval
        res_approved = risk_graph.invoke(Command(resume="APPROVE"), config=config)
        self.assertEqual(res_approved.get("risk_score"), 100)
        self.assertEqual(res_approved.get("risk_level"), "CRITICAL")
        self.assertEqual(res_approved.get("system_action"), "HOLD_AND_ESCALATE")
        self.assertEqual(res_approved.get("human_decision"), "APPROVE")
        self.assertEqual(res_approved.get("final_disposition"), "APPROVED AFTER HUMAN REVIEW")

        # Verify audit trail events
        events = [e[0] for e in get_audit_events(tx_id)]
        self.assertTrue(any("Human decision recorded: APPROVED BY HUMAN" in e for e in events))
        self.assertTrue(any("Final disposition: APPROVED AFTER HUMAN REVIEW" in e for e in events))

if __name__ == "__main__":
    unittest.main()
