"""Contract parity: broker-free serialization must reproduce Q's role inputs."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloodninja.broker import Broker
from bloodninja.adapter import consume_round
from bloodninja.direct import execute_serialized_round
from test_upstream_adapter import Lead, AdapterTests

class DirectControlTests(unittest.TestCase):
    def test_broker_free_matches_queue_role_inputs_and_outputs(self):
        q_events, direct_events = [], []
        q_lead, direct_lead = Lead(q_events), Lead(direct_events)
        q_plan = SimpleNamespace(subtasks=[SimpleNamespace(agent_id=r) for r in q_lead.subagents])
        direct_plan = SimpleNamespace(subtasks=[SimpleNamespace(agent_id=r) for r in direct_lead.subagents])
        broker = Broker()
        with AdapterTests._upstream_result_module():
            q = consume_round(q_lead, q_plan, 1, broker)
            e = execute_serialized_round(direct_lead, direct_plan, 1)
        self.assertEqual(q_events, direct_events)
        self.assertEqual([(r, q[r].findings, q[r].error_msg) for r in q],
                         [(r, e[r].findings, e[r].error_msg) for r in e])
        self.assertEqual([x[0] for x in direct_events],
                         ['publish_instruction'] * 3 + ['consume'] * 3)
        broker.close()

    def test_broker_free_matches_queue_failure_conversion(self):
        q_events, direct_events = [], []
        q_lead, direct_lead = Lead(q_events,fail=True), Lead(direct_events,fail=True)
        q_plan = SimpleNamespace(subtasks=[SimpleNamespace(agent_id=r) for r in q_lead.subagents])
        direct_plan = SimpleNamespace(subtasks=[SimpleNamespace(agent_id=r) for r in direct_lead.subagents])
        broker=Broker()
        with AdapterTests._upstream_result_module():
            q=consume_round(q_lead,q_plan,1,broker)
            e=execute_serialized_round(direct_lead,direct_plan,1)
        self.assertEqual(q_events,direct_events)
        self.assertEqual(q['b'].error_msg,e['b'].error_msg)
        broker.close()

if __name__ == '__main__':
    unittest.main()
