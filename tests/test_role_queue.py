import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from role_queue import run_task, SQLiteBroker
from analyze import summarize

class CheckArchitecture(unittest.TestCase):
    def test_queue_matches_direct_when_contexts_isolated(self):
        a, a_trace = run_task("same", mode="direct")
        b, b_trace = run_task("same", mode="queue")
        self.assertEqual(a, b)
        self.assertEqual([x['prior'] for x in a_trace.invocations],
                         [x['prior'] for x in b_trace.invocations])
        self.assertEqual(len(b_trace.messages), 10) # five jobs publish + consume

    def test_no_role_bleed_in_queue(self):
        _, t = run_task("a", mode="queue")
        for invocation in t.invocations:
            if invocation['role'] in ('filings', 'operations', 'markets'):
                self.assertEqual(invocation['prior'], [])
        self.assertEqual(len(t.invocations[-1]['prior']), 2)

    def test_shared_history_negative_control_detectable(self):
        out1, _ = run_task("same", mode="queue")
        out2, t = run_task("same", mode="leaky")
        self.assertNotEqual(out1, out2)
        self.assertEqual(len(t.invocations[-1]['prior']), 8)

    def test_broker_idempotent_and_fifo(self):
        b = SQLiteBroker()
        b.publish('filings','a','id-a')
        b.publish('filings','a','id-a')
        b.publish('markets','b','id-b')
        self.assertEqual(b.consume(), ('id-a','filings','a'))
        self.assertEqual(b.consume(), ('id-b','markets','b'))
        self.assertIsNone(b.consume())
        b.close()

    def test_paired_analytics(self):
        rows = {
           'x': {'single':[0.2,0.4], 'multi':[0.8], 'queue':[0.7]},
           'y': {'single':[0.4], 'multi':[0.6], 'queue':[0.6]},
        }
        n, result = summarize(rows, 300)
        self.assertEqual(n, 2)
        self.assertTrue(.7 < result['captured_gain']['estimate'] < 1)

if __name__ == '__main__':
    unittest.main()
