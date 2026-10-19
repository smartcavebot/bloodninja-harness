"""Dependency-free contract tests for the *actual* upstream dispatch replacement."""
from pathlib import Path
import ast
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloodninja.broker import Broker
from bloodninja.adapter import consume_round


class Result:
    def __init__(self, *, agent_id, findings, env_status, error_msg=None):
        self.agent_id, self.findings = agent_id, findings
        self.env_status, self.error_msg = env_status, error_msg


class Worker:
    def __init__(self, name, events, fail=False):
        self.name, self.events, self.fail = name, events, fail
        self.conv_history = types.SimpleNamespace(status='active')
        self.ctx = []
        self.env = types.SimpleNamespace(env_status=lambda: {'ok': not fail})

    def should_stop_due_to_rate_limiting(self):
        return False

    def process_orchestrator_message(self, message):
        self.events.append(('consume', self.name, message))
        self.ctx.append(message)
        if self.fail:
            raise RuntimeError('controlled worker failure')
        return Result(agent_id=self.name, findings=f'finding-{self.name}', env_status={'ok': True})


class Lead:
    def __init__(self, events, fail=False):
        self.events = events
        self.subagents = {name: Worker(name, events, fail=(fail and name == 'b'))
                          for name in ('a', 'b', 'c')}

    def _create_message_for_agent(self, agent, subtask, round_num):
        self.events.append(('publish_instruction', agent.name, round_num))
        # This intentionally observes agent state. Later worker results must NOT
        # be visible to planning instructions within the same round.
        seen = ','.join(f'{k}:{len(v.ctx)}' for k, v in self.subagents.items())
        return f'round={round_num};role={agent.name};seen={seen}'


class AdapterTests(unittest.TestCase):
    @staticmethod
    def _upstream_result_module():
        root = 'agent_scaling'
        return patch.dict(sys.modules, {
            root: types.ModuleType(root),
            root+'.agents': types.ModuleType(root+'.agents'),
            root+'.agents.multiagent_components': types.ModuleType(root+'.agents.multiagent_components'),
            root+'.agents.multiagent_components.conversation': types.SimpleNamespace(SubAgentRoundResult=Result),
        })

    def test_fifo_private_context_and_no_within_round_leak(self):
        events = []
        lead = Lead(events)
        plan = types.SimpleNamespace(subtasks=[types.SimpleNamespace(agent_id=k) for k in lead.subagents])
        with tempfile.TemporaryDirectory() as temp:
            broker = Broker(Path(temp) / 'jobs.db')
            with self._upstream_result_module():
                result = consume_round(lead, plan, 1, broker)
                self.assertEqual(list(result), ['a', 'b', 'c'])
                self.assertEqual([x[0] for x in events],
                                 ['publish_instruction'] * 3 + ['consume'] * 3)
                self.assertTrue(all('seen=a:0,b:0,c:0' in x[2]
                                    for x in events if x[0] == 'consume'))
                self.assertEqual({k: len(v.ctx) for k, v in lead.subagents.items()},
                                 dict(a=1, b=1, c=1))
                self.assertTrue(all(status == 'complete' for _, _, status in broker.states()))
                consume_round(lead, plan, 2, broker)
                self.assertTrue(all('seen=a:1,b:1,c:1' in x[2]
                                    for x in events if x[0] == 'consume' and 'round=2' in x[2]))
                self.assertEqual(len(broker.states()), 6)
            broker.close()
            # Durable queue audit survives reopen; no job is pending.
            broker = Broker(Path(temp) / 'jobs.db')
            self.assertEqual([s for _, _, s in broker.states()], ['complete'] * 6)
            self.assertIsNone(broker.take())
            broker.close()

    def test_worker_failure_is_recorded_without_replay(self):
        events = []
        lead = Lead(events, fail=True)
        plan = types.SimpleNamespace(subtasks=[types.SimpleNamespace(agent_id=k) for k in lead.subagents])
        broker = Broker()
        with self._upstream_result_module():
            result = consume_round(lead, plan, 1, broker)
        self.assertEqual(result['b'].error_msg, 'Agent task error: controlled worker failure')
        self.assertEqual([x[2] for x in broker.states()], ['complete', 'failed', 'complete'])
        self.assertIsNone(broker.take())
        broker.close()

    def test_idempotent_job_ids(self):
        broker = Broker()
        broker.publish('a', 1, 'same', job_id='r1:a')
        broker.publish('a', 1, 'same', job_id='r1:a')
        self.assertEqual(len(broker.states()), 1)
        job = broker.take()
        self.assertEqual((job.id, job.role, job.message), ('r1:a','a','same'))
        broker.finish(job.id)
        with self.assertRaises(ValueError):
            broker.finish(job.id)
        broker.close()

    def test_upstream_submodule_contract_if_present(self):
        source = Path(__file__).resolve().parents[1] / 'upstream/agent-scaling/agent_scaling/agents/multiagent_components/mas_lead_agent.py'
        if not source.exists():
            self.skipTest('submodule not initialized locally')
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LeadAgent')
        funcs = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertTrue({'_create_message_for_agent', '_coordinate_and_run_subagents', '_create_subagents'} <= funcs)
        worker_path = source.with_name('mas_subagent.py')
        worker_src = worker_path.read_text()
        self.assertIn('llm=agent.llm', worker_src)
        self.assertIn('self.conv_history = SubAgentConversationHistory', worker_src)


if __name__ == '__main__':
    unittest.main()
