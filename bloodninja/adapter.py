"""Real upstream agent-scaling adapter. Imports upstream only when installed.

The original LeadAgent handles planning, feedback, stopping, and synthesis.
The original WorkerSubagent handles tools, prompts and per-role private history.
Only lead-to-worker transport/scheduling changes: every role job enters SQLite,
and ONE consumer executes all roles serially using the shared underlying LLM.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import MethodType
from typing import Any

from .broker import Broker


def consume_round(lead: Any, plan: Any, round_num: int, broker: Broker) -> dict:
    """Two-phase scheduler: publish all coordination messages, then consume.

    Building every coordinator message *before* worker execution is crucial:
    the original centralized architecture issues all round-specific instructions
    before awaiting any worker result. A naive per-role publish/consume loop would
    leak preceding workers' new findings to later workers within the same round.
    """
    from agent_scaling.agents.multiagent_components.conversation import SubAgentRoundResult

    if not lead.subagents:
        raise ValueError('No subagents available for coordination')
    if not plan.subtasks:
        raise ValueError('Invalid plan: no subtasks found')
    active = [name for name, agent in lead.subagents.items()
              if agent.conv_history.status == 'active'
              and not agent.should_stop_due_to_rate_limiting()]
    if not active:
        return {}
    targets = {task.agent_id: task for task in plan.subtasks}
    # Single producer, same agent enumeration, same prompts, same coordination calls.
    expected = []
    for agent_id in active:
        subtask = targets.get(agent_id)
        if subtask is None:
            raise ValueError(f'Plan omits active worker {agent_id}')
        message = lead._create_message_for_agent(lead.subagents[agent_id], subtask, round_num)
        broker.publish(agent_id, round_num, message, job_id=f'round-{round_num}:{agent_id}')
        expected.append(agent_id)
    results = {}
    for _ in expected:
        job = broker.take()
        if job is None:
            raise RuntimeError('Queue unexpectedly empty')
        if job.role not in lead.subagents or job.round_num != round_num:
            raise RuntimeError('Invalid queue delivery')
        try:
            result = lead.subagents[job.role].process_orchestrator_message(job.message)
        except Exception as exc:
            broker.finish(job.id, error=f'{type(exc).__name__}: {exc}')
            result = SubAgentRoundResult(
                agent_id=job.role, findings='',
                env_status=lead.subagents[job.role].env.env_status(),
                error_msg=f'Agent task error: {exc}',
            )
        else:
            broker.finish(job.id)
        results[job.role] = result
    return results


async def queued_round(lead: Any, plan: Any, round_num: int) -> dict:
    """Serialize *all* worker calls through one broker consumer thread.

    No wall-time timeout is imposed here because cancelling asyncio.to_thread
    cannot stop tool side effects. Record wall-time separately. This arm will
    be non-parallel and may exceed the control's 300s round timeout: this is a
    known parity limitation requiring an equal-time ablation.
    """
    broker = getattr(lead, '_bloodninja_broker')
    return await asyncio.to_thread(consume_round, lead, plan, round_num, broker)


def register_queued_agent() -> None:
    """Register additional architecture in the upstream agent registry."""
    from agent_scaling.agents.multiagent_centralized import CentralizedMultiAgentSystem
    from agent_scaling.agents.registry import register_agent

    @register_agent('single-worker-queued')
    class SingleWorkerQueued(CentralizedMultiAgentSystem):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The original BaseAgent and every WorkerSubagent use the same
            # self.llm reference. Each binds tools to its own environment,
            # and receives only its own private history.
            self.lead_agent._coordinate_and_run_subagents = MethodType(
                queued_round, self.lead_agent)
            self.lead_agent._bloodninja_broker = None

        async def run_agent_async(self, instance, instance_dir=None,
                                  llm_params=None, instance_idx=None):
            db_path = Path(instance_dir) / 'role_queue.sqlite' if instance_dir else ':memory:'
            broker = Broker(db_path)
            self.lead_agent._bloodninja_broker = broker
            try:
                return await super().run_agent_async(instance, instance_dir,
                                                      llm_params, instance_idx)
            finally:
                broker.close()

    return None
