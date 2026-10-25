"""Broker-free serialized execution control (E arm).

All workers retain upstream role-private state, prompts, tools and model
identity. Coordinator builds EVERY round instruction first, then a single
thread invokes workers in deterministic order. No broker, database or queue.
Compare E against Q for queue/persistence effects, and M against E to isolate
serialization versus concurrency. E is not full single-agent virtualization.
"""
from __future__ import annotations

import asyncio
from types import MethodType
from typing import Any


def execute_serialized_round(lead: Any, plan: Any, round_num: int) -> dict:
    from agent_scaling.agents.multiagent_components.conversation import SubAgentRoundResult

    if not lead.subagents:
        raise ValueError('No subagents available for coordination')
    if not plan.subtasks:
        raise ValueError('Invalid plan: no subtasks found')

    active = [name for name, worker in lead.subagents.items()
              if worker.conv_history.status == 'active'
              and not worker.should_stop_due_to_rate_limiting()]
    if not active:
        return {}
    subtasks = {x.agent_id:x for x in plan.subtasks}
    # Preserve upstream round barrier: no worker has run when next role's
    # coordinator instructions are constructed.
    instructions = []
    for agent_id in active:
        if agent_id not in subtasks:
            raise ValueError(f'Plan omits active worker {agent_id}')
        instruction = lead._create_message_for_agent(
            lead.subagents[agent_id], subtasks[agent_id], round_num)
        instructions.append((agent_id, instruction))
    results = {}
    for agent_id, instruction in instructions:
        worker = lead.subagents[agent_id]
        try:
            result = worker.process_orchestrator_message(instruction)
        except Exception as exc:
            result = SubAgentRoundResult(
                agent_id=agent_id, findings='', env_status=worker.env.env_status(),
                error_msg=f'Agent task error: {exc}')
        results[agent_id] = result
    return results


async def serial_round(lead, plan, round_num):
    # Run in the same single worker thread arrangement as Q, but no message broker.
    return await asyncio.to_thread(execute_serialized_round, lead, plan, round_num)


def register_direct_agent():
    from agent_scaling.agents.multiagent_centralized import CentralizedMultiAgentSystem
    from agent_scaling.agents.registry import register_agent

    @register_agent('single-worker-direct')
    class SingleWorkerDirect(CentralizedMultiAgentSystem):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lead_agent._coordinate_and_run_subagents = MethodType(
                serial_round, self.lead_agent)
