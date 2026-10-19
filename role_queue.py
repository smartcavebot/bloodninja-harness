"""Reference queue-vs-direct role execution; standard-library only.

This is a transport/role-isolation testbed, NOT a Finance-Agent benchmark run.
Replace ExampleModel with the original agent-scaling LLM/tool adapter for the study.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
import json
import sqlite3
import uuid

ROLES = {
    "coordinator": "Decompose the research problem, then synthesize authorized evidence.",
    "filings": "Analyze filings only. State uncertainty and cite source evidence.",
    "operations": "Analyze operational impacts. State uncertainty and cite source evidence.",
    "markets": "Analyze market and macro factors. State uncertainty and cite source evidence.",
}
WORKERS = ("filings", "operations", "markets")

class Model(Protocol):
    def respond(self, role: str, system: str, history: list[dict], message: str) -> str: ...

class ExampleModel:
    """Deterministic offline test model. Its outputs are NOT research results."""
    def respond(self, role, system, history, message):
        return f"<{role}|prior={len(history)}|input={message[:70]}>"

@dataclass
class Trace:
    invocations: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

class RoleRuntime:
    """The *only* model-backed invocation point; each role has its OWN context."""
    def __init__(self, model: Model, trace: Trace, shared_context=False):
        self.model, self.trace = model, trace
        self.shared_context = shared_context
        self.histories = {r: [] for r in ROLES}
        self.global_history = []

    def invoke(self, role: str, message: str) -> str:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        history = self.global_history if self.shared_context else self.histories[role]
        # Snapshot prevents model-side mutation from polluting the record.
        prior = [dict(m) for m in history]
        self.trace.invocations.append({"role": role, "message": message, "prior": prior})
        output = self.model.respond(role, ROLES[role], prior, message)
        history.extend([{"speaker": "user", "text": message},
                        {"speaker": "assistant", "text": output}])
        return output

class SQLiteBroker:
    """Persisted, idempotently-enqueued, single-consumer FIFO.

    Prototype has no retries/lease recovery. Not production-safe.
    """
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS messages
            (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
             role TEXT NOT NULL, payload TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0)""")
        self.db.commit()

    def publish(self, role: str, payload: str, msg_id: str | None = None) -> str:
        msg_id = msg_id or str(uuid.uuid4())
        self.db.execute("INSERT OR IGNORE INTO messages(id, role, payload) VALUES(?,?,?)",
                        (msg_id, role, json.dumps(payload)))
        self.db.commit()
        return msg_id

    def consume(self):
        self.db.execute("BEGIN IMMEDIATE")
        row = self.db.execute("SELECT seq,id,role,payload FROM messages WHERE consumed=0 ORDER BY seq LIMIT 1").fetchone()
        if row:
            self.db.execute("UPDATE messages SET consumed=1 WHERE seq=?", (row[0],))
        self.db.commit()
        return (row[1], row[2], json.loads(row[3])) if row else None

    def close(self):
        self.db.close()

class SingleConsumer:
    """Exactly one consumer drives ONE RoleRuntime over all logical roles."""
    def __init__(self, model: Model, trace: Trace, shared_context=False):
        self.runtime = RoleRuntime(model, trace, shared_context)
        self.broker = SQLiteBroker()
        self.trace = trace

    def dispatch(self, role: str, payload: str) -> str:
        mid = self.broker.publish(role, payload)
        self.trace.messages.append({"event": "publish", "id": mid, "role": role})
        record = self.broker.consume()
        assert record and record[0] == mid, "FIFO/single-consumer invariant"
        self.trace.messages.append({"event": "consume", "id": mid, "role": role})
        return self.runtime.invoke(record[1], record[2])

def run_task(question: str, model: Model | None = None, mode: str = "queue") -> tuple[str, Trace]:
    """Same fixed dependency graph, prompt templates and work allocation for every arm.

    mode=direct: four logical runtimes, separate histories, no broker.
    mode=queue: one runtime, private role histories, persisted broker.
    mode=leaky: queue + illegally shared context (negative control).
    mode=single: one naive baseline role without worker decomposition.
    """
    model = model or ExampleModel()
    trace = Trace()
    if mode not in {"direct", "queue", "leaky", "single"}:
        raise ValueError(mode)
    if mode == "single":
        runtime = RoleRuntime(model, trace)
        return runtime.invoke("coordinator", "Answer directly: " + question), trace

    if mode == "direct":
        runtimes = {r: RoleRuntime(model, trace) for r in ROLES}
        call = lambda r, msg: runtimes[r].invoke(r, msg)
    else:
        consumer = SingleConsumer(model, trace, shared_context=mode == "leaky")
        call = consumer.dispatch

    # This simple topology is fixed, NOT the original benchmark's tool-using algorithm.
    plan = call("coordinator", "Plan analysis for: " + question)
    results = {}
    for role in WORKERS:
        results[role] = call(role, f"Question: {question}\nAuthorized coordinator instruction: {plan}")
    evidence = "\n".join(f"{role}: {results[role]}" for role in WORKERS)
    answer = call("coordinator", f"Synthesize results for {question}\nEvidence:\n{evidence}")
    if mode != "direct":
        consumer.broker.close()
    return answer, trace

if __name__ == "__main__":
    for arm in ("single", "direct", "queue", "leaky"):
        output, trace = run_task("What is the financial impact of a merger?", mode=arm)
        print(json.dumps({"arm": arm, "answer": output,
                          "calls": len(trace.invocations),
                          "role_context_lengths": [len(x["prior"]) for x in trace.invocations]}))
