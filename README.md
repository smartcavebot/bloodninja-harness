# bloodninja-harness

Experimental agent harness testing whether a **single LLM executing isolated roles through a message queue** can reproduce gains attributed to distinct multi-agent roles.

## Status

- The role-private SQLite transport demonstrator and paired bootstrap statistics are implemented and covered by five passing deterministic unit tests.
- Upstream `ybkim95/agent-scaling` is tracked as an actual Git submodule at **v2.2.3 / `8673d6a8d5a764a07b08e4283492380d80257223`**.
- Actions CI executes the offline transport tests. `baseline.yml` can manually reproduce the *published single-agent and centralized multi-agent controls* on a selected sample of Finance-Agent tasks, with user-supplied API secrets. **Q is now integrated as an additional upstream architecture** (`single-worker-queued`) through a launch-time registration adapter. The local `role_queue.py` remains a fake deterministic model and is not a scientific scoring result.

## Goals

Compare six architectures while preserving the published evaluator, model, prompts, role-specific views, allowed tools, communication topology, generation budgets and scoring:

| Arm | Description | Present state |
|---|---|---|
| S | Original single agent | Upstream runner |
| M | Original centralized multi-agent | Upstream runner |
| Q | One queued producer/consumer with private per-role histories | **Upstream Finance-Agent runner adapter + offline contract tests; billed API execution not yet verified** |
| L | Queue with globally shared history (negative control) | Transport proof only |
| B | Broker with multiple consumers (broker control) | Planned |
| E | One direct serialized worker without broker (serialization control) | Planned |

Primary statistic: `R=(Q-S)/(M-S)`, defined only for a positive replicated advantage `M>S` and accompanied by paired uncertainty estimates. See [PROTOCOL.md](PROTOCOL.md).

## Obtain the upstream submodule

```bash
git clone --recurse-submodules YOUR_GITHUB_REPO_URL
# or, on an existing clone:
git submodule update --init --recursive
git -C upstream/agent-scaling rev-parse HEAD
# expected: 8673d6a8d5a764a07b08e4283492380d80257223
```

## Offline verification

```bash
python -m unittest discover -s tests -v
python role_queue.py
```

## GitHub Actions

- **CI**: runs offline tests on push/PR; performs no billed LLM calls.
- **Finance-Agent experiment**: `Actions > Finance-Agent experiment (manual billed runs) > Run workflow`. Choose one of S, M, Q, or `all`; defaults to **single-agent on just one task**. Requires `OPENAI_API_KEY` and `TAVILY_API_KEY` GitHub Actions secrets. A full 50-task comparison consumes API credits and requires explicit selection. The workflow uploads aggregate and **paired task-level accuracy metrics**, optionally with bootstrap estimates when all three arms finish. It does not upload raw model transcripts or queue databases. Parallel runner instance count fixed at 1. All arms use the same model ID and task cutoff.

### What the integration changes

`bloodninja/adapter.py` replaces only the upstream lead→worker dispatch method. The lead generates all original role-specific instructions before jobs run. One consumer processes all queued jobs in SQLite FIFO order, calling the **original** `WorkerSubagent.process_orchestrator_message` with its untouched prompt construction, tool environment, private conversation history, and result type. `bloodninja/launch.py` registers this as `agent=single-worker-queued` while using the upstream evaluation runner unchanged. The original system **already shares one underlying model object across its logical workers**; the difference being tested is durable queued serialization versus concurrent handoff, not different model weights.

**Limitations:** Paid model runs have not yet verified output parity; upstream may be nondeterministic. Serialization removes parallelism and changes wall-clock budgets; unlike the original 300-second per-round concurrent timeout, the queue consumer does not abruptly terminate toolful jobs because cancelling a Python thread would leave non-idempotent side effects running. The durable queue records instructions and state but deliberately does not automatically replay failed running jobs. A consumer still executes each upstream worker object (rather than replacing all role executors with one mutable agent object); that is a defensible first experimental isolation, not proof of every interpretation of one-agent execution.

## Publishing an offline-prepared repository

This distribution includes a Git bundle (`bloodninja-harness.bundle`) that preserves a **genuine submodule gitlink** and a pinned upstream commit. If a GitHub repository was not created when the kit was prepared:

```bash
git clone bloodninja-harness.bundle bloodninja-harness
cd bloodninja-harness
git remote set-url origin YOUR_GITHUB_REPO_URL
git push -u origin main
git submodule update --init --recursive
```

No unverified study results are asserted here.
