# bloodninja-harness

Experimental agent harness testing whether a **single LLM executing isolated roles through a message queue** can reproduce gains attributed to distinct multi-agent roles.

## Status

- The role-private SQLite transport demonstrator and paired bootstrap statistics are implemented and covered by five passing deterministic unit tests.
- Upstream `ybkim95/agent-scaling` is tracked as an actual Git submodule at **v2.2.3 / `8673d6a8d5a764a07b08e4283492380d80257223`**.
- Actions CI executes the offline transport tests. `baseline.yml` can manually reproduce the *published single-agent and centralized multi-agent controls* on a selected sample of Finance-Agent tasks, with user-supplied API secrets. **There is no executable Q-vs-M live benchmark integration yet.** The local `role_queue.py` uses a fake deterministic model and is not a scientific scoring result.

## Goals

Compare six architectures while preserving the published evaluator, model, prompts, role-specific views, allowed tools, communication topology, generation budgets and scoring:

| Arm | Description | Present state |
|---|---|---|
| S | Original single agent | Upstream runner |
| M | Original centralized multi-agent | Upstream runner |
| Q | One queued producer/consumer with private per-role histories | Transport proof only |
| L | Queue with globally shared history (negative control) | Transport proof only |
| B | Broker with multiple consumers (broker control) | Planned |
| E | One direct serialized worker without broker (serialization control) | Planned |

Primary statistic: `R=(Q-S)/(M-S)`, defined only for a positive replicated advantage `M>S` and accompanied by paired uncertainty estimates. See [PROTOCOL.md](PROTOCOL.md).

## Obtain the upstream submodule

```bash
git clone --recurse-submodules https://github.com/smartcavebot/bloodninja-harness.git
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
- **Finance-Agent baseline**: `Actions > Finance-Agent baselines > Run workflow` runs S/M in series by explicit manual trigger, with `OPENAI_API_KEY` and `TAVILY_API_KEY` stored as GitHub Actions secrets. Defaults to **one task per architecture** for smoke testing. A full 50-task comparison consumes API credits and requires explicit selection. The workflow uploads only collected aggregate metric JSON files; it does not upload raw model transcripts. Parallel worker count fixed at 1 to avoid run-order variability within each arm. Both arms get the same model ID and task cutoff.

Note: running S/M establishes the control. We must still integrate the queue adapter with the upstream architecture and evaluate Q under the same benchmark before comparing results.

## Source provenance

Original research: Yubin Kim et al., *Capable language models can outgrow the benefits of collaboration*, *Nature Machine Intelligence* (2026). Upstream code: https://github.com/ybkim95/agent-scaling. The pinned submodule identifies the exact reproducible code baseline.

No unverified study results are asserted here.
