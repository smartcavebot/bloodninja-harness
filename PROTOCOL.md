# Protocol / falsification criteria

## Hypotheses

- **H1 (gain capture):** `R = (Q - S) / (M - S) >= 0.90`, conditional on the reproduced original advantage `M > S`. `S` = published design's single-agent baseline; `M` = published design's homogeneous centralized agents; `Q` = one-worker, role-private queue execution. `R` is unstable when `M-S` is small: report paired raw differences alongside it.
- **H2 (noninferiority):** `Q - M > -0.05` absolute rubric score (preliminary proposed margin, requires domain justification and adequate statistical power). A positive point estimate by itself is not proof of noninferiority.
- **H3 (role isolation):** for a given role at a given workflow node, `visible_prompt(Q)` is byte-for-byte identical to `visible_prompt(M)` and the tool visibility is identical, apart from irrelevant transport metadata. Hash the canonical prompt and tool-permission envelopes.
- **H4 (contamination):** deliberately sharing global history may change outcomes; effect sign is an empirical question.

## Control matrix

| Arm | Logical worker roles | Executing workers | Role histories | Broker | Purpose |
|---|---|---|---|---|---|
| S | 1 | 1 | Shared | No | Paper's single baseline |
| M | 3 + coordinator | 4 logical agents | Private | No/direct | Reproduce strongest multi-agent gain |
| Q | 3 + coordinator | **1** | **Private** | Yes | Primary proposed architecture |
| L | 3 + coordinator | **1** | Global/shared | Yes | Contamination/OneFlow-adjacent control |
| B | 3 + coordinator | Multiple | Private | Yes | Broker overhead control, optional |
| E | 3 + coordinator | 1 | Private | No/direct, serialized | Serialization control, optional |

All three core interventions should preserve model weights/version, role-specific system prompts, source data, visible tools, routing/aggregation policy, allowed turns, and maximum output lengths. Do not make a model that receives everyone's intermediate context masquerade as "isolated roles".

## What 'one model instance' means

Four distinct claims must not be merged:

1. **One model identity/weights:** Many "multi-agent" systems already have this property; all calls route to the same model endpoint.
2. **One inference process/server:** Multiple request contexts may share the same GPU/server without sharing attention state.
3. **One serialized consumer:** At most one active role execution at a time. Parallel throughput/latency advantages are forfeited, but quality need not change.
4. **One uninterrupted conversation/KV cache:** Previous roles' dialogue may become visible to later roles. This is NOT role-private isolation. Cache reuse and strict isolation are sometimes in tension.

For a stateless model, if conditional inputs, routing and independent sampler randomness match, direct multi-agent and one-consumer private-role execution are distributionally equivalent. Deviations identify model-serving nondeterminism, hidden agent state, tool nonstationarity, context leak, orchestration differences, or changed compute budgets. Different *model families* genuinely alter conditional behavior and cannot be emulated by a same-weights consumer merely by changing roles.

## Operational controls

- Use a causal event log with `task_id`, `role_id`, `turn_id`, `parent_event_ids`, `transport_mode`, `model_snapshot`, `decoding_seed` (where supported), `prompt_sha256`, `visible_evidence_sha256`, `tools_sha256`, `tokens`, `elapsed_ms`, and output hash.
- Snapshot web search results, filing data, tool error conditions and any mutable state, or record source timestamps and replay each observation. The paper's Finance-Agent tools include live web search; without snapshots, concurrent agent and serial queue conditions see potentially different environments.
- Same sampling temperature, seeds where supported, provider prompt caching policy, temperature 0 not assumed fully deterministic. Rotate architecture execution order to reduce drift.
- Record two cost regimes: *equal budget* (same aggregate number of model calls/output tokens/tool operations) and *natural architecture cost* (each architecture runs to its allowed stopping criterion).
- Pre-register the exact benchmark items, judge/rubric, model, max budget, and scoring aggregation. Never optimize prompts after test-set inspection.
- Report paired task-level differences with hierarchical/task-cluster bootstrap. Repeat across more than one model family only after first controlled replication.

## Evidence and references

1. Y. Kim et al. (2026), *Capable language models can outgrow the benefits of collaboration*, Nature Machine Intelligence, DOI: 10.1038/s42256-026-01268-y. arXiv:2512.08296. https://github.com/ybkim95/agent-scaling
2. J. Xu et al. (2026), *Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline*. arXiv:2601.12307. Note: GPT-4o-mini cost savings in the paper use **idealized token/KV-cache cost estimates**, not provider-billed measurements; Qwen open-weight experiments separately measure latency/throughput.
3. Y. Yang et al. (2025), *Revisiting Multi-Agent Debate as Test-Time Scaling*, arXiv:2505.22960. Notes weak advantages over strong single-agent scaling for simple mathematics problems.

## Current execution status

Implemented and verified locally: SQLite FIFO transport idempotence; same deterministic role-local calls under direct vs queue; negative-control role-history contamination; 5/5 unit tests. The upstream Q adapter now registers with the original Finance-Agent evaluator, preserving the upstream prompts, per-role tools, per-role conversation histories and feedback schedule. One SQLite-backed consumer serializes work. **No billed model API results have been produced yet**, and upstream model/tool compatibility has not yet been verified on a live task. Local tests cover transport and integration scheduling contracts. The original implementation already shares the core model object; do not claim model-identity novelty. There is a known wall-clock budget disparity (300s concurrent round timeout vs non-preemptible queue drain). Next: clean integration import, 1-task Q smoke, matched S/M/Q task runs, dataset source snapshots, and paired trials.

## Paired metrics artifact

`bloodninja/export_metrics.py` reads upstream `instance_runs/*/instance_save.yaml` plus matching `run_config.yaml` and extracts only `task_id`, arm, binary `is_correct` (primary published accuracy), optional rubric score, grading method and experiment path. It deliberately excludes raw questions, references, answers, trajectories, and model responses. The GitHub Actions manual experiment exports `paired-task-metrics.csv`, along with aggregate metrics. If all S/M/Q arms finish, it also emits `paired-bootstrap.txt`. This is only a valid comparison after checking task_id intersection, identical grader/model/settings, and the run's per-arm failure count.

## Data input pin

The benchmark workflow checks out the public Vals Finance-Agent repository at exactly `8ba65f81ab759a8e0d44e72aabc5a47cf839d563` (2026-07-21), then passes that directory to the unchanged upstream normalization script. Do not let the setup script silently fetch current `main`, or historical comparisons become non-repeatable. Web search and grader-model behavior still require additional snapshot controls; pinning the task CSV does not solve those sources of drift.
