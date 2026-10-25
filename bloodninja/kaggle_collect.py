"""Sanitized paired task-score export for Kaggle and other external runners.

No raw prompts, responses, inventory details, or tool traces are exported.
Only scored tasks with completed instance_save.yaml count as observations.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

ARMS = {
    "single-agent": "single",
    "multi-agent-centralized": "multi",
    "single-worker-queued": "queue",
    "single-worker-direct": "direct",
}
COLUMNS = ("task_id", "arm", "score", "rubric_score", "num_steps", "run_id")


def export_scores(root: Path, out: Path) -> list[dict]:
    import yaml
    result = []
    seen = set()
    for config_path in sorted(root.glob("**/run_config.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("agent"), dict):
            continue
        arm = ARMS.get(config["agent"].get("name"))
        if arm is None:
            continue
        run_dir = config_path.parent
        run_id = hashlib.sha256(str(run_dir).encode()).hexdigest()[:16]
        for instance_path in sorted((run_dir / "instance_runs").glob("*/instance_save.yaml")):
            obj = yaml.safe_load(instance_path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict) or not isinstance(obj.get("metrics"), dict):
                raise ValueError(f"Unparseable/missing metric record: {instance_path}")
            inp = obj.get("inp")
            if not isinstance(inp, dict):
                raise ValueError(f"Missing task identity: {instance_path}")
            metrics = obj["metrics"]
            # Make PlanCraft inventory/target identity stable across architecture arms,
            # without exporting inventory, text, or reference answers.
            normalized = json.dumps(inp, sort_keys=True, separators=(",", ":"), default=str)
            task_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            outcome = metrics.get("is_correct", metrics.get("success"))
            if type(outcome) is bool:
                score = int(outcome)
            elif type(outcome) is int and outcome in (0, 1):
                score = outcome
            else:
                raise ValueError(f"Missing/invalid benchmark binary score in {instance_path}")
            key = (run_id, task_id, arm)
            if key in seen:
                raise ValueError(f"Duplicated task score: {key}")
            seen.add(key)
            result.append({
                "task_id": task_id, "arm": arm, "score": score,
                "rubric_score": metrics.get("rubric_score", ""),
                "num_steps": metrics.get("num_steps", ""), "run_id": run_id,
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(result)
    if not result:
        raise ValueError("No valid graded task records were available")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = export_scores(args.root, args.out)
    print(f"Exported {len(rows)} scored task rows without raw prompts or responses")
