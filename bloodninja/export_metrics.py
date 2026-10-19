"""Export Finance-Agent *task-level* metrics without copying raw responses.

Aggregated JSON alone cannot support the paired bootstrap hypothesis test.
This reads the upstream runner's instance_save.yaml files and removes all
prompts, answers and references from the exported research artifact.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

COLUMNS = ('task_id', 'arm', 'score', 'rubric_score', 'grading_method', 'experiment_dir')
ARMS = {'single-agent': 'single',
        'multi-agent-centralized': 'multi',
        'single-worker-queued': 'queue'}


def export_metrics(root: Path, out: Path) -> list[dict]:
    import yaml  # Installed by the pinned upstream pyproject, not needed for offline CI.

    rows: list[dict] = []
    observed: set[tuple[str, str, str]] = set()
    for conf in sorted(root.glob('**/run_config.yaml')):
        data = yaml.safe_load(conf.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            continue
        agent = data.get('agent') or {}
        arch = agent.get('name') if isinstance(agent, dict) else None
        arm = ARMS.get(arch)
        if arm is None:
            continue
        exp_dir = conf.parent
        for file in sorted((exp_dir / 'instance_runs').glob('*/instance_save.yaml')):
            obj = yaml.safe_load(file.read_text(encoding='utf-8'))
            if not isinstance(obj, dict):
                raise ValueError(f'Unrecognized instance structure: {file}')
            inp, metrics = obj.get('inp'), obj.get('metrics')
            if not isinstance(inp, dict) or not isinstance(metrics, dict):
                raise ValueError(f'Missing input/metrics in {file}')
            task_id = str(inp.get('task_id', '')).strip()
            if not task_id:
                raise ValueError(f'No canonical task_id in {file}')
            if 'is_correct' not in metrics:
                raise ValueError(f'No published binary accuracy metric in {file}')
            score = metrics['is_correct']
            if not isinstance(score, (bool, int)) or score not in (True, False, 0, 1):
                raise ValueError(f'Invalid binary accuracy in {file}: {score!r}')
            key = (arm, task_id, str(exp_dir))
            if key in observed:
                raise ValueError(f'Duplicate task within run: {key}')
            observed.add(key)
            rubric_score = metrics.get('rubric_score')
            if rubric_score is not None and not (0 <= float(rubric_score) <= 1):
                raise ValueError(f'Out-of-range rubric score in {file}')
            rows.append({
                'task_id': task_id,
                'arm': arm,
                'score': int(score),
                'rubric_score': '' if rubric_score is None else float(rubric_score),
                'grading_method': metrics.get('grading_method', ''),
                'experiment_dir': exp_dir.as_posix(),
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('upstream/agent-scaling/exp_outputs'))
    p.add_argument('--out', type=Path, default=Path('metrics/paired-task-metrics.csv'))
    args = p.parse_args()
    rows = export_metrics(args.root, args.out)
    print(f'Exported {len(rows)} task-level accuracy scores to {args.out}; no response text included')


if __name__ == '__main__':
    main()
