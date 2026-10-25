"""Construct a Kaggle private GPU script submission from tracked sources.

The kernel upload API accepts one code file plus metadata. We bake *only*
non-sensitive run configuration into the script; secrets remain Kaggle Secrets.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

ARMS = {"single", "multi", "queue", "direct", "all", "all4"}
DATASETS = {"plancraft-test", "finance-agent"}


def prepare(root: Path, user: str, sha: str, dataset: str, arm: str,
            instances: int, destination: Path) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,39}", user):
        raise ValueError("Kaggle username must be 2-40 letters/digits/underscores/hyphens")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Commit must be pinned to a full 40-character SHA")
    if arm not in ARMS or dataset not in DATASETS:
        raise ValueError("Unsupported arm or dataset")
    if instances < 1 or instances > 50:
        raise ValueError("Instance count must be between 1 and 50")
    config = {"sha": sha, "dataset": dataset, "arm": arm, "max_instances": instances}
    script = (root / "kaggle/runner.py").read_text(encoding="utf-8")
    marker = "'__BLOODNINJA_CONFIG__'"
    if script.count(marker) != 1:
        raise ValueError("Kaggle runner configuration marker missing or duplicated")
    script = script.replace(marker, repr(json.dumps(config, sort_keys=True)), 1)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "runner.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": f"{user.lower()}/bloodninja-control",
        "title": "Bloodninja Control",
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (destination / "config-public.json").write_text(
        json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--user", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--dataset", choices=sorted(DATASETS), default="plancraft-test")
    p.add_argument("--arm", choices=sorted(ARMS), default="single")
    p.add_argument("--instances", type=int, default=1)
    p.add_argument("--destination", type=Path, default=Path("kaggle/submission"))
    a = p.parse_args()
    print(prepare(a.root, a.user, a.sha, a.dataset, a.arm,
                  a.instances, a.destination)["id"])
