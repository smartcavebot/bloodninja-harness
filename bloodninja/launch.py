"""Bootstrap the pinned upstream runner with the Q-arm registered.

Run inside `upstream/agent-scaling` with `uv run --no-sync python ../../bloodninja/launch.py ...`.
This keeps upstream sources and scoring code immutable.
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    upstream = here / 'upstream' / 'agent-scaling'
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(upstream))
    if not (upstream / 'scripts' / 'run_experiment.py').is_file():
        raise RuntimeError('Missing submodule; git submodule update --init --recursive')
    from bloodninja.adapter import register_queued_agent
    from bloodninja.grading import install_strict_grading
    register_queued_agent()
    install_strict_grading()
    # Extend the upstream Hydra configuration at runtime without modifying Git
    # tracked source. Keep every upstream parameter and only change agent.name.
    original = (upstream / 'run_conf' / 'agent' / 'multi-agent-centralized.yaml').read_text()
    if 'name: multi-agent-centralized' not in original:
        raise RuntimeError('Upstream configuration changed: adapter requires review')
    modified = original.replace('name: multi-agent-centralized',
                                'name: single-worker-queued', 1)
    (upstream / 'run_conf' / 'agent' / 'single-worker-queued.yaml').write_text(modified)
    # Avoid changing the original benchmark entrypoint (Hydra, eval, metrics).
    sys.argv[0] = str(upstream / 'scripts' / 'run_experiment.py')
    os.chdir(upstream)
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
