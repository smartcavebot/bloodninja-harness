"""Kaggle-hosted Qwen experiment runner.

Everything executes INSIDE a Kaggle GPU notebook; GitHub Actions only submits
and fetches the batch. Public experiment source is cloned at a pinned commit.
Raw upstream prompts/traces stay in ephemeral Kaggle scratch, not exported.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

# Replaced by prepare.py before submission. Never place secrets in config.
CONFIG = json.loads('__BLOODNINJA_CONFIG__')
MODEL = "Qwen/Qwen3-8B-AWQ"
MODEL_REVISION = "c664f5e955c33186a8b7a5cf0cbc3b220e4afdd4"
ROOT = Path("/kaggle/temp/bloodninja-harness")
OUT = Path("/kaggle/working/bloodninja-results")
PORT = 8000


def run(*argv, cwd=None, env=None):
    print("+", " ".join(str(arg) for arg in argv), flush=True)
    subprocess.run([str(a) for a in argv], cwd=cwd, env=env, check=True)


def acquire_tavily_if_needed():
    if CONFIG["dataset"] != "finance-agent":
        return
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["TAVILY_API_KEY"] = UserSecretsClient().get_secret("TAVILY_API_KEY")
    except Exception as exc:
        raise RuntimeError("Finance-Agent requires a Kaggle notebook secret named TAVILY_API_KEY") from exc
    if not os.environ["TAVILY_API_KEY"]:
        raise RuntimeError("Kaggle secret TAVILY_API_KEY was empty")


def wait_for_server(proc, seconds=480):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited before model was ready: exit={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3) as result:
                models = json.load(result)
            if models.get("data"):
                print("vLLM ready with", [m.get("id") for m in models["data"]], flush=True)
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("vLLM did not become ready before startup deadline")


def check_tools():
    # A real tool-call roundtrip must work before consuming research GPU quota.
    request = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Call probe with n=1. Do not provide a text answer."}],
        "tools": [{"type": "function", "function": {"name": "probe",
                   "description": "Validates function calls",
                   "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}}],
        "tool_choice": "required", "temperature": 0, "max_tokens": 256,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(request).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-only"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    calls = payload["choices"][0]["message"].get("tool_calls") or []
    if not calls or calls[0].get("function", {}).get("name") != "probe":
        raise RuntimeError("Tool calling smoke test failed; refusing to grade malformed model responses")
    print("Local model function-call smoke passed", flush=True)


def execute_arm(repo_env, arm, target):
    runner = ROOT / "upstream/agent-scaling"
    args = [
        str(runner / ".venv/bin/python"), "../../bloodninja/launch.py",
        f"agent={target}", f"dataset={CONFIG['dataset']}",
        f"llm.model=openai/{MODEL}",
        f"max_instances={CONFIG['max_instances']}",
        "debug=false", "num_workers=1",
        "llm.params.temperature=0.0", "use_disk_cache=false", "log_langfuse=false"
    ]
    if CONFIG["dataset"] == "finance-agent":
        args.append(f"dataset.eval_llm.model=openai/{MODEL}")
    print("START", arm, "max_instances", CONFIG["max_instances"], flush=True)
    run(*args, cwd=runner, env=repo_env)
    found = list((runner / "exp_outputs" / CONFIG["dataset"].replace("-", "_") / target).glob("**/instance_save.yaml"))
    # Finance-Agent writes finance_agent; PlanCraft writes plancraft-test.
    if not found:
        found = list((runner / "exp_outputs").glob(f"**/{target}/**/instance_save.yaml"))
    if len(found) != CONFIG["max_instances"]:
        raise RuntimeError(f"INVALID {arm}: {len(found)}/{CONFIG['max_instances']} task score files")
    print("FINISH", arm, "graded tasks", len(found), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = Path("/kaggle/temp")
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(scratch / "hf-cache")
    os.environ["UV_CACHE_DIR"] = str(scratch / "uv-cache")
    os.environ["PIP_CACHE_DIR"] = str(scratch / "pip-cache")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not shutil.which("nvidia-smi"):
        raise RuntimeError("This Kaggle run has no NVIDIA GPU; enable T4 x2")
    run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")
    acquire_tavily_if_needed()
    run("git", "clone", "--recurse-submodules",
        "https://github.com/smartcavebot/bloodninja-harness.git", str(ROOT))
    run("git", "checkout", "--detach", CONFIG["sha"], cwd=ROOT)
    run("git", "submodule", "update", "--init", "--recursive", cwd=ROOT)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != CONFIG["sha"]:
        raise RuntimeError("Failed to check out the exact intended code version")

    # Keep inference and benchmark dependencies in independent Python environments.
    try:
        import vllm  # noqa
    except ImportError:
        run(sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "vllm==0.11.2")
    vllm_exe = shutil.which("vllm")
    if not vllm_exe:
        raise RuntimeError("vLLM CLI unavailable after installation")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        run(sys.executable, "-m", "pip", "install", "uv")
        uv_bin = shutil.which("uv")
    if not uv_bin:
        raise RuntimeError("uv installation missing")
    runner = ROOT / "upstream/agent-scaling"
    run(uv_bin, "sync", "--prerelease=allow", cwd=runner)

    if CONFIG["dataset"] == "finance-agent":
        finance_src = runner / "third_party/finance-agent"
        run("git", "clone", "https://github.com/vals-ai/finance-agent.git", str(finance_src))
        run("git", "checkout", "8ba65f81ab759a8e0d44e72aabc5a47cf839d563", cwd=finance_src)
        run("bash", "scripts/setup_finance_agent.sh", cwd=runner)

    llm_env = dict(os.environ)
    llm_env["CUDA_VISIBLE_DEVICES"] = "0"
    log_file = scratch / "bloodninja-vllm.log"
    command = [
        vllm_exe, "serve", MODEL, "--revision", MODEL_REVISION,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--dtype", "half", "--quantization", "awq",
        "--max-model-len", "16384", "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.90",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        "--disable-log-requests",
    ]
    print("Starting local GPU inference service; model weights ~6.1GB", flush=True)
    with log_file.open("w") as log:
        server = subprocess.Popen(command, env=llm_env, stdout=log, stderr=subprocess.STDOUT)
        try:
            try:
                wait_for_server(server)
            except BaseException:
                print("vLLM startup diagnostics:\n", log_file.read_text()[-5000:], flush=True)
                raise
            check_tools()
            client_env = dict(os.environ)
            client_env["OPENAI_API_KEY"] = "local-only"
            client_env["OPENAI_API_BASE"] = f"http://127.0.0.1:{PORT}/v1"
            client_env["OPENAI_BASE_URL"] = client_env["OPENAI_API_BASE"]
            targets = {"single": "single-agent", "multi": "multi-agent-centralized",
                       "queue": "single-worker-queued", "direct": "single-worker-direct"}
            requested = CONFIG["arm"]
            selected = ("single", "multi", "queue", "direct") if requested == "all4" else (
                ("single", "multi", "queue") if requested == "all" else (requested,))
            # A failed arm stops the run. No missing scores are silently treated as zero.
            for arm in selected:
                execute_arm(client_env, arm, targets[arm])
            run(str(runner / ".venv/bin/python"), "../../bloodninja/kaggle_collect.py",
                "--root", str(runner / "exp_outputs"),
                "--out", str(OUT / "task-scores.csv"), cwd=runner)
            provenance = {**CONFIG, "model": MODEL, "model_revision": MODEL_REVISION,
                          "max_model_len": 16384, "quantization": "AWQ", "runtime": "vLLM"}
            (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
            print("RESEARCH_OUTPUT_READY", OUT, flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
