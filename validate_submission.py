from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def check_openenv_yaml() -> bool:
    path = REPO_ROOT / "openenv.yaml"
    if not path.exists():
        _fail("openenv.yaml missing")
        return False
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        if "id: easy" in text and "id: medium" in text and "id: hard" in text:
            _ok("openenv.yaml present (install pyyaml for strict YAML parse)")
            return True
        _fail("openenv.yaml: expected easy/medium/hard task ids")
        return False

    data = yaml.safe_load(text)
    tasks = data.get("tasks") or []
    if len(tasks) < 3:
        _fail(f"openenv.yaml must define >= 3 tasks, got {len(tasks)}")
        return False
    _ok(f"openenv.yaml: {len(tasks)} tasks, name={data.get('name')!r}")
    return True


def check_typed_models() -> bool:
    try:
        from env.models import SREAction, SREObservation, SREReward  # noqa: F401

        _ = SREAction.model_json_schema()
        _ = SREObservation.model_json_schema()
        _ = SREReward.model_json_schema()
    except Exception as exc:
        _fail(f"typed models import/schema: {exc}")
        return False
    _ok("Pydantic models (SREAction, SREObservation, SREReward) and JSON schemas")
    return True


def check_http_endpoints() -> bool:
    try:
        from fastapi.testclient import TestClient

        from api.server import app
    except Exception as exc:
        _fail(f"FastAPI TestClient import: {exc}")
        return False

    client = TestClient(app)
    try:
        r = client.get("/health")
        if r.status_code != 200 or r.json().get("status") != "healthy":
            _fail(f"/health: {r.status_code} {r.text}")
            return False

        m = client.get("/metadata")
        if m.status_code != 200 or "name" not in m.json():
            _fail("/metadata missing name")
            return False

        sc = client.get("/schema")
        if sc.status_code != 200:
            _fail("/schema failed")
            return False
        body = sc.json()
        for key in ("action", "observation", "state"):
            if not isinstance(body.get(key), dict):
                _fail(f"/schema missing {key} object")
                return False

        for tid in ("easy", "medium", "hard"):
            rr = client.post("/reset", json={"task_id": tid})
            if rr.status_code != 200:
                _fail(f"POST /reset {tid}: {rr.status_code}")
                return False
            sr = client.post(
                "/step",
                json={"action": {"action_type": "check_metrics", "service": "api-gateway"}},
            )
            if sr.status_code != 200:
                _fail(f"POST /step: {sr.status_code}")
                return False
            st = client.get("/state")
            if st.status_code != 200 or "service_metrics" not in st.json():
                _fail("GET /state invalid")
                return False
    except Exception as exc:
        _fail(f"HTTP endpoint check: {exc}")
        return False

    _ok("GET /health, /metadata, /schema; POST /reset, /step; GET /state (all tasks)")
    return True


def check_graders_all_tasks() -> bool:
    from env.environment import EpisodeStep
    from env.models import SREAction
    from env.tasks.base import load_scenario
    from graders.grader import TaskGrader

    for task_id in ("easy", "medium", "hard"):
        task = load_scenario(task_id)
        gt = task.ground_truth
        hist = [
            EpisodeStep(0, SREAction(action_type="check_metrics", service=gt.root_cause_service)),
            EpisodeStep(1, SREAction(action_type=gt.correct_fix_action, service=gt.root_cause_service)),
        ]
        res = SREAction(
            action_type="resolve",
            service=gt.root_cause_service,
            root_cause_description=gt.root_cause_description,
            fix_applied=gt.correct_fix_action,
        )
        grader = TaskGrader(task)
        score = grader.grade_episode(hist, res)
        if not (0.0 <= score <= 1.0):
            _fail(f"grader {task_id}: score {score} out of range")
            return False
        print(f"      task={task_id} grader_score={score:.4f}")

    _ok("3 tasks: TaskGrader scores in [0.0, 1.0]")
    return True


def check_pytest() -> bool:
    exe = shutil.which("pytest")
    if not exe:
        _fail("pytest not on PATH")
        return False
    r = subprocess.run(
        [exe, str(REPO_ROOT / "tests"), "-q", "--tb=short"],
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        _fail("pytest failed")
        return False
    _ok("pytest tests/")
    return True


def check_docker_build() -> bool:
    if not shutil.which("docker"):
        _fail("docker not installed (skip with --skip-docker)")
        return False
    r = subprocess.run(
        ["docker", "build", "-t", "sre-incident-env-validation", "."],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        _fail(f"docker build failed:\n{r.stderr or r.stdout}")
        return False
    _ok("docker build -t sre-incident-env-validation .")
    return True


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def check_inference_script() -> bool:
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subproc_environ = os.environ.copy()
    subproc_environ["SRE_ENV_URL"] = base
    subproc_environ["ENV_BASE_URL"] = base
    subproc_environ.pop("HF_TOKEN", None)
    subproc_environ.pop("OPENAI_API_KEY", None)

    try:
        for _ in range(50):
            try:
                import urllib.request

                urllib.request.urlopen(f"{base}/health", timeout=0.5).read()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _fail("uvicorn did not become ready")
            return False

        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "inference.py")],
            cwd=str(REPO_ROOT),
            env=subproc_environ,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode != 0:
            _fail(f"inference.py failed:\n{r.stderr or r.stdout}")
            return False
        _ok("inference.py completed (scripted baseline, no HF_TOKEN)")
        print(r.stdout)
        return True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def check_openenv_cli() -> bool:
    exe = shutil.which("openenv")
    if not exe:
        print("[--] openenv CLI not found; skip `openenv validate .`")
        return True
    r = subprocess.run(
        [exe, "validate", str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        _fail(f"openenv validate failed:\n{r.stderr or r.stdout}")
        return False
    _ok("openenv validate .")
    return True


def check_hf_space(url: str) -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(url.rstrip("/") + "/health", method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            if code != 200:
                _fail(f"Space /health status {code}")
                return False
        payload = json.dumps({"task_id": "easy"}).encode()
        req2 = urllib.request.Request(
            url.rstrip("/") + "/reset",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req2, timeout=30) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            if code != 200:
                _fail(f"Space POST /reset status {code}")
                return False
            body = json.loads(resp.read().decode())
            if body.get("observation", {}).get("step") != 0:
                _fail("Space /reset did not return step=0")
                return False
    except Exception as exc:
        _fail(f"HF Space check: {exc}")
        return False
    _ok(f"HF Space reachable: {url}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-submission validation")
    ap.add_argument("--skip-docker", action="store_true")
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument(
        "--space-url",
        default=os.environ.get("HF_SPACE_URL"),
        help="Deployed Space base URL (or set HF_SPACE_URL)",
    )
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    checks: List[Tuple[str, bool]] = []

    print("=== Pre-submission validation ===\n")

    def add(name: str, ok: bool) -> None:
        checks.append((name, ok))

    add("openenv.yaml", check_openenv_yaml())
    add("typed models", check_typed_models())
    add("HTTP API (reset/step/state)", check_http_endpoints())
    add("graders 3 tasks", check_graders_all_tasks())
    add("pytest", check_pytest())
    add("openenv validate (local)", check_openenv_cli())

    if not args.skip_docker:
        add("docker build", check_docker_build())
    else:
        print("[--] skipped docker build")

    if not args.skip_inference:
        add("inference.py", check_inference_script())
    else:
        print("[--] skipped inference.py")

    if args.space_url:
        add("HF Space URL", check_hf_space(args.space_url))
    else:
        print("[--] HF_SPACE_URL / --space-url not set; skipped remote Space ping")

    failed = [n for n, ok in checks if not ok]
    print()
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("=== All checks passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
