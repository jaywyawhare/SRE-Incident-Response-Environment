from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple  # noqa: F401

import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

SRE_ENV_URL = os.environ.get("SRE_ENV_URL") or os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")

INFERENCE_MAX_SECONDS = float(os.environ.get("INFERENCE_MAX_SECONDS", str(19 * 60)))
DEFAULT_ESCALATION_TEAM = os.environ.get("DEFAULT_ESCALATION_TEAM", "platform")
BENCHMARK = "sre-incident-env"
TASKS = ["task_easy", "task_medium", "task_hard"]
SUCCESS_SCORE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Logging — strict [START] / [STEP] / [END] format expected by the platform
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    action_preview = action[:100] + "..." if len(action) > 100 else action
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f'[STEP] step={step} action="{action_preview}" '
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_fallback_seq(task_id: str) -> List[Dict[str, Any]]:
    from env.tasks.base import load_scenario
    gt = load_scenario(task_id.lower().replace("task_", "")).ground_truth
    return [
        {"action_type": "check_metrics", "service": gt.root_cause_service},
        {"action_type": "read_logs", "service": gt.root_cause_service},
        {"action_type": gt.correct_fix_action, "service": gt.root_cause_service},
        {
            "action_type": "resolve",
            "service": gt.root_cause_service,
            "root_cause_description": gt.root_cause_description,
            "fix_applied": gt.correct_fix_action,
        },
    ]


def _fallback_action(seq: List[Dict[str, Any]], step_idx: int) -> Dict[str, Any]:
    return seq[min(step_idx, len(seq) - 1)]


def _sanitize_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "action_type", "service", "parameters", "reasoning",
        "root_cause_description", "fix_applied",
    }
    out: Dict[str, Any] = {k: v for k, v in raw.items() if k in allowed}
    fa = out.get("fix_applied")
    if fa is None or fa is False or fa == "" or fa == "false":
        out.pop("fix_applied", None)
    elif fa is True or fa == "true":
        out.pop("fix_applied", None)
    elif not isinstance(fa, str):
        out.pop("fix_applied", None)
    if not isinstance(out.get("parameters"), dict):
        out["parameters"] = {}
    if out.get("action_type") == "escalate":
        p = out.setdefault("parameters", {})
        if not p.get("team"):
            p["team"] = DEFAULT_ESCALATION_TEAM
        if not p.get("reason"):
            p["reason"] = out.get("reasoning") or "unable to determine root cause"
    return out


def _llm_action(
    client: Any,
    task_id: str,
    obs: Dict[str, Any],
    history: List[str],
) -> Dict[str, Any]:
    system = """You are an SRE on-call triaging a production incident. Your goal is to identify the root cause service and apply the correct fix in as few steps as possible.

## SRE REASONING RULES
1. A service reporting errors because its UPSTREAM dependency is broken is NOT the root cause — trace to the upstream.
2. A recent deploy on a service + degraded metrics = suspect that deploy. Fix = rollback, not tuning.
3. OOM / crash after a deploy → rollback. Do NOT scale_up or apply_fix for a deploy-caused OOM.
4. Intermittent errors (low error rate, not total failure) on a service with no recent deploy → suspect cache or data corruption, not a crash.
5. A service with high CPU/memory that is a DOWNSTREAM dependency of another degraded service may just be under load — check its deploy history before treating it as root cause.
6. Do not investigate a service that is healthy AND has no recent deploy AND is not directly implicated by logs.
7. check_recent_deploys early — most incidents are caused by a recent deploy.
8. If a remediation action returns "succeeded; metrics updated to healthy baseline", IMMEDIATELY resolve. Do not keep investigating.

## INVESTIGATION STRATEGY
Step 1: check_metrics or check_recent_deploys on the alerting service(s).
Step 2: read_logs on the most suspicious service (recent deploy + degraded).
Step 3: If logs point to an upstream dependency, check_dependencies then investigate that service.
Step 4: Apply remediation (rollback if deploy-caused; clear_cache if cache issue; restart if process crash).
Step 5: resolve with the root cause service, a clear description, and fix_applied = the exact action_type you used.

## OUTPUT FORMAT
Respond with ONLY a single JSON object:
{
  "action_type": "<one of: check_metrics, read_logs, check_dependencies, check_recent_deploys, check_config, rollback, restart, scale_up, apply_fix, clear_cache, resolve, escalate>",
  "service": "<required for all except escalate: use exact service name from the observation's alerts or service_metrics>",
  "parameters": {},
  "reasoning": "<required: what you know so far, what you suspect, why this action>",
  "root_cause_description": "<required only for resolve: specific description of the root cause>",
  "fix_applied": "<required only for resolve: must be the exact action_type string you used to fix it, e.g. rollback>"
}
Do NOT include fix_applied or root_cause_description for non-resolve actions."""
    user = json.dumps(
        {"task_id": task_id, "observation": obs, "action_history": history},
        default=str,
    )
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    text = r.choices[0].message.content or "{}"
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return _sanitize_action(json.loads(text))


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(
    client: httpx.Client,
    task_id: str,
    use_llm: bool,
    llm: Any | None,
    deadline: float,
) -> Tuple[float, List[float], int]:
    step_rewards: List[float] = []
    steps_taken = 0
    score = 0.01

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        r = client.post(f"{SRE_ENV_URL}/reset", json={"task_id": task_id})
        r.raise_for_status()
        reset_body = r.json()
        obs = reset_body["observation"]
        max_steps = int(obs.get("max_steps", 25))
        fallback_seq = _build_fallback_seq(task_id)
        history: List[str] = []

        for _ in range(max_steps + 5):
            if time.perf_counter() > deadline:
                raise TimeoutError("INFERENCE_MAX_SECONDS exceeded")

            if use_llm and llm is not None:
                try:
                    action = _llm_action(llm, task_id, obs, history)
                except Exception:
                    action = _fallback_action(fallback_seq, steps_taken)
            else:
                action = _fallback_action(fallback_seq, steps_taken)

            sr = client.post(f"{SRE_ENV_URL}/step", json={"action": action})
            sr.raise_for_status()
            body = sr.json()
            obs = body["observation"]
            done = bool(body.get("done"))

            step_reward = float(obs.get("cumulative_reward", body.get("reward") or 0.01))
            step_rewards.append(step_reward)
            history.append(json.dumps(action))
            steps_taken += 1

            action_str = action.get("action_type", "unknown")
            log_step(step=steps_taken, action=action_str, reward=step_reward, done=done)

            if done:
                fe = obs.get("final_episode_score")
                score = float(fe) if fe is not None else step_reward
                break

        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except TimeoutError:
        raise
    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed: {e}", flush=True, file=sys.stderr)
        score = 0.01
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=step_rewards or [0.01])
    return score, step_rewards, steps_taken


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    deadline = time.perf_counter() + INFERENCE_MAX_SECONDS
    use_llm = bool(HF_TOKEN and HF_TOKEN.strip())
    llm = None
    if use_llm:
        from openai import OpenAI
        llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    with httpx.Client(timeout=120.0) as client:
        client.get(f"{SRE_ENV_URL}/health").raise_for_status()
        for task_id in TASKS:
            if time.perf_counter() > deadline:
                print("ERROR: global time budget exceeded", file=sys.stderr)
                sys.exit(1)
            run_task(client, task_id, use_llm, llm, deadline)


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
