from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.interfaces import EnvironmentMetadata

from env.models import SREAction, SREObservation, SREState
from env.reward import (
    INVESTIGATION_ACTIONS,
    REMEDIATION_ACTIONS,
    clamp_episode_score,
    compute_step_reward,
    compute_terminal_reward,
    format_action,
)
from env.tasks.base import TaskConfig, load_scenario


@dataclass
class EpisodeStep:
    step_index: int
    action: SREAction


class SREIncidentEnv(Environment[SREAction, SREObservation, SREState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._task: TaskConfig
        self.reset("easy")

    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SREObservation:
        self._task = load_scenario(task_id)
        self._step = 0
        self._sla_seconds_remaining = self._task.sla_seconds_total
        self._sla_breached = False
        self._service_metrics = self._copy_metrics(self._task.service_metrics)
        self._log_cache: Dict[str, str] = {}
        self._actions_taken: List[str] = []
        self._investigated_services: Set[str] = set()
        self._cumulative_reward = 0.0
        self._incident_resolved = False
        self._resolution_submitted: Optional[Dict[str, Any]] = None
        self._last_action_result = "Environment reset. Investigate alerts and metrics."
        self._last_action_error: Optional[str] = None
        self._done = False
        self._episode_history: List[EpisodeStep] = []
        self._resolution_action: Optional[SREAction] = None
        return self._build_observation()

    def step(
        self,
        action: SREAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SREObservation:
        if self._done:
            obs = self._build_observation()
            obs.done = True
            return obs

        if self._step >= self._task.max_steps:
            obs = self._build_observation()
            obs.done = True
            return obs

        err = self._validate_action(action)
        if err:
            self._last_action_error = err
            self._last_action_result = ""
            self._advance_time()
            penalty = -0.03
            self._cumulative_reward += penalty
            self._check_terminal_conditions()
            obs = self._build_observation(
                step_reward=penalty,
                breakdown={"sla_bleed": -0.02, "invalid_action": -0.01},
            )
            return obs

        self._last_action_error = None
        action_str = format_action(action)

        step_r, breakdown = compute_step_reward(
            action,
            self._actions_taken,
            self._investigated_services,
            self._task.ground_truth.root_cause_service,
            self._task.ground_truth.red_herring_services,
            self._task.ground_truth.correct_fix_action,
        )

        terminal_r = 0.0
        terminal_breakdown: Dict[str, float] = {}
        if action.action_type in ("resolve", "escalate"):
            terminal_r, terminal_breakdown = compute_terminal_reward(
                action,
                self._sla_seconds_remaining,
                self._task.sla_seconds_total,
                self._task.ground_truth.root_cause_service,
                self._task.ground_truth.correct_fix_action,
                self._task.ground_truth.escalation_acceptable,
            )
            breakdown.update(terminal_breakdown)

        total_step_reward = step_r + terminal_r
        self._cumulative_reward += total_step_reward

        result_text = self._apply_action(action)
        self._last_action_result = result_text

        self._actions_taken.append(action_str)
        self._episode_history.append(EpisodeStep(step_index=self._step, action=action))

        if action.action_type in INVESTIGATION_ACTIONS and action.service:
            self._investigated_services.add(action.service)

        if action.action_type in ("resolve", "escalate"):
            self._resolution_action = action
            self._finalize_terminal_action(action)

        self._advance_time()
        self._check_terminal_conditions()

        final_score: Optional[float] = None
        if self._done:
            from graders.grader import TaskGrader
            final_score = TaskGrader(self._task).grade_episode(
                self._episode_history, self._resolution_action
            )

        obs = self._build_observation(
            step_reward=total_step_reward,
            breakdown=breakdown,
            final_episode_score=final_score,
        )
        return obs

    @property
    def state(self) -> SREState:
        return SREState(
            step=self._step,
            max_steps=self._task.max_steps,
            sla_seconds_remaining=self._sla_seconds_remaining,
            sla_breached=self._sla_breached,
            alerts=[a.model_dump(by_alias=True) for a in self._task.alerts],
            service_metrics={
                k: v.model_dump(by_alias=True) for k, v in self._service_metrics.items()
            },
            log_cache=dict(self._log_cache),
            actions_taken=list(self._actions_taken),
            incident_resolved=self._incident_resolved,
            resolution_submitted=self._resolution_submitted,
            task_id=self._task.task_id,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="sre-incident-env",
            description=(
                "Agent acts as an on-call SRE: triage alerts, read metrics and logs, "
                "remediate, and resolve production incidents before SLA expires."
            ),
            version="1.0.0",
        )

    def close(self) -> None:
        return None

    @staticmethod
    def _copy_metrics(metrics):
        return {k: v.model_copy(deep=True) for k, v in metrics.items()}

    def _advance_time(self) -> None:
        self._step += 1
        self._sla_seconds_remaining = max(0, self._sla_seconds_remaining - 1)
        if self._sla_seconds_remaining <= 0:
            self._sla_breached = True

    def _check_terminal_conditions(self) -> None:
        if self._incident_resolved or self._sla_breached:
            self._done = True
            return
        if self._step >= self._task.max_steps:
            self._done = True

    def _finalize_terminal_action(self, action: SREAction) -> None:
        self._incident_resolved = True
        if action.action_type == "resolve":
            desc = action.root_cause_description or action.parameters.get("root_cause_description", "")
            fix = action.fix_applied or action.parameters.get("fix_applied", "")
            self._resolution_submitted = {
                "root_cause_service": action.service,
                "root_cause_description": desc,
                "fix_applied": fix,
            }
        else:
            self._resolution_submitted = {
                "team": action.parameters.get("team"),
                "reason": action.parameters.get("reason"),
            }

    def _build_observation(
        self,
        step_reward: float = 0.0,
        breakdown: Optional[Dict[str, float]] = None,
        final_episode_score: Optional[float] = None,
    ) -> SREObservation:
        return SREObservation(
            step=self._step,
            max_steps=self._task.max_steps,
            sla_seconds_remaining=self._sla_seconds_remaining,
            sla_breached=self._sla_breached,
            alerts=list(self._task.alerts),
            service_metrics=self._copy_metrics(self._service_metrics),
            log_cache=dict(self._log_cache),
            actions_taken=list(self._actions_taken),
            last_action_result=self._last_action_result,
            last_action_error=self._last_action_error,
            incident_resolved=self._incident_resolved,
            resolution_submitted=self._resolution_submitted,
            hint=self._task.hint,
            done=self._done,
            reward=step_reward,
            cumulative_reward=max(0.0, min(1.0, self._cumulative_reward)),
            reward_breakdown=breakdown or {},
            final_episode_score=final_episode_score,
        )

    def _validate_action(self, action: SREAction) -> Optional[str]:
        if action.action_type in INVESTIGATION_ACTIONS or action.action_type in REMEDIATION_ACTIONS:
            if not action.service:
                return "service is required for this action"
            if action.service not in self._task.service_metrics:
                return f"unknown service: {action.service}"

        if action.action_type == "scale_up":
            if "instances" not in action.parameters:
                return "scale_up requires parameters.instances"

        if action.action_type == "apply_fix":
            if "fix_type" not in action.parameters:
                return "apply_fix requires parameters.fix_type"

        if action.action_type == "resolve":
            if not action.service:
                return "resolve requires service (root cause service)"
            desc = action.root_cause_description or action.parameters.get("root_cause_description")
            fix = action.fix_applied or action.parameters.get("fix_applied", "")
            if not desc or not str(desc).strip():
                return "resolve requires root_cause_description"
            if not fix or not str(fix).strip():
                return "resolve requires fix_applied"

        if action.action_type == "escalate":
            if not action.parameters.get("team") or not action.parameters.get("reason"):
                return "escalate requires parameters.team and parameters.reason"

        return None

    def _remediation_restores(self, action: SREAction) -> bool:
        gt = self._task.ground_truth
        svc = action.service
        if not svc or svc != gt.root_cause_service:
            return False
        at = action.action_type
        if at == gt.correct_fix_action:
            return True
        if at in gt.acceptable_fixes:
            params = gt.acceptable_fix_params
            if not params or all(action.parameters.get(k) == v for k, v in params.items()):
                return True
        return False

    def _apply_action(self, action: SREAction) -> str:
        if action.action_type == "check_metrics":
            assert action.service
            m = self._service_metrics[action.service]
            d = m.model_dump(by_alias=True)
            return f"Metrics for {action.service}: {d}"

        if action.action_type == "read_logs":
            assert action.service
            level = action.parameters.get("level", "ERROR")
            text = self._task.logs.get(action.service, f"(no logs for {action.service})")
            self._log_cache[action.service] = text
            return f"Logs ({level}) for {action.service}:\n{text}"

        if action.action_type == "check_dependencies":
            assert action.service
            dep = self._task.dependencies.get(action.service, {"upstream": [], "downstream": []})
            return (
                f"Dependencies for {action.service}: "
                f"upstream={dep.get('upstream', [])}, downstream={dep.get('downstream', [])}"
            )

        if action.action_type == "check_recent_deploys":
            assert action.service
            hist = self._task.deploy_history.get(action.service, [])
            return f"Recent deploys for {action.service}: {hist[:5]}"

        if action.action_type == "check_config":
            assert action.service
            cfg = self._task.configs.get(action.service, "(no config on file)")
            return f"Config for {action.service}: {cfg}"

        if action.action_type in REMEDIATION_ACTIONS:
            if self._remediation_restores(action):
                self._service_metrics = self._copy_metrics(self._task.resolved_service_metrics)
                return (
                    f"{action.action_type} on {action.service} succeeded; metrics updated to healthy baseline."
                )
            return (
                f"{action.action_type} on {action.service} applied; scenario state unchanged "
                f"(wrong service or wrong remediation)."
            )

        if action.action_type == "resolve":
            return "Resolution submitted. Episode complete."

        if action.action_type == "escalate":
            team = action.parameters.get("team", "")
            return f"Escalated to {team}. Episode complete."

        return "OK"

    @property
    def episode_history(self) -> List[EpisodeStep]:
        return list(self._episode_history)

    @property
    def task(self) -> TaskConfig:
        return self._task
