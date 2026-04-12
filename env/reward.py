from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from env.models import SREAction

INVESTIGATION_ACTIONS: Set[str] = {
    "check_metrics",
    "read_logs",
    "check_dependencies",
    "check_recent_deploys",
    "check_config",
}

REMEDIATION_ACTIONS: Set[str] = {
    "rollback",
    "restart",
    "scale_up",
    "apply_fix",
    "clear_cache",
}


def format_action(action: SREAction) -> str:
    if action.action_type == "resolve":
        svc = action.service or ""
        return f"resolve({svc})"
    if action.action_type == "escalate":
        team = action.parameters.get("team", "")
        return f"escalate({team})"
    if action.action_type == "read_logs":
        lvl = action.parameters.get("level", "ERROR")
        return f"read_logs({action.service},{lvl})"
    if action.action_type == "scale_up":
        n = action.parameters.get("instances", "?")
        return f"scale_up({action.service},{n})"
    if action.action_type == "apply_fix":
        ft = action.parameters.get("fix_type", "?")
        return f"apply_fix({action.service},{ft})"
    if action.service:
        return f"{action.action_type}({action.service})"
    return action.action_type


def fuzzy_match_service(agent_value: str | None, truth: str) -> float:
    if not agent_value:
        return 0.0
    a = agent_value.strip().lower()
    b = truth.strip().lower()
    if a == b:
        return 1.0
    if a.replace("_", "-") == b.replace("_", "-"):
        return 1.0
    return 0.0


def normalize_fix(fix: str | None) -> str:
    if not fix:
        return ""
    return fix.strip().lower()


def compute_step_reward(
    action: SREAction,
    actions_taken: List[str],
    investigated_services: Set[str],
    root_cause_service: str,
    red_herring_services: List[str],
    correct_fix_action: str,
) -> Tuple[float, Dict[str, float]]:
    reward = 0.0
    breakdown: Dict[str, float] = {}

    sla_penalty = -0.02
    reward += sla_penalty
    breakdown["sla_bleed"] = sla_penalty

    action_str = format_action(action)

    if action.action_type in INVESTIGATION_ACTIONS and action.service:
        if action.service == root_cause_service:
            if root_cause_service not in investigated_services:
                reward += 0.08
                breakdown["correct_investigation"] = 0.08
        elif action.service in red_herring_services:
            reward -= 0.03
            breakdown["red_herring_penalty"] = -0.03

    if action.action_type in REMEDIATION_ACTIONS and action.service:
        if action.service == root_cause_service:
            if action.action_type == correct_fix_action:
                reward += 0.15
                breakdown["correct_remediation"] = 0.15
            else:
                reward -= 0.08
                breakdown["wrong_fix"] = -0.08
        else:
            reward -= 0.05
            breakdown["wrong_service_fix"] = -0.05

    if actions_taken and action_str in actions_taken[-3:]:
        reward -= 0.05
        breakdown["repeat_penalty"] = -0.05

    return reward, breakdown


def compute_terminal_reward(
    action: SREAction,
    sla_seconds_remaining: int,
    sla_seconds_total: int,
    root_cause_service: str,
    correct_fix_action: str,
    escalation_acceptable: bool,
) -> Tuple[float, Dict[str, float]]:
    reward = 0.0
    breakdown: Dict[str, float] = {}

    if action.action_type == "resolve":
        svc = action.service or ""
        cause_match = fuzzy_match_service(svc, root_cause_service)
        reward += 0.30 * cause_match
        breakdown["terminal_cause"] = 0.30 * cause_match

        fix_val = action.fix_applied or action.parameters.get("fix_applied", "")
        fix_ok = normalize_fix(fix_val) == normalize_fix(correct_fix_action)
        reward += 0.40 if fix_ok else 0.0
        breakdown["terminal_fix"] = 0.40 if fix_ok else 0.0

        if sla_seconds_total > 0:
            sla_fraction = max(0.0, min(1.0, sla_seconds_remaining / sla_seconds_total))
        else:
            sla_fraction = 0.0
        speed = 0.20 * sla_fraction
        reward += speed
        breakdown["terminal_speed"] = speed

    elif action.action_type == "escalate":
        if escalation_acceptable:
            reward += 0.10
            breakdown["escalate_ok"] = 0.10
        else:
            reward -= 0.05
            breakdown["escalate_bad"] = -0.05

    return reward, breakdown


def clamp_episode_score(cumulative_reward: float) -> float:
    return max(0.01, min(0.99, cumulative_reward))
