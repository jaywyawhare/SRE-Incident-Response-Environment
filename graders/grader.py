from __future__ import annotations

from typing import List, Optional

from env.environment import EpisodeStep
from env.models import SREAction
from env.reward import REMEDIATION_ACTIONS, normalize_fix
from env.tasks.base import TaskConfig


class TaskGrader:
    def __init__(self, task: TaskConfig) -> None:
        self.task = task

    def grade_episode(
        self,
        episode_history: List[EpisodeStep],
        resolution: Optional[SREAction],
    ) -> float:
        if resolution is None or resolution.action_type != "resolve":
            return 0.0

        score = 0.0
        svc = resolution.service or ""
        gt = self.task.ground_truth

        if svc == gt.root_cause_service:
            score += 0.30
        elif svc in gt.related_services:
            score += 0.10

        fix_val = resolution.fix_applied or resolution.parameters.get("fix_applied", "")
        fix_norm = normalize_fix(str(fix_val) if fix_val else "")
        correct = normalize_fix(gt.correct_fix_action)
        acceptable_norms = [normalize_fix(x) for x in gt.acceptable_fixes]

        if svc == gt.root_cause_service:
            if fix_norm == correct:
                score += 0.40
            elif fix_norm in acceptable_norms:
                score += 0.20
        elif svc in gt.related_services:
            if fix_norm == correct:
                score += 0.20
            elif fix_norm in acceptable_norms:
                score += 0.10

        steps_used = len(episode_history)
        if self.task.max_steps > 0:
            efficiency = 1.0 - (steps_used / self.task.max_steps)
            score += 0.20 * max(efficiency, 0.0)

        wrong_remediation = [
            s
            for s in episode_history
            if s.action.action_type in REMEDIATION_ACTIONS
            and (s.action.service or "") != gt.root_cause_service
        ]
        if not wrong_remediation:
            score += 0.10

        return round(min(score, 1.0), 4)
