from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from env.models import Alert, ServiceMetrics


@dataclass
class GroundTruth:
    root_cause_service: str
    root_cause_description: str
    correct_fix_action: str
    red_herring_services: List[str]
    escalation_acceptable: bool
    related_services: List[str] = field(default_factory=list)
    acceptable_fixes: List[str] = field(default_factory=list)
    acceptable_fix_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    task_id: str
    name: str
    max_steps: int
    sla_seconds_total: int
    hint: Optional[str]
    alerts: List[Alert]
    service_metrics: Dict[str, ServiceMetrics]
    logs: Dict[str, str]
    deploy_history: Dict[str, List[Dict[str, Any]]]
    configs: Dict[str, str]
    dependencies: Dict[str, Dict[str, List[str]]]
    ground_truth: GroundTruth
    resolved_service_metrics: Dict[str, ServiceMetrics]


def _scenarios_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "scenarios"


def load_scenario(task_id: str) -> TaskConfig:
    path = _scenarios_dir() / f"{task_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    gt = raw["ground_truth"]
    ground = GroundTruth(
        root_cause_service=gt["root_cause_service"],
        root_cause_description=gt["root_cause_description"],
        correct_fix_action=gt["correct_fix_action"],
        red_herring_services=list(gt.get("red_herring_services", [])),
        escalation_acceptable=bool(gt.get("escalation_acceptable", False)),
        related_services=list(gt.get("related_services", [])),
        acceptable_fixes=list(gt.get("acceptable_fixes", [])),
        acceptable_fix_params=dict(gt.get("acceptable_fix_params", {})),
    )

    sm: Dict[str, ServiceMetrics] = {}
    for name, m in raw["service_metrics"].items():
        sm[name] = ServiceMetrics.model_validate(m)

    resolved: Dict[str, ServiceMetrics] = {}
    for name, m in raw["resolved_service_metrics"].items():
        resolved[name] = ServiceMetrics.model_validate(m)

    alerts = [Alert.model_validate(a) for a in raw.get("alerts", [])]

    return TaskConfig(
        task_id=raw["task_id"],
        name=raw.get("name", task_id),
        max_steps=int(raw["max_steps"]),
        sla_seconds_total=int(raw["sla_seconds_total"]),
        hint=raw.get("hint"),
        alerts=alerts,
        service_metrics=sm,
        logs=dict(raw.get("logs", {})),
        deploy_history=dict(raw.get("deploy_history", {})),
        configs=dict(raw.get("configs", {})),
        dependencies=dict(raw.get("dependencies", {})),
        ground_truth=ground,
        resolved_service_metrics=resolved,
    )
