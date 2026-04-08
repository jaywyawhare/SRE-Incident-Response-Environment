from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class Alert(BaseModel):
    service: str
    severity: Literal["info", "warning", "critical"]
    message: str


class ServiceMetrics(BaseModel):
    cpu_percent: int = Field(..., serialization_alias="cpu", validation_alias="cpu")
    memory_percent: int = Field(..., serialization_alias="memory", validation_alias="memory")
    error_rate: float
    latency_p99_ms: int
    request_rate_rps: float = 0.0
    status: Literal["healthy", "degraded", "down"]
    last_deploy_ago_minutes: int = 0

    model_config = {"populate_by_name": True}


class SREAction(Action):
    action_type: Literal[
        "check_metrics",
        "read_logs",
        "check_dependencies",
        "check_recent_deploys",
        "check_config",
        "rollback",
        "restart",
        "scale_up",
        "apply_fix",
        "clear_cache",
        "resolve",
        "escalate",
    ]
    service: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None
    root_cause_description: Optional[str] = None
    fix_applied: Optional[str] = None

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
        "populate_by_name": True,
    }


class SREObservation(Observation):
    step: int
    max_steps: int
    sla_seconds_remaining: int
    sla_breached: bool
    alerts: List[Alert]
    service_metrics: Dict[str, ServiceMetrics]
    log_cache: Dict[str, str]
    actions_taken: List[str]
    last_action_result: str
    last_action_error: Optional[str] = None
    incident_resolved: bool
    resolution_submitted: Optional[Dict[str, Any]] = None
    hint: Optional[str] = None
    cumulative_reward: float = 0.0
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    final_episode_score: Optional[float] = None

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
        "populate_by_name": True,
    }


class SREState(State):
    step: int = 0
    max_steps: int = 0
    sla_seconds_remaining: int = 0
    sla_breached: bool = False
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    service_metrics: Dict[str, Any] = Field(default_factory=dict)
    log_cache: Dict[str, str] = Field(default_factory=dict)
    actions_taken: List[str] = Field(default_factory=list)
    incident_resolved: bool = False
    resolution_submitted: Optional[Dict[str, Any]] = None
    task_id: str = ""

    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}


class SREReward(BaseModel):
    step_reward: float
    cumulative_reward: float
    breakdown: Dict[str, float]


Reward = SREReward
