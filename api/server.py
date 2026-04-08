from __future__ import annotations

import threading
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.routing import APIRoute
from openenv.core.env_server.http_server import create_app
from pydantic import ValidationError

from env.environment import SREIncidentEnv
from env.models import SREAction, SREObservation, SREState

app = create_app(
    SREIncidentEnv,
    SREAction,
    SREObservation,
    env_name="sre-incident-env",
    max_concurrent_envs=1,
)

app.routes[:] = [
    r for r in app.routes
    if not (isinstance(r, APIRoute) and r.path in ("/reset", "/step", "/state"))
]

_lock = threading.Lock()
_env = SREIncidentEnv()


def _obs_dict(obs: SREObservation) -> Dict[str, Any]:
    return obs.model_dump(mode="json", by_alias=True, exclude={"done", "reward", "metadata"})


@app.post("/reset")
def reset(body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    task_id = (body or {}).get("task_id", "easy")
    with _lock:
        obs = _env.reset(task_id)
    return {"observation": _obs_dict(obs), "reward": obs.reward, "done": obs.done}


@app.post("/step")
def step_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = SREAction.model_validate(body.get("action", body))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    with _lock:
        obs = _env.step(action)
    return {"observation": _obs_dict(obs), "reward": obs.reward, "done": obs.done}


@app.get("/state")
def state_endpoint() -> Dict[str, Any]:
    with _lock:
        return _env.state.model_dump(mode="json")
