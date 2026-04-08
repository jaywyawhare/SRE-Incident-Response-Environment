"""SRE Incident Response Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from env.models import SREAction, SREObservation, SREState


class SREIncidentEnv(EnvClient[SREAction, SREObservation, SREState]):
    """
    Client for the SRE Incident Response Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with SREIncidentEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset(task_id="easy")
        ...     obs = result.observation
        ...
        ...     action = SREAction(action_type="check_metrics", service="payment-service")
        ...     result = client.step(action)
        ...     print(result.reward, result.done)

    Example with Docker:
        >>> client = SREIncidentEnv.from_docker_image("sre-incident-env:latest")
        >>> try:
        ...     result = client.reset(task_id="easy")
        ...     result = client.step(SREAction(action_type="check_metrics"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SREAction) -> Dict:
        return {"action": action.model_dump(mode="json")}

    def _parse_result(self, payload: Dict) -> StepResult[SREObservation]:
        obs_data = payload.get("observation", {})
        observation = SREObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SREState:
        return SREState(**payload)
