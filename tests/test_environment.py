from env.environment import SREIncidentEnv
from env.models import SREAction
from env.reward import clamp_episode_score


def test_reset_returns_valid_observation() -> None:
    env = SREIncidentEnv()
    obs = env.reset("easy")
    assert obs.step == 0
    assert obs.max_steps == 25
    assert "payment-service" in obs.service_metrics
    assert obs.sla_seconds_remaining == 120


def test_step_returns_correct_types() -> None:
    env = SREIncidentEnv()
    env.reset("easy")
    obs = env.step(SREAction(action_type="check_metrics", service="payment-service"))
    assert obs.last_action_error is None
    assert isinstance(obs.cumulative_reward, float)
    assert "sla_bleed" in obs.reward_breakdown


def test_done_on_resolve() -> None:
    env = SREIncidentEnv()
    env.reset("easy")
    env.step(SREAction(action_type="rollback", service="payment-service"))
    obs = env.step(
        SREAction(
            action_type="resolve",
            service="payment-service",
            root_cause_description="OOM after deploy",
            fix_applied="rollback",
        )
    )
    assert obs.done
    assert obs.incident_resolved
    assert obs.final_episode_score is not None


def test_done_on_max_steps() -> None:
    env = SREIncidentEnv()
    env.reset("easy")
    done = False
    for _ in range(30):
        obs = env.step(SREAction(action_type="check_metrics", service="api-gateway"))
        if obs.done:
            done = True
            break
    assert done
    assert env.state.step >= env.state.max_steps


def test_sla_timer_decrements() -> None:
    env = SREIncidentEnv()
    obs = env.reset("easy")
    start = obs.sla_seconds_remaining
    env.step(SREAction(action_type="check_metrics", service="api-gateway"))
    assert env.state.sla_seconds_remaining == start - 1


def test_repeat_action_penalty() -> None:
    env = SREIncidentEnv()
    env.reset("easy")
    a = SREAction(action_type="check_metrics", service="api-gateway")
    env.step(a)
    env.step(a)
    obs = env.step(a)
    assert "repeat_penalty" in obs.reward_breakdown


def test_clamp_episode_score() -> None:
    assert clamp_episode_score(1.5) == 0.99
    assert clamp_episode_score(-0.5) == 0.01
