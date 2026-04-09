from env.environment import EpisodeStep, SREIncidentEnv
from env.models import SREAction
from env.tasks.base import load_scenario
from graders.grader import TaskGrader


def test_perfect_solution_scores_near_1() -> None:
    task = load_scenario("easy")
    grader = TaskGrader(task)
    hist: list[EpisodeStep] = [
        EpisodeStep(0, SREAction(action_type="check_metrics", service="payment-service")),
        EpisodeStep(1, SREAction(action_type="read_logs", service="payment-service")),
        EpisodeStep(2, SREAction(action_type="rollback", service="payment-service")),
    ]
    res = SREAction(
        action_type="resolve",
        service="payment-service",
        root_cause_description="OOM",
        fix_applied="rollback",
    )
    score = grader.grade_episode(hist, res)
    assert 0.85 <= score < 1.0


def test_wrong_service_scores_low() -> None:
    task = load_scenario("easy")
    grader = TaskGrader(task)
    hist: list[EpisodeStep] = []
    res = SREAction(
        action_type="resolve",
        service="database",
        root_cause_description="guess",
        fix_applied="rollback",
    )
    score = grader.grade_episode(hist, res)
    assert score < 0.5


def test_correct_cause_wrong_fix_partial_credit() -> None:
    task = load_scenario("easy")
    grader = TaskGrader(task)
    hist = [
        EpisodeStep(
            i,
            SREAction(action_type="check_metrics", service="payment-service"),
        )
        for i in range(15)
    ]
    res = SREAction(
        action_type="resolve",
        service="payment-service",
        root_cause_description="OOM",
        fix_applied="restart",
    )
    score = grader.grade_episode(hist, res)
    assert 0.4 <= score <= 0.75


def test_scores_between_0_and_1() -> None:
    task = load_scenario("hard")
    grader = TaskGrader(task)
    for _ in range(3):
        s = grader.grade_episode([], None)
        assert 0.0 < s < 1.0


def test_deterministic_same_input_same_output() -> None:
    task = load_scenario("medium")
    grader = TaskGrader(task)
    hist = [
        EpisodeStep(0, SREAction(action_type="check_metrics", service="auth-service")),
    ]
    res = SREAction(
        action_type="resolve",
        service="auth-service",
        root_cause_description="pool exhausted",
        fix_applied="rollback",
    )
    a = grader.grade_episode(hist, res)
    b = grader.grade_episode(hist, res)
    assert a == b


def test_grader_uses_episode_env() -> None:
    env = SREIncidentEnv()
    env.reset("easy")
    env.step(SREAction(action_type="rollback", service="payment-service"))
    task = env.task
    grader = TaskGrader(task)
    score = grader.grade_episode(
        env.episode_history,
        SREAction(
            action_type="resolve",
            service="payment-service",
            root_cause_description="OOM",
            fix_applied="rollback",
        ),
    )
    assert 0.0 < score < 1.0
