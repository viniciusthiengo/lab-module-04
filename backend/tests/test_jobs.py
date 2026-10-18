import pytest

from app.evaluation.runner import EvaluationReport, Progress
from app.jobs import EvaluationRun, EvaluationRuns, RunNotFoundError
from app.rag import Cost
from tests.conftest import HeldExecutor, InlineExecutor


def report() -> EvaluationReport:
    return EvaluationReport(
        dataset="d",
        codebases=["c"],
        k=5,
        settings=["bm25"],
        answer_setting=None,
        metrics={},
        generation=None,
        examples=[],
        cost=Cost(),
        generator_model=None,
        judge_model=None,
        duration_ms=1,
    )


def evaluate_in_steps(steps: int):
    def evaluate(progress: Progress | None) -> EvaluationReport:
        for done in range(1, steps + 1):
            if progress:
                progress(done, steps)
        return report()

    return evaluate


def fail(message: str):
    def evaluate(_: Progress | None) -> EvaluationReport:
        raise RuntimeError(message)

    return evaluate


class TestEvaluationRuns:
    def test_a_run_reports_its_progress_and_result(self) -> None:
        runs = EvaluationRuns(InlineExecutor())

        started = runs.start(evaluate_in_steps(3))
        run = runs.get(started.id)

        assert started.status == "running"
        assert (run.status, run.done, run.total) == ("completed", 3, 3)
        assert run.version == 4  # three steps and the end
        assert run.report is not None
        assert run.finished_at is not None
        assert runs.latest() == run

    def test_a_failure_is_reported(self) -> None:
        runs = EvaluationRuns(InlineExecutor())

        run = runs.get(runs.start(fail("judge unavailable")).id)
        blank = runs.get(runs.start(fail("  ")).id)

        assert (run.status, run.error, run.report) == ("failed", "judge unavailable", None)
        assert blank.error == "The evaluation failed unexpectedly."
        assert runs.latest() is None

    def test_running_runs_are_counted_and_snapshots_do_not_change(self) -> None:
        runner = HeldExecutor()
        runs = EvaluationRuns(runner)
        run = runs.start(evaluate_in_steps(2))

        snapshot = runs.get(run.id)
        assert runs.running() == 1
        runner.run_all()

        assert (snapshot.status, snapshot.version) == ("running", 0)
        assert runs.get(run.id).status == "completed"
        assert runs.running() == 0

    def test_completed_runs_are_handed_over(self) -> None:
        saved: list[str] = []

        def save(run: EvaluationRun) -> None:
            saved.append(run.status)
            raise OSError("disk full")  # logged; the run is still available

        runs = EvaluationRuns(InlineExecutor(), on_complete=save)
        run = runs.start(evaluate_in_steps(1))
        runs.start(fail("no"))

        assert saved == ["completed"]
        assert runs.get(run.id).status == "completed"

    def test_old_runs_are_dropped(self) -> None:
        runs = EvaluationRuns(InlineExecutor(), max_runs=2)
        first = runs.start(evaluate_in_steps(1))
        runs.start(evaluate_in_steps(1))
        newest = runs.start(evaluate_in_steps(1))

        with pytest.raises(RunNotFoundError):
            runs.get(first.id)
        assert runs.latest().id == newest.id  # type: ignore[union-attr]
