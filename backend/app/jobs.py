"""Evaluation runs in the background: the API answers at once, clients poll the run."""

import logging
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from typing import Literal

from app.evaluation.runner import EvaluationReport, Progress

log = logging.getLogger("rag")


class RunNotFoundError(Exception):
    """No evaluation run has this id, or it was dropped."""


@dataclass
class EvaluationRun:
    """An evaluation and its progress; ``version`` grows with every change."""

    id: str
    status: Literal["running", "completed", "failed"] = "running"
    done: int = 0
    total: int = 0
    report: EvaluationReport | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    version: int = 0


class EvaluationRuns:
    """The last ``max_runs`` runs of this process, newest last.

    Runs change in a worker thread; ``start``, ``get`` and ``latest`` answer snapshots, whose
    fields are consistent with each other. ``on_complete`` receives each completed run (to
    save it, for one).
    """

    def __init__(
        self,
        runner: Executor,
        max_runs: int = 20,
        on_complete: Callable[[EvaluationRun], None] | None = None,
    ) -> None:
        self._runner = runner
        self._max_runs = max_runs
        self._on_complete = on_complete
        self._runs: OrderedDict[str, EvaluationRun] = OrderedDict()
        self._lock = threading.Lock()

    def start(self, evaluate: Callable[[Progress], EvaluationReport]) -> EvaluationRun:
        run = EvaluationRun(secrets.token_urlsafe(9))
        with self._lock:
            self._runs[run.id] = run
            while len(self._runs) > self._max_runs:
                self._runs.popitem(last=False)
            snapshot = replace(run)
        self._runner.submit(self._execute, run, evaluate)
        return snapshot

    def get(self, run_id: str) -> EvaluationRun:
        with self._lock:
            run = self._runs.get(run_id)
            snapshot = replace(run) if run is not None else None
        if snapshot is None:
            raise RunNotFoundError(run_id)
        return snapshot

    def latest(self) -> EvaluationRun | None:
        """The newest completed run."""
        with self._lock:
            completed = [run for run in self._runs.values() if run.status == "completed"]
            return replace(completed[-1]) if completed else None

    def running(self) -> int:
        with self._lock:
            return sum(1 for run in self._runs.values() if run.status == "running")

    def _execute(
        self, run: EvaluationRun, evaluate: Callable[[Progress], EvaluationReport]
    ) -> None:
        def progress(done: int, total: int) -> None:
            with self._lock:
                run.done, run.total = done, total
                run.version += 1

        try:
            report = evaluate(progress)
        except Exception as error:  # every failure is reported to the client
            with self._lock:
                run.status, run.error = "failed", _message(error)
                run.finished_at, run.version = time.time(), run.version + 1
            return
        with self._lock:
            run.status, run.report = "completed", report
            run.finished_at, run.version = time.time(), run.version + 1
            completed = replace(run)
        if self._on_complete is not None:
            try:
                self._on_complete(completed)
            except Exception:  # the run stays available in memory
                log.exception("could not hand over evaluation run %s", run.id)


def _message(error: Exception) -> str:
    text = str(error).strip()
    return text if text else "The evaluation failed unexpectedly."
