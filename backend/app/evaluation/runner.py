"""The evaluation suite: retrieval metrics for each retrieval setting, then the answers of one
setting graded by the judge.

For every question of the dataset and every setting (vector, BM25, hybrid, hybrid with
reranking), the top ``k`` chunks are compared with the code the dataset says answers the
question. Then the RAG answers each question with the chosen setting, and the judge grades
the answer against the retrieved code and the reference answer. Questions run in parallel.

The judge check is a negative control: each question is graded again with the answer to
another question. A judge whose grades mean something fails nearly all of those.
"""

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.chunking import Chunk
from app.evaluation import metrics
from app.evaluation.dataset import EvalDataset, EvalExample
from app.evaluation.judge import Judge, Judgement
from app.rag import CodeRag, Cost, QueryResult
from app.retrieval import Mode


@dataclass(frozen=True)
class Setting:
    """A retrieval configuration to evaluate."""

    mode: Mode
    rerank: bool = False

    @property
    def label(self) -> str:
        return f"{self.mode}+rerank" if self.rerank else self.mode


SETTINGS = {
    setting.label: setting
    for setting in (
        Setting("vector"),
        Setting("bm25"),
        Setting("hybrid"),
        Setting("hybrid", rerank=True),
    )
}


@dataclass(frozen=True)
class RetrievalScore:
    retrieved: list[str]  # keys of the top k chunks
    precision: float
    recall: float
    reciprocal_rank: float
    ndcg: float
    hit: float


@dataclass(frozen=True)
class AnswerScore:
    answer: str
    grounded: bool
    sources_used: list[int]
    judgement: Judgement | None
    context: tuple[Chunk, ...] = field(default=(), repr=False)  # the chunks it was given


@dataclass(frozen=True)
class ExampleResult:
    example: EvalExample
    retrieval: dict[str, RetrievalScore]
    answer: AnswerScore | None


@dataclass(frozen=True)
class SettingMetrics:
    precision: float
    recall: float
    mrr: float
    ndcg: float
    hit_rate: float


@dataclass(frozen=True)
class JudgeCheck:
    """The judge's grades of answers given to other questions: they should fail."""

    judged: int
    pass_rate: float
    faithfulness: float
    relevance: float
    correctness: float


@dataclass(frozen=True)
class GenerationMetrics:
    faithfulness: float
    relevance: float
    correctness: float
    pass_rate: float  # answers with every grade at least 4
    grounded_rate: float
    judged: int
    check: JudgeCheck | None = None


@dataclass(frozen=True)
class EvaluationReport:
    dataset: str
    codebases: list[str]
    k: int
    settings: list[str]
    answer_setting: str | None
    metrics: dict[str, SettingMetrics]
    generation: GenerationMetrics | None
    examples: list[ExampleResult]
    cost: Cost
    generator_model: str | None
    judge_model: str | None
    duration_ms: int
    created_at: float = field(default_factory=time.time)


Progress = Callable[[int, int], None]


class Evaluator:
    def __init__(self, rag: CodeRag, judge: Judge | None, *, workers: int = 4) -> None:
        self._rag = rag
        self._judge = judge
        self._workers = workers

    @property
    def can_judge(self) -> bool:
        return self._judge is not None and self._rag.can_answer

    def run(
        self,
        dataset: EvalDataset,
        *,
        k: int,
        settings: Sequence[str],
        answer_setting: str | None,
        judge_check: bool = False,
        progress: Progress | None = None,
    ) -> EvaluationReport:
        """Evaluates ``settings`` on every example; with ``answer_setting``, also generates
        and judges an answer per example, and with ``judge_check`` checks the judge."""
        started = time.perf_counter()
        chosen = [SETTINGS[label] for label in settings]
        answering = SETTINGS[answer_setting] if answer_setting else None
        if answering is not None and not self.can_judge:
            raise ValueError("judging answers needs a model")
        checking = answering is not None and judge_check
        done = 0
        total = len(dataset.examples) * (2 if checking else 1)
        lock = threading.Lock()

        def tick() -> None:
            nonlocal done
            with lock:
                done += 1
                if progress is not None:
                    progress(done, total)

        def evaluate(example: EvalExample) -> tuple[ExampleResult, Cost]:
            result = self._example(example, dataset, k, chosen, answering)
            tick()
            return result

        check = None
        with ThreadPoolExecutor(self._workers, thread_name_prefix="evaluation") as pool:
            outcomes = list(pool.map(evaluate, dataset.examples))
            examples = [result for result, _ in outcomes]
            cost = Cost()
            for _, example_cost in outcomes:
                cost = cost.merged(example_cost)
            if checking:
                check, check_cost = self._check_judge(pool, examples, tick)
                cost = cost.merged(check_cost)
        return EvaluationReport(
            dataset=dataset.name,
            codebases=list(dataset.codebases),
            k=k,
            settings=[setting.label for setting in chosen],
            answer_setting=answering.label if answering else None,
            metrics={setting.label: _aggregate(examples, setting.label) for setting in chosen},
            generation=_generation(examples, check) if answering else None,
            examples=examples,
            cost=cost,
            generator_model=self._rag.answer_model if answering else None,
            judge_model=self._judge.model if answering and self._judge else None,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )

    def _example(
        self,
        example: EvalExample,
        dataset: EvalDataset,
        k: int,
        settings: Sequence[Setting],
        answering: Setting | None,
    ) -> tuple[ExampleResult, Cost]:
        cost = Cost()
        scores: dict[str, RetrievalScore] = {}
        answered: QueryResult | None = None
        for setting in settings:
            if setting == answering:
                answered = self._rag.ask(
                    example.question, dataset.codebases, k, setting.mode, setting.rerank
                )
                hits, cost = answered.search.hits, cost.merged(answered.cost)
            else:
                search = self._rag.search(
                    example.question, dataset.codebases, k, setting.mode, setting.rerank
                )
                hits, cost = search.hits, cost.merged(search.cost)
            scores[setting.label] = _score([hit.chunk.key for hit in hits], example.relevant, k)
        if answering is not None and answered is None:
            answered = self._rag.ask(
                example.question, dataset.codebases, k, answering.mode, answering.rerank
            )
            cost = cost.merged(answered.cost)
        answer = None
        if answered is not None and self._judge is not None:
            judgement = self._judge.grade(
                example.question,
                [hit.chunk for hit in answered.search.hits],
                answered.answer,
                example.expected_answer,
            )
            cost = cost.add(judgement.model, judgement.usage)
            answer = AnswerScore(
                answered.answer,
                answered.grounded,
                answered.sources_used,
                judgement,
                tuple(hit.chunk for hit in answered.search.hits),
            )
        return ExampleResult(example, scores, answer), cost

    def _check_judge(
        self, pool: ThreadPoolExecutor, examples: Sequence[ExampleResult], tick: Callable[[], None]
    ) -> tuple[JudgeCheck, Cost]:
        """Grades each question with the answer to the next question, against its own
        retrieved code and reference answer."""
        judge = self._judge
        if judge is None:
            raise ValueError("checking the judge needs a model")
        pairs = [
            (example.example, example.answer, other.answer)
            for example, other in zip(examples, [*examples[1:], *examples[:1]], strict=True)
            if example.answer is not None and other.answer is not None
        ]

        def grade(pair: tuple[EvalExample, AnswerScore, AnswerScore]) -> Judgement:
            example, own, swapped = pair
            judgement = judge.grade(
                example.question, own.context, swapped.answer, example.expected_answer
            )
            tick()
            return judgement

        judgements = list(pool.map(grade, pairs)) if len(pairs) > 1 else []
        cost = Cost()
        for judgement in judgements:
            cost = cost.add(judgement.model, judgement.usage)
        check = JudgeCheck(
            judged=len(judgements),
            pass_rate=metrics.mean([1.0 if j.passed else 0.0 for j in judgements]),
            faithfulness=metrics.mean([j.faithfulness for j in judgements]),
            relevance=metrics.mean([j.relevance for j in judgements]),
            correctness=metrics.mean([j.correctness for j in judgements]),
        )
        return check, cost


def _score(keys: list[str], targets: Sequence[str], k: int) -> RetrievalScore:
    return RetrievalScore(
        retrieved=keys,
        precision=metrics.precision_at_k(keys, targets, k),
        recall=metrics.recall_at_k(keys, targets, k),
        reciprocal_rank=metrics.reciprocal_rank(keys, targets),
        ndcg=metrics.ndcg_at_k(keys, targets, k),
        hit=metrics.hit_at_k(keys, targets, k),
    )


def _aggregate(examples: Sequence[ExampleResult], label: str) -> SettingMetrics:
    scores = [example.retrieval[label] for example in examples]
    return SettingMetrics(
        precision=metrics.mean([score.precision for score in scores]),
        recall=metrics.mean([score.recall for score in scores]),
        mrr=metrics.mean([score.reciprocal_rank for score in scores]),
        ndcg=metrics.mean([score.ndcg for score in scores]),
        hit_rate=metrics.mean([score.hit for score in scores]),
    )


def _generation(
    examples: Sequence[ExampleResult], check: JudgeCheck | None = None
) -> GenerationMetrics:
    answers = [example.answer for example in examples if example.answer is not None]
    judged = [answer.judgement for answer in answers if answer.judgement is not None]
    return GenerationMetrics(
        faithfulness=metrics.mean([j.faithfulness for j in judged]),
        relevance=metrics.mean([j.relevance for j in judged]),
        correctness=metrics.mean([j.correctness for j in judged]),
        pass_rate=metrics.mean([1.0 if j.passed else 0.0 for j in judged]),
        grounded_rate=metrics.mean([1.0 if answer.grounded else 0.0 for answer in answers]),
        judged=len(judged),
        check=check,
    )
