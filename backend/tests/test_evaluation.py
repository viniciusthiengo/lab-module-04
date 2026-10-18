import json
from pathlib import Path

import pytest

from app.evaluation import metrics
from app.evaluation.dataset import EvalDataset, load_dataset
from app.evaluation.judge import Judge
from app.evaluation.runner import Evaluator
from app.llm.base import Usage
from app.llm.fake import FakeLLMClient
from app.outputs import Answer, Ranking, Verdict
from tests.conftest import Rig, make_rig
from tests.fakes import Excerpt, ScriptedClaude

ALL = ["vector", "bm25", "hybrid", "hybrid+rerank"]


class TestDataset:
    def test_has_enough_unique_questions_with_answers_and_targets(
        self, dataset: EvalDataset
    ) -> None:
        assert len(dataset.examples) >= 10
        assert len({example.id for example in dataset.examples}) == len(dataset.examples)
        for example in dataset.examples:
            assert example.question.endswith("?")
            assert example.expected_answer
            assert example.relevant
            assert all(target.split(":")[0] in dataset.codebases for target in example.relevant)

    def test_every_target_is_a_chunk_of_the_demo_codebases(
        self, dataset: EvalDataset, demo_rig: Rig
    ) -> None:
        keys = [chunk.key for chunk in demo_rig.store.chunks(list(dataset.codebases))]

        missing = [
            target
            for example in dataset.examples
            for target in example.relevant
            if not any(metrics.matches(key, target) for key in keys)
        ]

        assert missing == []

    def test_load_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.json"
        example = {"id": "a", "question": "Q?", "expected_answer": "A.", "relevant": ["x:y::z"]}
        path.write_text(json.dumps({"name": "n", "codebases": ["x"], "examples": [example]}))

        dataset = load_dataset(path)

        assert (dataset.name, dataset.description, dataset.codebases) == ("n", "", ("x",))
        assert dataset.examples[0].relevant == ("x:y::z",)


def evaluator(rig: Rig, judge: FakeLLMClient | None = None) -> Evaluator:
    model = judge or rig.llm
    return Evaluator(rig.rag, Judge(model) if model else None)


class TestEvaluator:
    def test_measures_every_setting_and_judges_every_answer(
        self, dataset: EvalDataset, demo_rig: Rig
    ) -> None:
        progress: list[tuple[int, int]] = []

        report = evaluator(demo_rig).run(
            dataset,
            k=5,
            settings=ALL,
            answer_setting="hybrid+rerank",
            progress=lambda done, total: progress.append((done, total)),
        )

        n = len(dataset.examples)
        assert (report.k, report.settings, report.answer_setting) == (5, ALL, "hybrid+rerank")
        assert set(report.metrics) == set(ALL)
        for value in report.metrics.values():
            for score in (value.precision, value.recall, value.mrr, value.ndcg, value.hit_rate):
                assert 0 <= score <= 1
        assert report.metrics["bm25"].hit_rate > 0.5
        generation = report.generation
        assert generation is not None
        assert (generation.judged, generation.pass_rate, generation.grounded_rate) == (n, 1, 1)
        assert (generation.faithfulness, generation.relevance, generation.correctness) == (5, 5, 4)
        assert report.cost.calls == 3 * n  # reranking, answer and judge per question
        assert (report.generator_model, report.judge_model) == ("fake-model", "fake-model")
        assert sorted(progress) == [(i, n) for i in range(1, n + 1)]
        assert all(example.answer and example.answer.judgement for example in report.examples)

    def test_the_answers_come_from_the_retrieval_that_is_measured(
        self, dataset: EvalDataset, demo_rig: Rig
    ) -> None:
        report = evaluator(demo_rig).run(
            dataset, k=3, settings=["hybrid+rerank"], answer_setting="hybrid+rerank"
        )

        reranks = demo_rig.llm.calls_for(Ranking)  # type: ignore[union-attr]
        assert len(reranks) == len(dataset.examples)  # one retrieval per question, not two
        first = report.examples[0]
        assert len(first.retrieval["hybrid+rerank"].retrieved) == 3

    def test_retrieval_only_needs_no_model(self, dataset: EvalDataset) -> None:
        rig = make_rig(None).index_demo()

        report = Evaluator(rig.rag, None).run(
            dataset, k=5, settings=["vector", "bm25", "hybrid"], answer_setting=None
        )

        assert report.generation is None
        assert report.cost.calls == 0
        assert (report.generator_model, report.judge_model) == (None, None)
        assert all(example.answer is None for example in report.examples)

    def test_answers_with_a_setting_that_is_not_measured(
        self, dataset: EvalDataset, demo_rig: Rig
    ) -> None:
        report = evaluator(demo_rig).run(dataset, k=5, settings=["bm25"], answer_setting="hybrid")

        assert list(report.metrics) == ["bm25"]
        assert report.generation and report.generation.judged == len(dataset.examples)
        assert demo_rig.llm.calls_for(Ranking) == []  # type: ignore[union-attr]

    def test_a_separate_judge_model_grades_the_answers(
        self, dataset: EvalDataset, demo_rig: Rig
    ) -> None:
        judge = FakeLLMClient(ScriptedClaude(), model="judge-model", usage=Usage(2000, 100))

        report = evaluator(demo_rig, judge).run(
            dataset, k=5, settings=["hybrid"], answer_setting="hybrid"
        )

        assert len(judge.calls_for(Verdict)) == len(dataset.examples)
        assert demo_rig.llm.calls_for(Verdict) == []  # type: ignore[union-attr]
        assert len(demo_rig.llm.calls_for(Answer)) == len(dataset.examples)  # type: ignore[union-attr]
        assert (report.generator_model, report.judge_model) == ("fake-model", "judge-model")
        assert report.cost.models == ("fake-model", "judge-model")

    def test_the_judge_check_grades_answers_given_to_other_questions(
        self, dataset: EvalDataset, demo_rig: Rig
    ) -> None:
        def verdict(question: str, answer: str, found: list[Excerpt]) -> dict[str, object]:
            # An answer passes when it names code among the question's own excerpts.
            grade = 5 if any(f"`{excerpt.name}`" in answer for excerpt in found) else 1
            return {
                "faithfulness": grade,
                "relevance": grade,
                "correctness": grade,
                "rationale": ".",
            }

        judge = FakeLLMClient(ScriptedClaude(verdict=verdict), model="judge-model")
        progress: list[tuple[int, int]] = []

        report = evaluator(demo_rig, judge).run(
            dataset,
            k=5,
            settings=["hybrid"],
            answer_setting="hybrid",
            judge_check=True,
            progress=lambda done, total: progress.append((done, total)),
        )

        n = len(dataset.examples)
        generation = report.generation
        assert generation is not None and generation.check is not None
        assert generation.pass_rate == 1
        assert generation.check.judged == n
        assert generation.check.pass_rate < 0.5
        assert generation.check.correctness < generation.correctness
        assert len(judge.calls_for(Verdict)) == 2 * n
        assert sorted(progress)[-1] == (2 * n, 2 * n)
        swapped = judge.calls_for(Verdict)[-1].prompt
        assert "<question>" in swapped and "<reference_answer>" in swapped

    def test_no_judge_check_without_answers(self, dataset: EvalDataset, demo_rig: Rig) -> None:
        report = evaluator(demo_rig).run(
            dataset, k=5, settings=["bm25"], answer_setting=None, judge_check=True
        )

        assert report.generation is None
        assert demo_rig.llm.requests == []  # type: ignore[union-attr]

    def test_judging_needs_a_model(self, dataset: EvalDataset) -> None:
        rig = make_rig(None).index_demo()

        with pytest.raises(ValueError, match="needs a model"):
            Evaluator(rig.rag, None).run(dataset, k=5, settings=["bm25"], answer_setting="bm25")


class TestJudge:
    def test_grades_are_clamped_and_pass_at_four(self) -> None:
        verdict = {
            "faithfulness": 9,
            "relevance": 4,
            "correctness": 0,
            "rationale": " Wrong.\u200b",
        }
        judge = Judge(FakeLLMClient(ScriptedClaude(verdict=verdict)))

        judgement = judge.grade("Q?", [], "An answer.", "The reference.")

        assert (judgement.faithfulness, judgement.relevance, judgement.correctness) == (5, 4, 1)
        assert (judgement.rationale, judgement.passed) == ("Wrong.", False)

    @pytest.mark.parametrize(
        ("rationale", "clean"),
        [
            ("Matches the reference.\u200b", "Matches the reference."),
            ("Matches the reference.ăn", "Matches the reference."),
            ("Supported by the excerpts.'", "Supported by the excerpts."),
            ("Graded 4.5", "Graded 4.5"),
            ("Says so in v2.", "Says so in v2."),
        ],
    )
    def test_stray_characters_are_removed_from_the_rationale(
        self, rationale: str, clean: str
    ) -> None:
        verdict = {"faithfulness": 5, "relevance": 5, "correctness": 5, "rationale": rationale}

        judgement = Judge(FakeLLMClient(ScriptedClaude(verdict=verdict))).grade("Q?", [], "A", "R")

        assert judgement.rationale == clean

    def test_the_judge_sees_the_question_excerpts_answer_and_reference(self, demo_rig: Rig) -> None:
        llm = FakeLLMClient(ScriptedClaude())
        chunks = demo_rig.store.chunks(["bookshop-api"])[:2]

        judgement = Judge(llm).grade("How?", chunks, "Like this [1].", "Like that.")

        prompt = llm.requests[0].prompt
        assert prompt.startswith("<question>How?</question>")
        assert "<answer>\nLike this [1].\n</answer>" in prompt
        assert "<reference_answer>\nLike that.\n</reference_answer>" in prompt
        assert prompt.count("<source id=") == 2
        assert judgement.passed
