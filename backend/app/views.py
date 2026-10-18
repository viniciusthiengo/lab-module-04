"""The API's view of the domain objects."""

from app.evaluation.dataset import EvalDataset
from app.evaluation.runner import EvaluationReport, ExampleResult
from app.jobs import EvaluationRun
from app.rag import Cost, IndexReport, QueryResult, SearchResult
from app.retrieval import Hit, Mode
from app.schemas import (
    AnswerScoreView,
    CodebaseView,
    DatasetExampleView,
    DatasetView,
    EvaluationReportView,
    EvaluationRunView,
    ExampleView,
    FileView,
    GenerationMetricsView,
    HitView,
    IndexResponse,
    JudgeCheckView,
    JudgementView,
    QueryResponse,
    RetrievalScoreView,
    SearchResponse,
    SettingMetricsView,
    UsageView,
)
from app.store import CodebaseInfo


def hit_view(hit: Hit) -> HitView:
    chunk = hit.chunk
    return HitView(
        id=chunk.id,
        key=chunk.key,
        codebase=chunk.codebase,
        path=chunk.path,
        language=chunk.language,
        kind=chunk.kind,
        name=chunk.name,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        content=chunk.content,
        score=round(hit.score, 6),
        vector_rank=hit.vector_rank,
        bm25_rank=hit.bm25_rank,
        relevance=hit.relevance,
    )


def usage_view(cost: Cost) -> UsageView:
    usage = cost.usage
    return UsageView(
        model_calls=cost.calls,
        input_tokens=usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=round(cost.cost_usd, 6),
        models=list(cost.models),
    )


def search_view(result: SearchResult, mode: Mode, rerank: bool, k: int) -> SearchResponse:
    return SearchResponse(
        hits=[hit_view(hit) for hit in result.hits],
        codebases=result.codebases,
        mode=mode,
        rerank=rerank,
        k=k,
        candidates=result.candidates,
        usage=usage_view(result.cost),
        duration_ms=result.duration_ms,
    )


def query_view(result: QueryResult, mode: Mode, rerank: bool, k: int) -> QueryResponse:
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        grounded=result.grounded,
        sources_used=result.sources_used,
        sources=[hit_view(hit) for hit in result.search.hits],
        codebases=result.search.codebases,
        mode=mode,
        rerank=rerank,
        k=k,
        cached=result.cached,
        usage=usage_view(result.cost),
        duration_ms=result.duration_ms,
    )


def codebase_view(info: CodebaseInfo, demo: bool = False) -> CodebaseView:
    return CodebaseView(
        name=info.name,
        demo=demo,
        files=[
            FileView(path=f.path, language=f.language, lines=f.lines, chunks=f.chunks)
            for f in sorted(info.files.values(), key=lambda f: f.path)
        ],
        chunks=info.chunks,
        languages=info.languages,
        version=info.version,
        updated_at=info.updated_at,
    )


def index_view(report: IndexReport, demo: bool = False) -> IndexResponse:
    return IndexResponse(
        codebase=codebase_view(report.info, demo),
        files=report.files,
        chunks=report.chunks,
        embedded=report.embedded,
        reused=report.reused,
        removed=report.removed,
        kinds=report.kinds,
        duration_ms=report.duration_ms,
    )


def report_view(report: EvaluationReport) -> EvaluationReportView:
    generation = report.generation
    return EvaluationReportView(
        dataset=report.dataset,
        codebases=report.codebases,
        k=report.k,
        settings=report.settings,
        answer_setting=report.answer_setting,
        metrics={
            label: SettingMetricsView(
                precision=round(m.precision, 4),
                recall=round(m.recall, 4),
                mrr=round(m.mrr, 4),
                ndcg=round(m.ndcg, 4),
                hit_rate=round(m.hit_rate, 4),
            )
            for label, m in report.metrics.items()
        },
        generation=GenerationMetricsView(
            faithfulness=round(generation.faithfulness, 3),
            relevance=round(generation.relevance, 3),
            correctness=round(generation.correctness, 3),
            pass_rate=round(generation.pass_rate, 4),
            grounded_rate=round(generation.grounded_rate, 4),
            judged=generation.judged,
            check=JudgeCheckView(
                judged=generation.check.judged,
                pass_rate=round(generation.check.pass_rate, 4),
                faithfulness=round(generation.check.faithfulness, 3),
                relevance=round(generation.check.relevance, 3),
                correctness=round(generation.check.correctness, 3),
            )
            if generation.check
            else None,
        )
        if generation
        else None,
        examples=[_example_view(example) for example in report.examples],
        usage=usage_view(report.cost),
        generator_model=report.generator_model,
        judge_model=report.judge_model,
        duration_ms=report.duration_ms,
        created_at=report.created_at,
    )


def _example_view(result: ExampleResult) -> ExampleView:
    answer = result.answer
    judgement = answer.judgement if answer else None
    return ExampleView(
        id=result.example.id,
        question=result.example.question,
        expected_answer=result.example.expected_answer,
        relevant=list(result.example.relevant),
        retrieval={
            label: RetrievalScoreView(
                retrieved=score.retrieved,
                precision=round(score.precision, 4),
                recall=round(score.recall, 4),
                reciprocal_rank=round(score.reciprocal_rank, 4),
                ndcg=round(score.ndcg, 4),
                hit=score.hit,
            )
            for label, score in result.retrieval.items()
        },
        answer=AnswerScoreView(
            answer=answer.answer,
            grounded=answer.grounded,
            sources_used=answer.sources_used,
            judgement=JudgementView(
                faithfulness=judgement.faithfulness,
                relevance=judgement.relevance,
                correctness=judgement.correctness,
                passed=judgement.passed,
                rationale=judgement.rationale,
            )
            if judgement
            else None,
        )
        if answer
        else None,
    )


def run_view(run: EvaluationRun) -> EvaluationRunView:
    return EvaluationRunView(
        id=run.id,
        status=run.status,
        done=run.done,
        total=run.total,
        error=run.error,
        created_at=run.created_at,
        finished_at=run.finished_at,
        version=run.version,
        report=report_view(run.report) if run.report else None,
    )


def dataset_view(dataset: EvalDataset) -> DatasetView:
    return DatasetView(
        name=dataset.name,
        description=dataset.description,
        codebases=list(dataset.codebases),
        examples=[
            DatasetExampleView(
                id=example.id,
                question=example.question,
                expected_answer=example.expected_answer,
                relevant=list(example.relevant),
            )
            for example in dataset.examples
        ],
    )
