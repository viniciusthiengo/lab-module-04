from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.llm.base import LLMError
from app.llm.fake import FakeLLMClient, FakeRequest, ScriptedTurn
from app.outputs import Answer, Verdict
from tests.conftest import HeldExecutor
from tests.fakes import ScriptedClaude

CART = "def total(cart):\n    return sum(book.price for book in cart)\n"
QUESTION = "How are passwords hashed?"


def index(client: TestClient, codebase: str = "shop", **overrides: Any):
    body = {"codebase": codebase, "files": [{"path": "cart.py", "content": CART}], **overrides}
    return client.post("/index/files", json=body)


class TestHealthAndCodebases:
    def test_health_reports_models_index_caches_and_spending(self, client: TestClient) -> None:
        body = client.get("/health").json()

        assert body["status"] == "ok"
        assert (body["ready"], body["model"], body["judge_model"]) == (
            True,
            "fake-model",
            "fake-model",
        )
        assert body["embedding_model"] == "hash-embedder"
        assert (body["codebases"], body["chunks"]) == (2, 77)
        assert body["spent_today_usd"] == 0
        assert "key" not in str(body).lower()

    def test_the_demo_codebases_are_indexed_at_startup(self, client: TestClient) -> None:
        codebases = client.get("/codebases").json()

        assert [(c["name"], c["demo"], len(c["files"])) for c in codebases] == [
            ("bookshop-api", True, 13),
            ("bookshop-web", True, 7),
        ]
        api = codebases[0]
        assert api["languages"] == ["markdown", "python", "text"]
        assert {"path": "bookshop/auth.py", "language": "python", "lines": 54, "chunks": 6} in (
            api["files"]
        )

    def test_without_the_demo_nothing_is_indexed(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(demo=False)

        assert client.get("/codebases").json() == []
        response = client.post("/query", json={"question": QUESTION})
        assert response.status_code == 409
        assert response.json()["detail"] == "No codebase is indexed yet. Index a codebase first."


class TestIndexing:
    def test_indexes_files_into_a_new_codebase(self, client: TestClient) -> None:
        response = index(client)

        assert response.status_code == 200
        body = response.json()
        assert (body["files"], body["chunks"], body["embedded"], body["reused"]) == (1, 1, 1, 0)
        assert body["kinds"] == {"function": 1}
        assert body["codebase"]["name"] == "shop"
        assert body["codebase"]["demo"] is False
        assert body["codebase"]["files"] == [
            {"path": "cart.py", "language": "python", "lines": 2, "chunks": 1}
        ]
        assert [c["name"] for c in client.get("/codebases").json()] == [
            "bookshop-api",
            "bookshop-web",
            "shop",
        ]

    def test_reindexing_reuses_unchanged_chunks(self, client: TestClient) -> None:
        index(client)

        body = index(client).json()

        assert (body["embedded"], body["reused"]) == (0, 1)

    def test_an_upload_in_several_requests_replaces_the_codebase(self, client: TestClient) -> None:
        old = {"path": "old.py", "content": CART}
        index(client, files=[old, {"path": "cart.py", "content": CART}])
        index(client, files=[{"path": "cart.py", "content": CART}])

        body = index(
            client,
            files=[{"path": "auth.py", "content": "def login():\n    pass\n"}],
            replace=True,
            keep=["./cart.py"],
        ).json()

        assert [file["path"] for file in body["codebase"]["files"]] == ["auth.py", "cart.py"]
        assert body["removed"] == 1

    def test_paths_are_normalized(self, client: TestClient) -> None:
        response = index(client, files=[{"path": "./src\\cart.py", "content": CART}])

        assert response.json()["codebase"]["files"][0]["path"] == "src/cart.py"

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ({"codebase": "Shop!"}, "A codebase name has lower-case letters"),
            ({"files": []}, "Add at least one file."),
            (
                {"files": [{"path": "logo.png", "content": "x"}]},
                "'logo.png' is not a supported file type.",
            ),
            (
                {"files": [{"path": "../etc/app.py", "content": "x"}]},
                "The path '../etc/app.py' must not have empty, '.' or '..' parts.",
            ),
            (
                {"files": [{"path": "/etc/app.py", "content": "x"}]},
                "The path '/etc/app.py' must be relative.",
            ),
            ({"files": [{"path": "a.py", "content": "\0"}]}, "Binary files cannot be indexed."),
            (
                {"files": [{"path": "a.py", "content": "x"}, {"path": "a.py", "content": "y"}]},
                "Two files have the same path: a.py.",
            ),
            ({"files": [{"path": f"f{n}.py", "content": "x"} for n in range(201)]}, "at most 200"),
            ({"extra": 1}, "Unknown field 'extra'."),
        ],
    )
    def test_invalid_requests_are_refused(
        self, client: TestClient, body: dict[str, Any], message: str
    ) -> None:
        payload = {"codebase": "shop", "files": [{"path": "cart.py", "content": CART}], **body}

        response = client.post("/index/files", json=payload)

        assert response.status_code == 422
        assert message in response.json()["detail"]

    def test_missing_body(self, client: TestClient) -> None:
        response = client.post("/index/files")

        assert (response.status_code, response.json()) == (422, {"detail": "Send a JSON body."})

    def test_the_demo_codebases_are_read_only(self, client: TestClient) -> None:
        indexed = index(client, "bookshop-api")
        deleted = client.delete("/codebases/bookshop-web")

        assert (indexed.status_code, deleted.status_code) == (409, 409)
        assert "demo codebase" in indexed.json()["detail"]
        assert len(client.get("/codebases").json()) == 2

    def test_delete_a_codebase(self, client: TestClient) -> None:
        index(client)

        assert client.delete("/codebases/shop").status_code == 204
        missing = client.delete("/codebases/shop")
        assert (missing.status_code, missing.json()["detail"]) == (
            404,
            "No codebase named 'shop' is indexed.",
        )
        assert client.delete("/codebases/Bad Name").status_code == 422

    def test_the_index_has_a_size_limit(self, make_client: Callable[..., TestClient]) -> None:
        client = make_client(max_codebases=2)

        response = index(client)

        assert response.status_code == 409
        assert response.json()["detail"] == "The index holds 2 codebases at most; delete one first."

    def test_index_demo_again_reuses_every_chunk(self, client: TestClient) -> None:
        body = client.post("/index/demo").json()

        assert [(r["codebase"]["name"], r["embedded"], r["reused"]) for r in body] == [
            ("bookshop-api", 0, 57),
            ("bookshop-web", 0, 20),
        ]


class TestSearch:
    def test_finds_code_without_the_model(self, make_client: Callable[..., TestClient]) -> None:
        client = make_client(llm=None)

        response = client.post("/search", json={"question": QUESTION, "mode": "bm25", "k": 3})

        assert response.status_code == 200
        body = response.json()
        assert body["hits"][0]["key"] == "bookshop-api:bookshop/auth.py::hash_password"
        assert body["hits"][0]["content"].startswith("def hash_password")
        assert (body["mode"], body["rerank"], body["k"], len(body["hits"])) == ("bm25", False, 3, 3)
        assert body["usage"]["model_calls"] == 0

    def test_reranking_needs_the_api_key(self, make_client: Callable[..., TestClient]) -> None:
        client = make_client(llm=None)

        response = client.post("/search", json={"question": QUESTION, "rerank": True})

        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]

    def test_search_some_codebases(self, client: TestClient) -> None:
        body = client.post(
            "/search", json={"question": "access token", "codebases": ["bookshop-web"], "k": 10}
        ).json()

        assert body["codebases"] == ["bookshop-web"]
        assert {hit["codebase"] for hit in body["hits"]} == {"bookshop-web"}

    @pytest.mark.parametrize(
        ("body", "status", "message"),
        [
            ({"question": "  "}, 422, "The question is empty."),
            ({"question": "x" * 1001}, 422, "'question' is too long (1000 characters at most)."),
            ({"question": "q", "k": 11}, 422, "'k' input should be less than or equal to 10."),
            ({"question": "q", "mode": "fuzzy"}, 422, "Invalid value for 'mode'"),
            ({"question": "q", "codebases": ["Nope!"]}, 422, "A codebase name has"),
            ({"question": "q", "codebases": ["shop"]}, 409, "Not indexed: shop."),
        ],
    )
    def test_invalid_searches(
        self, client: TestClient, body: dict[str, Any], status: int, message: str
    ) -> None:
        response = client.post("/search", json=body)

        assert response.status_code == status
        assert message in response.json()["detail"]


class TestQuery:
    def test_answers_with_the_sources_it_cites(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": QUESTION})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "It is done in `hash_password` (bookshop/auth.py) [1]."
        assert (body["grounded"], body["sources_used"], body["cached"]) == (True, [1], False)
        assert body["sources"][0]["name"] == "hash_password"
        assert body["sources"][0]["relevance"] == 3
        assert (body["mode"], body["rerank"], body["k"]) == ("hybrid", True, 5)
        assert body["usage"]["model_calls"] == 2
        assert body["usage"]["models"] == ["fake-model"]
        # The fake model is unknown, so it is priced as the most expensive model.
        assert body["usage"]["cost_usd"] == pytest.approx(2 * (1000 * 5 + 400 * 25) / 1e6)

    def test_a_repeated_question_comes_from_the_cache(self, client: TestClient) -> None:
        client.post("/query", json={"question": QUESTION})

        body = client.post("/query", json={"question": QUESTION}).json()

        assert body["cached"] is True
        assert body["usage"]["model_calls"] == 0
        assert client.get("/health").json()["query_cache"] == {"hits": 1, "misses": 1}

    def test_the_conversation_is_sent_to_the_model(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        llm = FakeLLMClient(ScriptedClaude())
        client = make_client(llm=llm)
        history = [
            {"role": "user", "content": "What does the cart do?"},
            {"role": "assistant", "content": "It holds books [1]."},
        ]

        client.post("/query", json={"question": "And coupons?", "history": history})

        prompt = llm.calls_for(Answer)[0].prompt
        assert "User: What does the cart do?\nAssistant: It holds books [1]." in prompt

    def test_without_the_api_key(self, make_client: Callable[..., TestClient]) -> None:
        client = make_client(llm=None)

        response = client.post("/query", json={"question": QUESTION})

        assert response.status_code == 503
        assert client.get("/health").json()["ready"] is False

    @pytest.mark.parametrize(
        ("error", "status"),
        [
            (LLMError("The AI provider is busy (rate limited).", retryable=True), 503),
            (LLMError("The AI provider rejected the API key."), 502),
        ],
    )
    def test_provider_failures(
        self, make_client: Callable[..., TestClient], error: LLMError, status: int
    ) -> None:
        def failing(_: FakeRequest) -> ScriptedTurn:
            raise error

        client = make_client(llm=FakeLLMClient(failing))

        response = client.post("/query", json={"question": QUESTION})

        assert (response.status_code, response.json()["detail"]) == (status, str(error))

    def test_the_daily_budget_stops_model_calls(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(daily_budget_usd=0.02)  # a query costs 0.03: two calls

        first = client.post("/query", json={"question": QUESTION})
        second = client.post("/query", json={"question": "How are carts priced?"})

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["detail"] == (
            "Today's budget for AI calls is used up; try again tomorrow."
        )
        assert client.get("/health").json()["spent_today_usd"] == pytest.approx(0.03)

    def test_rate_limit_per_client(self, make_client: Callable[..., TestClient]) -> None:
        now = [0.0]
        client = make_client(queries_per_hour=2, rate_clock=lambda: now[0])
        headers = {"x-client-ip": "203.0.113.7"}

        statuses = [
            client.post("/search", json={"question": "cart"}, headers=headers).status_code
            for _ in range(3)
        ]
        other = client.post("/search", json={"question": "cart"}, headers={"x-client-ip": "x"})
        now[0] = 3600.0
        later = client.post("/search", json={"question": "cart"}, headers=headers)

        assert statuses == [200, 200, 429]
        assert other.status_code == 200
        assert later.status_code == 200


class TestEvaluation:
    def test_the_dataset(self, client: TestClient) -> None:
        body = client.get("/evaluation/dataset").json()

        assert body["name"] == "bookshop"
        assert body["codebases"] == ["bookshop-api", "bookshop-web"]
        assert len(body["examples"]) >= 10
        assert set(body["examples"][0]) == {"id", "question", "expected_answer", "relevant"}

    def test_evaluate_answers_with_the_report(self, client: TestClient) -> None:
        response = client.post("/evaluate", json={})

        assert response.status_code == 200
        body = response.json()
        assert set(body["metrics"]) == {"vector", "bm25", "hybrid", "hybrid+rerank"}
        assert set(body["metrics"]["hybrid"]) == {"precision", "recall", "mrr", "ndcg", "hit_rate"}
        assert body["generation"]["judged"] == len(body["examples"])
        example = body["examples"][0]
        assert set(example["retrieval"]) == set(body["metrics"])
        assert example["answer"]["judgement"]["passed"] is True
        assert body["usage"]["model_calls"] == 3 * len(body["examples"])

    def test_the_judge_check(self, client: TestClient) -> None:
        body = client.post(
            "/evaluate",
            json={"settings": ["hybrid"], "answer_setting": "hybrid", "judge_check": True},
        ).json()

        check = body["generation"]["check"]
        assert check["judged"] == len(body["examples"])
        assert set(check) == {"judged", "pass_rate", "faithfulness", "relevance", "correctness"}
        assert body["usage"]["model_calls"] == 3 * len(body["examples"])  # answer, judge, check

    def test_retrieval_metrics_only(self, make_client: Callable[..., TestClient]) -> None:
        client = make_client(llm=None)

        response = client.post(
            "/evaluate",
            json={"k": 3, "settings": ["vector", "bm25", "hybrid"], "answer_setting": None},
        )

        assert response.status_code == 200
        body = response.json()
        assert (body["k"], body["generation"], body["usage"]["model_calls"]) == (3, None, 0)

    def test_answers_and_reranking_need_the_api_key(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(llm=None)

        assert client.post("/evaluate", json={}).status_code == 503
        assert client.post("/evaluations", json={"answer_setting": None}).status_code == 503

    def test_a_background_evaluation(self, client: TestClient) -> None:
        started = client.post("/evaluations", json={"settings": ["bm25"], "answer_setting": None})

        assert started.status_code == 202
        run_id = started.json()["id"]
        run = client.get(f"/evaluations/{run_id}").json()
        assert (run["status"], run["done"], run["total"]) == ("completed", 18, 18)
        assert run["report"]["settings"] == ["bm25"]
        assert client.get("/evaluations/latest").json()["id"] == run_id

    def test_the_latest_evaluation_survives_a_restart(
        self, make_client: Callable[..., TestClient], tmp_path: Path
    ) -> None:
        body = {"settings": ["bm25"], "answer_setting": None}
        first = make_client(data_dir=tmp_path)
        run_id = first.post("/evaluations", json=body).json()["id"]

        restarted = make_client(data_dir=tmp_path)

        latest = restarted.get("/evaluations/latest").json()
        assert (latest["id"], latest["status"]) == (run_id, "completed")
        assert (
            latest["report"]["metrics"]["bm25"]
            == first.get(f"/evaluations/{run_id}").json()["report"]["metrics"]["bm25"]
        )

    def test_long_polling_answers_when_the_run_changed(self, client: TestClient) -> None:
        run_id = client.post("/evaluations", json={"answer_setting": None}).json()["id"]

        run = client.get(f"/evaluations/{run_id}", params={"since": 0, "wait": 5}).json()

        assert run["status"] == "completed"
        assert run["version"] > 0

    def test_long_polling_waits_at_most_wait_seconds(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(runner=HeldExecutor())
        run_id = client.post("/evaluations", json={"answer_setting": None}).json()["id"]

        run = client.get(f"/evaluations/{run_id}", params={"since": 0, "wait": 0.3}).json()

        assert (run["status"], run["version"], run["report"]) == ("running", 0, None)

    def test_two_running_evaluations_at_most(self, make_client: Callable[..., TestClient]) -> None:
        client = make_client(runner=HeldExecutor())
        body = {"settings": ["bm25"], "answer_setting": None}

        statuses = [client.post("/evaluations", json=body).status_code for _ in range(3)]

        assert statuses == [202, 202, 503]

    def test_unknown_runs(self, client: TestClient) -> None:
        assert client.get("/evaluations/latest").status_code == 404
        missing = client.get("/evaluations/abcdefghijkl")
        assert (missing.status_code, missing.json()["detail"]) == (
            404,
            "Evaluation run not found.",
        )
        assert client.get("/evaluations/bad!id").status_code == 422

    def test_a_failed_run_reports_its_error(self, make_client: Callable[..., TestClient]) -> None:
        def judge_fails(request: FakeRequest) -> ScriptedTurn:
            if request.output_type is Verdict:
                raise LLMError("The model declined this request.")
            return ScriptedClaude()(request)

        client = make_client(llm=FakeLLMClient(judge_fails))

        run_id = client.post("/evaluations", json={"settings": ["bm25"]}).json()["id"]
        run = client.get(f"/evaluations/{run_id}").json()

        assert (run["status"], run["error"]) == ("failed", "The model declined this request.")

    def test_evaluations_are_rate_limited(self, make_client: Callable[..., TestClient]) -> None:
        client = make_client(evaluations_per_hour=1)
        body = {"settings": ["bm25"], "answer_setting": None}

        assert client.post("/evaluate", json=body).status_code == 200
        assert client.post("/evaluate", json=body).status_code == 429

    def test_invalid_settings(self, client: TestClient) -> None:
        response = client.post("/evaluate", json={"settings": ["fuzzy"]})

        assert response.status_code == 422
        assert response.json()["detail"].startswith("Invalid value for 'settings.0'")


def test_cors_is_off_unless_configured(make_client: Callable[..., TestClient]) -> None:
    closed = make_client()
    open_ = make_client(cors_origins=("https://rag.example",))
    headers = {"Origin": "https://rag.example", "Access-Control-Request-Method": "POST"}

    assert "access-control-allow-origin" not in closed.options("/query", headers=headers).headers
    allowed = open_.options("/query", headers=headers)
    assert allowed.headers["access-control-allow-origin"] == "https://rag.example"


def test_the_api_is_documented(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "Codebase RAG API"
    assert {"/index/files", "/query", "/evaluate", "/search", "/evaluations/{run_id}"} <= set(
        schema["paths"]
    )
    assert "409" in schema["paths"]["/query"]["post"]["responses"]
