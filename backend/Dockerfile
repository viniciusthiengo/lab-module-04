# Codebase RAG API. Set ANTHROPIC_API_KEY for answers, reranking and judging; indexing and
# search work without it.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_DIR=/opt/models
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# The embedding model (all-MiniLM-L6-v2 in ONNX, 79 MB) is part of the image, so the server
# never downloads it at startup.
COPY app/__init__.py app/embeddings.py ./app/
RUN python -c "import os, pathlib; from app.embeddings import MiniLMEmbedder; \
MiniLMEmbedder(pathlib.Path(os.environ['MODEL_DIR'])).embed(['warm up'])" \
    && rm "$MODEL_DIR"/all-MiniLM-L6-v2/onnx.tar.gz

# Tests and lint: docker build --target test -t rag-test . && docker run --rm rag-test
FROM base AS test
COPY requirements-dev.txt pyproject.toml ./
RUN pip install -r requirements-dev.txt
COPY main.py ./
COPY app ./app
COPY demo ./demo
COPY tests ./tests
CMD ["sh", "-c", "ruff check . && ruff format --check . && pytest"]

FROM base AS runtime
COPY main.py ./
COPY app ./app
COPY demo ./demo
# The index lives in DATA_DIR: mount a volume there to keep it across deploys.
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
USER app
ENV PORT=8000 DATA_DIR=/data
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
