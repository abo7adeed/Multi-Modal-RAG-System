# Backend image.
#
# Multi-stage so the runtime layer carries no compiler toolchain:
# only the virtual environment and the application source.
#
# CLIP weights (~350 MB) are downloaded on first startup and cached
# in the HF cache. Mount a volume there to avoid re-downloading on
# every rebuild.
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies are installed from the lockfile before the source is
# copied, so editing application code does not invalidate the layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app
COPY eval ./eval
COPY scripts ./scripts
RUN uv sync --frozen --no-dev


FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    # Chroma writes to the vector store directory; uploads and
    # processed images are mounted volumes in compose.
    VECTOR_STORE_DIRECTORY=/data/vector_store \
    UPLOADS_DIRECTORY=/data/uploads \
    PROCESSED_IMAGES_DIRECTORY=/data/processed/images

# curl is used by the compose healthcheck; nothing else is installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app /app

# The image runs unprivileged: a RAG service processes untrusted
# uploads, so it should not be root.
RUN useradd --create-home --uid 10001 raguser \
    && mkdir -p /data /cache \
    && chown -R raguser:raguser /app /data /cache

USER raguser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]