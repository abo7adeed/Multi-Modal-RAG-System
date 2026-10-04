# Multi-Modal RAG System

A production-grade multimodal Retrieval-Augmented Generation system:
ingest PDFs and images, index text and visual content with CLIP
embeddings in ChromaDB, retrieve with Reciprocal Rank Fusion, and
answer questions with grounded, source-backed generation — served
through a FastAPI backend and a professional React frontend.

## Architecture

```
FRONTEND (React + Vite + Tailwind)
        │
        ▼
FASTAPI  (app/api)            ← the only boundary the UI sees
        │
        ▼
RAG PIPELINE (app/pipeline)
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
   INGESTION       RETRIEVAL      GENERATION
  (app/ingestion) (app/retrieval) (app/generation)
   PDF/images  →   CLIP + Chroma   OpenRouter/Ollama
```

Key invariants:

- The frontend talks **only** to the FastAPI API — never to Chroma,
  CLIP, or providers directly.
- The API layer contains **no** RAG business logic; routes delegate
  to `MultimodalRAGPipeline` and application services.
- Raw SDK exceptions never leave the provider modules; the pipeline
  translates everything into application errors (`app/errors.py`)
  with user-facing messages, while technical details go to logs.

## Repository layout

```
app/
├── api/                  # FastAPI layer (routes, schemas, services, DI)
├── ingestion/            # PDF/image loaders, chunking, normalization
├── retrieval/            # CLIP embedder, Chroma store, multimodal retriever
├── generation/           # Context builder, generator, providers
├── config.py             # Environment-based settings (pydantic-settings)
├── errors.py             # Application error taxonomy
└── pipeline.py           # Orchestrating multimodal RAG pipeline
frontend/                 # React + TypeScript + Vite + Tailwind UI
scripts/                  # Indexing and CLI entrypoints
tests/                    # pytest suites (API + pipeline) and manual scripts
```

## Backend API

Base URL: `http://localhost:8000`

| Method | Path                 | Description                          |
| ------ | -------------------- | ------------------------------------ |
| GET    | `/api/v1/health`     | Liveness probe → `{"status":"ok"}`   |
| POST   | `/api/v1/rag/query`  | Ask a question, get grounded answer + sources |
| POST   | `/api/v1/documents`  | Upload PDF/image for ingestion + indexing |
| GET    | `/media/{path}`      | Serves retrieved images (path-safe)  |

Example:

```bash
curl -s http://localhost:8000/api/v1/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What Dell products are shown in the catalog?"}'
```

```json
{
  "answer": "The catalog contains Dell laptops, desktops and accessories.",
  "sources": [
    {
      "chunk_id": "…",
      "page": 4,
      "content_type": "text",
      "source": "dell_catalog.pdf",
      "image_path": null,
      "image_url": null
    },
    {
      "chunk_id": "…",
      "page": 3,
      "content_type": "image",
      "source": "dell_catalog.pdf",
      "image_path": null,
      "image_url": "/media/dell_catalog/page_3_image_0.jpg"
    }
  ]
}
```

Errors return structured payloads with safe user-facing messages:

```json
{
  "error": {
    "message": "Generation service is temporarily rate limited. Please try again in a moment.",
    "type": "ProviderRateLimitError"
  }
}
```

HTTP mapping: `400` invalid input · `422` upload processing failure ·
`429` provider rate limit · `503` retrieval/pipeline unavailable ·
`504` provider timeout · `500` unexpected.

### Asking about your own image

`POST /api/v1/rag/query` also accepts an image with the question:

```bash
curl -s http://localhost:8000/api/v1/rag/query \
  -H 'Content-Type: application/json' \
  -d "{\"query\": \"What device is this?\",
       \"image\": {\"filename\": \"photo.jpg\",
                  \"media_type\": \"image/jpeg\",
                  \"data\": \"$(base64 -w0 photo.jpg)\"}}"
```

The attached image is the *subject* of the question, so it goes
straight to the vision model and the index is only used to add
supporting context: `RETRIEVAL_VISUAL_TOP_K` page images that look
like the upload are retrieved with CLIP's visual tower and sent
alongside it. That means an image question is answerable even when
no document is textually related to it — which is exactly the case
the off-topic gate blocks for plain text questions.

Notes:

- `data` is raw base64 without a `data:` URL prefix, capped at
  `API_MAX_QUERY_IMAGE_SIZE_MB`. The bytes are verified as a real
  image, so a renamed PDF is rejected with `400`.
- The upload is written to a temporary file that is deleted once the
  answer is returned, and its filename is never used to build a path.
- A text-only model (`GENERATION_PROVIDER=opencode`) cannot answer
  these questions; the request is refused rather than guessed.

In the UI, attach an image with the paperclip button or by pasting one
straight into the composer, then ask your question.

## Configuration

All configuration is environment-based (`.env`, see `.env.example`).
Never commit real `.env` files or API keys.

### Choosing a generation provider

Set `GENERATION_PROVIDER` to `openrouter`, `ollama`, or `opencode`
and restart the backend. No code change is required.

| Provider | Rate limits | Multimodal (images) | Notes |
| -------- | ----------- | -------------------- | ----- |
| `openrouter` | `:free` models are heavily rate limited | Yes | Default. Paid models avoid the free-tier limits |
| `ollama` | None (local) or Cloud quotas | Yes (with a vision model) | Works with a local server **or** Ollama Cloud (`https://ollama.com`) |
| `opencode` | None (billed per request) | **No** — text-only | Zen gateway; image context is dropped and logged |

```bash
# Ollama Cloud models (hosted - no local Ollama install needed)
# Create a key at https://ollama.com/settings/keys
GENERATION_PROVIDER=ollama
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your-ollama-cloud-key
OLLAMA_MODEL=gpt-oss:120b-cloud

# Local Ollama (no rate limits, keeps image evidence)
GENERATION_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl:4b          # must be a vision model

# OpenCode Zen (paid, text-only)
GENERATION_PROVIDER=opencode
OPENCODE_API_KEY=...
OPENCODE_MODEL=gpt-5
```

Ollama Cloud is reached over HTTPS with an `Authorization: Bearer`
token, and **images are base64-encoded** before sending, so no local
server and no local model pull is required. Pick a vision-capable
cloud model if you want images to influence the answer.

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `GENERATION_PROVIDER` | `openrouter` | Active generation backend |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `OPENROUTER_MODEL` | `qwen/qwen3.8-27b:free` | OpenRouter model |
| `OPENROUTER_TIMEOUT` / `_MAX_RETRIES` | `120` / `2` | Timeout and retries |
| `OPENCODE_API_KEY` | — | OpenCode Zen API key |
| `OPENCODE_BASE_URL` | `https://opencode.ai/zen/v1` | Zen endpoint |
| `OPENCODE_MODEL` | `gpt-5` | Zen model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local server, or `https://ollama.com` for Ollama Cloud |
| `OLLAMA_API_KEY` | — | Ollama Cloud API key (local needs none) |
| `OLLAMA_MODEL` | `qwen3-vl:4b` | Ollama model (vision model to use images) |
| `OLLAMA_TIMEOUT` | `300` | Ollama request timeout |
| `API_CORS_ORIGINS` | localhost dev origins | Comma-separated CORS allowlist |
| `API_MAX_UPLOAD_SIZE_MB` | `25` | Upload size limit |
| `API_MAX_QUERY_IMAGE_SIZE_MB` | `5` | Size limit for an image attached to a question |
| `API_MAX_SOURCES` | `1` | Sources returned per answer |
| `RETRIEVAL_VISUAL_TOP_K` | `2` | Similar page images sent with an attached image |
| `PROCESSED_IMAGES_DIRECTORY` | `data/processed` | Media root served under `/media` |
| `VITE_API_BASE_URL` (frontend) | `http://localhost:8000` | API base URL for the UI |

## Running

### 1. Index a document (one-time, CLI)

```bash
uv run scripts/index_catalog.py
```

### 2. Start the backend

```bash
uv run python -m app.api.main
# API on http://localhost:8000 (docs at /docs)
```

### 3. Start the frontend

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_BASE_URL if needed
npm run dev            # UI on http://localhost:5173
```

## Testing

Backend (unit/API tests, no models or network required):

```bash
uv run python -m pytest tests/test_pipeline.py tests/test_api.py -v
```

Frontend:

```bash
cd frontend
npm run typecheck   # TypeScript
npm run lint        # oxlint
npm run test        # vitest
npm run build       # production build
```

## Design decisions

- **No LangChain/LangGraph.** The custom pipeline is the source of
  truth; the existing ingestion/retrieval logic is untouched.
- **No fake streaming.** The backend currently returns complete
  responses; the UI therefore renders a typing indicator instead of
  pretending to stream.
- **Upload is real.** `POST /api/v1/documents` persists the file and
  runs the existing loaders → chunker → normalizer → CLIP → Chroma
  stack, so uploaded documents become queryable immediately.
- **Media is sandboxed.** Image URLs are resolved and served only
  from the processed-images directory; traversal is rejected.
