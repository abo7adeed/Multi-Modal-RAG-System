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

| Method | Path                    | Description                                  |
| ------ | ----------------------- | -------------------------------------------- |
| GET    | `/api/v1/health`        | Liveness probe → `{"status":"ok"}`           |
| POST   | `/api/v1/rag/query`     | Buffered: ask a question, get answer + sources |
| POST   | `/api/v1/rag/stream`    | SSE: the same answer, token by token          |
| POST   | `/api/v1/documents`     | Upload PDF/image for ingestion + indexing     |
| DELETE | `/api/v1/documents/{id}`| Remove a document and all its chunks from the index |
| GET    | `/media/{path}`         | Serves retrieved images (path-safe)           |

Every response carries an `X-Request-ID` header; send your own to
have it echoed, and it appears on every log line for that request.

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
      "image_url": null,
      "snippet": "Dell Precision laptop with Intel Core i7.",
      "score": 0.0164,
      "kind": "document"
    },
    {
      "chunk_id": "…",
      "page": 3,
      "content_type": "image",
      "source": "dell_catalog.pdf",
      "image_path": null,
      "image_url": "/media/dell_catalog/page_3_image_0.jpg",
      "score": 0.0161,
      "kind": "document"
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

### Streaming answers

`POST /api/v1/rag/stream` answers with Server-Sent Events, so the UI
renders text as it is generated instead of waiting for the whole
reply:

```bash
curl -N http://localhost:8000/api/v1/rag/stream \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the Dell OptiPlex 3020?"}'
```

```
event: sources
data: {"sources":[{"chunk_id":"…","page":13,"kind":"document","score":0.0164,…}]}

event: token
data: {"text":"The OptiPlex 3020 Micro"}

event: token
data: {"text":" is a space-saving design"}

event: done
data: {}
```

The wire format:

| Event     | When                | Meaning                                    |
| --------- | ------------------- | ------------------------------------------ |
| `sources` | once, first         | The citations, before any token            |
| `token`   | many                | Append `text` to what you already have     |
| `done`    | once, last          | The answer finished                        |
| `error`   | on mid-stream failure | Carries the same `{message, type}` as an HTTP error |

Rules worth knowing when writing a client:

- **A missing `done` means failure.** A stream that simply stops was
  truncated; treat the answer as incomplete rather than finished.
- **Validation and retrieval still use real HTTP status codes.**
  A blank query returns `400`, not a `200` carrying an `error` event,
  because those run before the response body starts.
- **Sources come first** so they can be rendered while the text is
  still being written.
- Payloads are JSON-encoded, so a token containing a newline cannot
  break the framing.

If you put nginx (or any proxy) in front, turn buffering off —
`proxy_buffering off;` — or it will hold the entire answer before
forwarding a byte. `frontend/nginx.conf` is configured this way.

### Follow-up questions

Send the recent conversation with your request and the server resolves
references the retriever cannot:

```bash
curl -s http://localhost:8000/api/v1/rag/stream \
  -H 'Content-Type: application/json' \
  -d '{"query": "What about its warranty?",
       "history": [{"role": "user", "content": "What is the Dell OptiPlex 3020?"}]}'
```

"How much memory does **it** have?" contains no subject the index can
match. Before retrieval the query is resolved against the previous
*question*, so the search sees something standalone:

| You asked                              | Retrieval actually searches for                                        |
| -------------------------------------- | --------------------------------------------------------------------- |
| "What about its warranty?"             | "What is the Dell OptiPlex 3020? What about its warranty?"            |

Three deliberate constraints:

- **The model still receives your own wording**, plus the earlier turns
  as context. Only retrieval uses the rewritten form, so the answer is
  about what you asked.
- **Only queries that look like a follow-up are rewritten** — a vague
  phrase, a pronoun, or too few words to name a subject. A query that
  names its subject is never touched, so a wrong guess cannot replace
  your question.
- **Only earlier *questions* are used**, never earlier answers. Folding
  model-written text into the search would let the model bias its own
  retrieval.

History is capped at 6 turns and 2000 characters, and `system` turns
are rejected so a client cannot inject instructions. The API keeps no
session state; the client owns the transcript.

### Refusing off-topic questions

A question that shares too little with the index is answered *without*
calling the model:

> I don't have enough information in the provided documents.

The share of a query's rarity-weighted content terms that must be
present is `RETRIEVAL_LEXICAL_MIN_COVERAGE` (default `0.30`). This is
what keeps a grounded system from drifting into general chat.

It does not apply when you attach an image — the picture is evidence in
itself, so the gate must not end the request.

Measured on a 25-query labelled set, see [`eval/BASELINE.md`](eval/BASELINE.md):
recall@5 is unaffected at 0.900 while false answers drop from 0.80 to
0.60. Answerable and off-topic queries overlap in coverage, so the
remaining gap needs a semantic judge, not a better threshold.

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
| `API_MAX_SOURCES` | `3` | Ranked sources returned per answer (1-10 per request) |
| `API_KEY` | — (disabled) | Shared secret for the `X-API-Key` header |
| `API_RATE_LIMIT_REQUESTS` | `30` | Requests allowed per client IP… |
| `API_RATE_LIMIT_WINDOW_SECONDS` | `60` | …within this sliding window |
| `RETRIEVAL_VISUAL_TOP_K` | `2` | Similar page images sent with an attached image |
| `RETRIEVAL_LEXICAL_MIN_COVERAGE` | `0.30` | Below this, a text question is treated as off-topic |
| `PROCESSED_IMAGES_DIRECTORY` | `data/processed` | Media root served under `/media` |
| `VITE_API_BASE_URL` (frontend) | `http://localhost:8000` | API base URL for the UI |

## Running

### With Docker

```bash
cp .env.example .env    # set your provider API key - required
docker compose up --build
# UI on http://localhost:5173, API on http://localhost:8000
```

The stack runs the backend, a built frontend behind nginx, and an
Ollama server. To use Ollama Cloud instead, drop the `ollama` service
and set `OLLAMA_BASE_URL=https://ollama.com` with a key. The
vector store, uploads, extracted images and the CLIP weight cache are
all volumes, so rebuilds keep your data.

### Locally

```bash
# 1. Index a document (one-time, CLI)
uv run scripts/index_catalog.py

# 2. Start the backend
uv run python -m app.api.main
# API on http://localhost:8000 (docs at /docs)

# 3. Start the frontend
cd frontend
npm install
cp .env.example .env   # set VITE_API_BASE_URL if needed
npm run dev            # UI on http://localhost:5173
```

The backend **refuses to start** when the selected provider has no API
key. A missing key would otherwise look like a provider outage on
every request.

### Evaluating retrieval quality

```bash
uv run python scripts/evaluate_retrieval.py --k 5 --verbose
```

Runs the labelled query set in `eval/queries.jsonl` against the real
index and reports recall@k, MRR and the off-topic false-answer rate.
`eval/BASELINE.md` records the numbers before and after each change.

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
- **Real streaming, honestly degraded.** `/api/v1/rag/stream` emits
  real provider tokens. A provider with no streaming support falls
  back to one chunk rather than failing, so the wire format is always
  the same — and a stream that stops without `done` is reported as
  truncated instead of being rendered as a complete answer.
- **Upload is real.** `POST /api/v1/documents` persists the file and
  runs the existing loaders → chunker → normalizer → CLIP → Chroma
  stack, so uploaded documents become queryable immediately.
- **Media is sandboxed.** Image URLs are resolved and served only
  from the processed-images directory; traversal is rejected.
- **Blocking work never runs on the event loop.** CLIP embedding and
  generation are synchronous and slow; the routes that call them are
  plain `def`, so FastAPI runs them in its threadpool and concurrent
  queries overlap instead of queueing. `tests/test_concurrency.py`
  fails if that regresses.
- **The retrieval gate is measured, not guessed.** Threshold changes
  are justified by `eval/BASELINE.md`, and the limitation is written
  down there rather than left implicit.
