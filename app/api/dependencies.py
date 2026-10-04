"""
API dependency wiring.

Heavy components (CLIP embedder, Chroma store, pipeline) are created
once at startup in the FastAPI lifespan and exposed to routes through
fastapi.Depends - no ad-hoc global mutable state.
"""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request

from app.api.services import (
    AttachmentService,
    DocumentIngestionService,
    MediaService,
)
from app.config import settings
from app.generation.context_builder import ContextBuilder
from app.generation.generator import MultimodalGenerator
from app.generation.providers import build_provider
from app.pipeline import MultimodalRAGPipeline
from app.retrieval.embeddings.clip_embedder import CLIPEmbedder
from app.retrieval.retriever import MultimodalRetriever
from app.retrieval.vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    pipeline: MultimodalRAGPipeline
    ingestion_service: DocumentIngestionService
    media_service: MediaService
    attachment_service: AttachmentService = field(
        default_factory=AttachmentService
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("startup: loading retrieval and generation components")

    embedder = CLIPEmbedder()

    vector_store = ChromaVectorStore(
        persist_directory=settings.vector_store_directory,
        collection_name="dell_catalog",
    )

    retriever = MultimodalRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    provider = build_provider()

    pipeline = MultimodalRAGPipeline(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generator=MultimodalGenerator(provider=provider),
    )

    media_service = MediaService()

    ingestion_service = DocumentIngestionService(retriever=retriever)

    app.state.rag = AppState(
        pipeline=pipeline,
        ingestion_service=ingestion_service,
        media_service=media_service,
        attachment_service=AttachmentService(),
    )

    provider_info = pipeline.describe_provider()
    logger.info(
        "startup: complete provider=%s model=%s",
        provider_info.get("provider"),
        provider_info.get("model"),
    )

    yield

    logger.info("shutdown: releasing RAG components")


def get_rag_state(request: Request) -> AppState:
    """FastAPI dependency exposing the lifespan-built components."""
    return request.app.state.rag
