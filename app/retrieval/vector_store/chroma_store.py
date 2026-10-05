from pathlib import Path
from typing import Any

import chromadb


class ChromaVectorStore:

    def __init__(
        self,
        persist_directory: str = "data/vector_store",
        collection_name: str = "multimodal_chunks",
    ):
        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

    def add(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ):
        if not chunk_ids:
            return

        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        content_type: str | None = None,
    ):
        where = None

        if content_type:
            where = {
                "content_type": content_type
            }

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    def count(self) -> int:
        return self.collection.count()

    def delete_where(self, where: dict[str, Any]) -> int:
        """
        Delete every chunk matching a metadata filter.

        Returns the number of chunks removed.

        Chroma requires the ids, so they are looked up first: deleting
        a whole document is by definition a bulk operation, and the
        filter is built server-side rather than by scanning every
        chunk in Python.
        """
        matches = self.collection.get(
            where=where,
            include=[],
        )
        ids = matches.get("ids") or []

        if not ids:
            return 0

        self.collection.delete(ids=ids)
        return len(ids)

    def list_metadata(self, where: dict[str, Any] | None = None):
        """Return stored metadata, for tooling and diagnostics."""
        return self.collection.get(
            where=where,
            include=["metadatas"],
        )