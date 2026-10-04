from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store vectors with their documents and metadata."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search for the most similar vectors."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""
        pass