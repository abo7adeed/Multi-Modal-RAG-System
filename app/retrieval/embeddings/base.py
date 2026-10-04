from abc import ABC, abstractmethod


class BaseMultimodalEmbedder(ABC):

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Convert text into an embedding vector."""
        pass

    @abstractmethod
    def embed_image(self, image_path: str) -> list[float]:
        """Convert an image into an embedding vector."""
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts into embedding vectors."""
        pass

    @abstractmethod
    def embed_images(
        self,
        image_paths: list[str],
    ) -> list[list[float]]:
        """Convert multiple images into embedding vectors."""
        pass