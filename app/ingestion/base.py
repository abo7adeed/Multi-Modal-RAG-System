from abc import ABC, abstractmethod
from pathlib import Path

from .schemas import Document


class BaseLoader(ABC):

    @abstractmethod
    def load(self, file_path: Path) -> Document:
        """Load a file and convert it into a normalized Document."""
        pass