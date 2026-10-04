from abc import ABC, abstractmethod

from .schemas import GenerationContext


class BaseGenerator(ABC):

    @abstractmethod
    def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> str:
        """
        Generate a grounded answer from multimodal context.
        """
        raise NotImplementedError