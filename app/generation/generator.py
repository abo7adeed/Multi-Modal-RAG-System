from collections.abc import Iterator

from .base import BaseGenerator
from .providers.base import BaseMultimodalProvider
from .schemas import GenerationContext


class MultimodalGenerator(BaseGenerator):

    def __init__(
        self,
        provider: BaseMultimodalProvider,
    ):
        self.provider = provider

    @property
    def supports_images(self) -> bool:
        """
        Whether the configured model can actually read images.

        Text-only gateways silently drop image parts, so callers must
        check before answering a question about an attached image.
        Defaults to True because the flag is optional on the provider
        contract.
        """
        return bool(
            getattr(self.provider, "supports_images", True)
        )

    def generate(
        self,
        query: str,
        context: GenerationContext,
    ) -> str:

        return self.provider.generate(
            query=query,
            text_context=context.text_context,
            image_context=context.image_context,
            conversation=context.conversation,
        )

    def stream(
        self,
        query: str,
        context: GenerationContext,
    ) -> Iterator[str]:
        """
        Yield the answer incrementally.

        Mirrors generate() so both paths build the same context and
        hand it to the same provider contract; the provider decides
        whether tokens actually arrive one by one.
        """

        return self.provider.stream(
            query=query,
            text_context=context.text_context,
            image_context=context.image_context,
            conversation=context.conversation,
        )