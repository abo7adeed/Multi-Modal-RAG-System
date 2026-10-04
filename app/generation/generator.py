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
        )