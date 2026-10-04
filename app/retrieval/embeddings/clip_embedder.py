from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:

    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading CLIP on: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(
            self.MODEL_NAME
        )

        self.model = CLIPModel.from_pretrained(
            self.MODEL_NAME
        ).to(self.device)

        self.model.eval()

    # --------------------------------------------------
    # Text
    # --------------------------------------------------

    def embed_text(self, text: str) -> list[float]:

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = self._move_to_device(inputs)

        with torch.no_grad():
            output = self.model.get_text_features(
                **inputs
            )

        features = self._extract_features(output)

        features = self._normalize(features)

        return features[0].cpu().tolist()

    # --------------------------------------------------
    # Image
    # --------------------------------------------------

    def embed_image(
        self,
        image_path: str,
    ) -> list[float]:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(path) as image:
            image = image.convert("RGB")

            inputs = self.processor(
                images=image,
                return_tensors="pt",
            )

        inputs = self._move_to_device(inputs)

        with torch.no_grad():
            output = self.model.get_image_features(
                **inputs
            )

        features = self._extract_features(output)

        features = self._normalize(features)

        return features[0].cpu().tolist()

    # --------------------------------------------------
    # Batch text
    # --------------------------------------------------

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = self._move_to_device(inputs)

        with torch.no_grad():
            output = self.model.get_text_features(
                **inputs
            )

        features = self._extract_features(output)

        features = self._normalize(features)

        return features.cpu().tolist()

    # --------------------------------------------------
    # Batch images
    # --------------------------------------------------

    def embed_images(
        self,
        image_paths: list[str],
    ) -> list[list[float]]:

        if not image_paths:
            return []

        images = []

        for image_path in image_paths:

            path = Path(image_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            with Image.open(path) as image:
                images.append(
                    image.convert("RGB")
                )

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        inputs = self._move_to_device(inputs)

        with torch.no_grad():
            output = self.model.get_image_features(
                **inputs
            )

        features = self._extract_features(output)

        features = self._normalize(features)

        return features.cpu().tolist()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _extract_features(self, output):

        if isinstance(output, torch.Tensor):
            return output

        if hasattr(output, "pooler_output"):
            return output.pooler_output

        raise TypeError(
            f"Unexpected CLIP output type: {type(output)}"
        )

    def _normalize(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        return features / features.norm(
            dim=-1,
            keepdim=True,
        )

    def _move_to_device(self, inputs):

        return {
            key: value.to(self.device)
            for key, value in inputs.items()
        }