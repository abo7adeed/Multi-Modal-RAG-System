from pathlib import Path

from PIL import Image


class ImageNormalizer:

    def normalize(self, input_path: Path, output_path: Path) -> Path:
        if not input_path.exists():
            raise FileNotFoundError(f"Image not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(input_path) as image:
            image = image.convert("RGB")
            image.save(output_path, format="JPEG")

        return output_path