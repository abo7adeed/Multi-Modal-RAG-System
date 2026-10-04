from pathlib import Path

from app.ingestion.image_normalizer import ImageNormalizer


INPUT_DIR = Path("data/processed/images/dell_catalog")
OUTPUT_DIR = Path("data/processed/normalized/dell_catalog")

SUPPORTED_EXTENSIONS = {".jp2", ".png", ".jpg", ".jpeg"}


def main():
    normalizer = ImageNormalizer()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print("=" * 60)
    print("IMAGE NORMALIZATION")
    print("=" * 60)
    print(f"Input directory : {INPUT_DIR}")
    print(f"Images found    : {len(image_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    normalized_count = 0

    for image_path in image_files:
        output_path = OUTPUT_DIR / f"{image_path.stem}.jpg"

        normalizer.normalize(
            input_path=image_path,
            output_path=output_path,
        )

        normalized_count += 1

        print(f"[OK] {image_path.name} -> {output_path.name}")

    print("=" * 60)
    print(f"Normalized images: {normalized_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()