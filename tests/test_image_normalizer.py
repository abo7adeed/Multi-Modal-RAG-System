from pathlib import Path

from app.ingestion.image_normalizer import ImageNormalizer


def main():
    input_path = Path(
        "data/processed/images/dell_catalog/page_17_image_0.jp2"
    )

    output_path = Path(
        "data/processed/normalized/page_17_image_0.jpg"
    )

    normalizer = ImageNormalizer()

    result = normalizer.normalize(
        input_path,
        output_path,
    )

    print("=" * 50)
    print("IMAGE NORMALIZATION TEST")
    print("=" * 50)

    print(f"Input : {input_path}")
    print(f"Output: {result}")
    print(f"Exists: {result.exists()}")

    print("=" * 50)


if __name__ == "__main__":
    main()