from pathlib import Path

from PIL import Image


def main():
    image_path = Path(
        "data/processed/normalized/page_17_image_0.jpg"
    )

    with Image.open(image_path) as image:
        print("=" * 50)
        print("NORMALIZED IMAGE TEST")
        print("=" * 50)

        print(f"Path   : {image_path}")
        print(f"Format : {image.format}")
        print(f"Mode   : {image.mode}")
        print(f"Size   : {image.size}")
        print(f"Width  : {image.width}")
        print(f"Height : {image.height}")

        print("=" * 50)


if __name__ == "__main__":
    main()