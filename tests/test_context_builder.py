from app.generation.context_builder import ContextBuilder


def main():

    results = [
        {
            "chunk_id": "text-001",
            "document": "Dell Precision laptop with Intel Core i7.",
            "metadata": {
                "content_type": "text",
                "page_number": 10,
                "image_path": "",
            },
            "source": "retrieved",
        },
        {
            "chunk_id": "image-001",
            "document": "Image from page 10",
            "metadata": {
                "content_type": "image",
                "page_number": 10,
                "image_path": (
                    "data/processed/normalized/"
                    "dell_catalog/page_10_image_0.jpg"
                ),
            },
            "source": "related",
        },
    ]

    builder = ContextBuilder()

    context = builder.build(
        query="Dell laptop",
        results=results,
    )

    print(context)


if __name__ == "__main__":
    main()