from app.retrieval.embeddings.clip_embedder import CLIPEmbedder


def main():

    embedder = CLIPEmbedder()

    text_vector = embedder.embed_text(
        "Dell laptop computer"
    )

    print("=" * 60)
    print("CLIP TEST")
    print("=" * 60)

    print("Text vector length:", len(text_vector))
    print("First 5 values:", text_vector[:5])

    assert len(text_vector) == 512

    print("=" * 60)
    print("CLIP TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()