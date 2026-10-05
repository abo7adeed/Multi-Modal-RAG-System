import threading
from typing import Any

from app.config import settings
from app.ingestion.chunk_schemas import Chunk

from .embeddings.clip_embedder import CLIPEmbedder
from .lexical import BM25Index
from .vector_store.chroma_store import ChromaVectorStore


class MultimodalRetriever:

    def __init__(
        self,
        embedder: CLIPEmbedder,
        vector_store: ChromaVectorStore,
        lexical_index: BM25Index | None = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        # Built lazily from documents already stored in Chroma, and
        # rebuilt after new text is indexed.
        self._lexical_index = lexical_index
        self._lexical_lock = threading.Lock()

    @property
    def lexical_index(self) -> BM25Index:
        if self._lexical_index is None:
            with self._lexical_lock:
                if self._lexical_index is None:
                    self._lexical_index = BM25Index.from_vector_store(
                        self.vector_store
                    )
        return self._lexical_index

    def invalidate_lexical_index(self) -> None:
        """
        Force the lexical index to be rebuilt from the vector store on
        next use. Called after indexing so newly ingested text is
        searchable.
        """
        with self._lexical_lock:
            self._lexical_index = None

    # ============================================================
    # INDEXING
    # ============================================================

    def index_chunks(
        self,
        chunks: list[Chunk],
    ):
        if not chunks:
            return

        chunk_ids = []
        embeddings = []
        documents = []
        metadatas = []

        text_chunks = []
        image_chunks = []

        for chunk in chunks:

            if chunk.content_type == "text":
                text_chunks.append(chunk)

            elif chunk.content_type == "image":
                image_chunks.append(chunk)

        # --------------------------------------------------------
        # Text embeddings
        # --------------------------------------------------------

        if text_chunks:

            texts = [
                chunk.text or ""
                for chunk in text_chunks
            ]

            text_embeddings = (
                self.embedder.embed_texts(texts)
            )

            for chunk, embedding in zip(
                text_chunks,
                text_embeddings,
            ):

                chunk_ids.append(
                    chunk.chunk_id
                )

                embeddings.append(
                    embedding
                )

                documents.append(
                    chunk.text or ""
                )

                metadatas.append(
                    {
                        "document_id": chunk.document_id,
                        "content_type": chunk.content_type,
                        "page_number": (
                            chunk.page_number
                            if chunk.page_number is not None
                            else -1
                        ),
                        "image_path": "",
                        "related_chunk_ids": ",".join(
                            chunk.related_chunk_ids
                        ),
                        **chunk.metadata,
                    }
                )

        # --------------------------------------------------------
        # Image embeddings
        # --------------------------------------------------------

        if image_chunks:

            image_chunks_with_paths = [
                chunk
                for chunk in image_chunks
                if chunk.image_path
            ]

            image_paths = [
                chunk.image_path
                for chunk in image_chunks_with_paths
            ]

            image_embeddings = (
                self.embedder.embed_images(
                    image_paths
                )
            )

            for chunk, embedding in zip(
                image_chunks_with_paths,
                image_embeddings,
            ):

                chunk_ids.append(
                    chunk.chunk_id
                )

                embeddings.append(
                    embedding
                )

                documents.append(
                    f"Image from page {chunk.page_number}"
                )

                metadatas.append(
                    {
                        "document_id": chunk.document_id,
                        "content_type": chunk.content_type,
                        "page_number": (
                            chunk.page_number
                            if chunk.page_number is not None
                            else -1
                        ),
                        "image_path": (
                            chunk.image_path or ""
                        ),
                        "related_chunk_ids": ",".join(
                            chunk.related_chunk_ids
                        ),
                        **chunk.metadata,
                    }
                )

        # --------------------------------------------------------
        # Store vectors
        # --------------------------------------------------------

        self.vector_store.add(
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # New text invalidates the in-memory lexical index. Drop it
        # so freshly ingested chunks are immediately searchable.
        self.invalidate_lexical_index()

    def delete_document(self, document_id: str) -> int:
        """
        Remove every chunk belonging to a document.

        Returns the number of chunks removed, so the caller can tell
        an empty delete (unknown id) from a real one.
        """
        removed = self.vector_store.delete_where(
            {"document_id": document_id}
        )

        # The lexical index still holds the deleted documents' text,
        # so they would keep matching BM25 queries until it is
        # rebuilt. Dropping it is the simplest correct response; it
        # is rebuilt lazily on the next search.
        self.invalidate_lexical_index()

        return removed

    # ============================================================
    # MODALITY-SPECIFIC SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        query_embedding = (
            self.embedder.embed_text(query)
        )

        # --------------------------------------------------------
        # Text retrieval
        # --------------------------------------------------------

        text_results = (
            self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                content_type="text",
            )
        )

        # --------------------------------------------------------
        # Image retrieval
        # --------------------------------------------------------

        image_results = (
            self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                content_type="image",
            )
        )

        return {
            "text": text_results,
            "image": image_results,
        }

    def search_by_image(
        self,
        image_path: str,
        top_k: int = 2,
    ) -> list[dict]:
        """
        Find indexed page images that look like the given image.

        This is the image-to-image counterpart of search(): the query
        is a picture rather than text, so CLIP's *visual* tower does
        the matching. Used when the user attaches their own image, to
        surface catalog pages showing a similar product.
        """
        image_embedding = self.embedder.embed_image(image_path)

        raw_results = self.vector_store.search(
            query_embedding=image_embedding,
            top_k=top_k,
            content_type="image",
        )

        return self._prepare_results(raw_results)

    # ============================================================
    # PREPARE CHROMA RESULTS
    # ============================================================

    def _prepare_results(
        self,
        results,
    ):
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        prepared = []

        for rank, (
            chunk_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):

            prepared.append(
                {
                    "chunk_id": chunk_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": rank,
                    "source": "retrieved",
                }
            )

        return prepared

    # ============================================================
    # RECIPROCAL RANK FUSION
    # ============================================================

    def _rrf_fusion(
        self,
        text_results: list[dict],
        image_results: list[dict],
        top_k: int = 5,
        rrf_k: int = 60,
    ):
        """
        Fuse independently ranked text and image results.

        RRF is used because raw CLIP distances from different
        modalities should not be directly compared.
        """

        fused = {}

        # --------------------------------------------------------
        # Text ranking
        # --------------------------------------------------------

        for result in text_results:

            chunk_id = result["chunk_id"]
            rank = result["rank"]

            score = 1.0 / (
                rrf_k + rank
            )

            if chunk_id not in fused:

                fused[chunk_id] = {
                    **result,
                    "score": 0.0,
                }

            fused[chunk_id]["score"] += score

        # --------------------------------------------------------
        # Image ranking
        # --------------------------------------------------------

        for result in image_results:

            chunk_id = result["chunk_id"]
            rank = result["rank"]

            score = 1.0 / (
                rrf_k + rank
            )

            if chunk_id not in fused:

                fused[chunk_id] = {
                    **result,
                    "score": 0.0,
                }

            fused[chunk_id]["score"] += score

        # --------------------------------------------------------
        # Final fused ranking
        # --------------------------------------------------------

        ranked = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        return ranked[:top_k]

    # ============================================================
    # RELATED CHUNK EXPANSION
    # ============================================================

    def expand_related_chunks(
        self,
        results: list[dict],
        max_related_per_chunk: int = 2,
    ):
        """
        Add related chunks as supporting context.

        IMPORTANT:
        Related chunks never replace or outrank the primary
        retrieved results.
        """

        primary_results = []
        related_results = []

        seen_ids = set()

        # --------------------------------------------------------
        # Process primary results
        # --------------------------------------------------------

        for result in results:

            chunk_id = result["chunk_id"]

            if chunk_id in seen_ids:
                continue

            seen_ids.add(chunk_id)

            primary_results.append(
                {
                    **result,
                    "source": "retrieved",
                }
            )

        # --------------------------------------------------------
        # Expand each primary result
        # --------------------------------------------------------

        for result in primary_results:

            related_ids_string = (
                result["metadata"].get(
                    "related_chunk_ids",
                    "",
                )
            )

            if not related_ids_string:
                continue

            related_ids = [
                related_id.strip()
                for related_id in related_ids_string.split(",")
                if related_id.strip()
            ]

            # Remove duplicates
            related_ids = list(
                dict.fromkeys(
                    related_ids
                )
            )

            # Remove already retrieved chunks
            related_ids = [
                related_id
                for related_id in related_ids
                if related_id not in seen_ids
            ]

            if not related_ids:
                continue

            # ----------------------------------------------------
            # Batch database lookup
            # ----------------------------------------------------

            related_candidates = (
                self.vector_store.collection.get(
                    ids=related_ids,
                    include=[
                        "documents",
                        "metadatas",
                    ],
                )
            )

            found_ids = (
                related_candidates["ids"]
            )

            documents = (
                related_candidates["documents"]
            )

            metadatas = (
                related_candidates["metadatas"]
            )

            primary_page = (
                result["metadata"].get(
                    "page_number"
                )
            )

            candidates = []

            # ----------------------------------------------------
            # Build related candidates
            # ----------------------------------------------------

            for (
                related_id,
                document,
                metadata,
            ) in zip(
                found_ids,
                documents,
                metadatas,
            ):

                if related_id in seen_ids:
                    continue

                related_page = metadata.get(
                    "page_number"
                )

                # Same-page chunks are preferred
                same_page = (
                    primary_page is not None
                    and related_page == primary_page
                )

                # Related score is ONLY used to choose
                # which related chunks to include.
                #
                # It is deliberately much smaller than
                # the primary RRF score.
                related_score = (
                    1.0 if same_page else 0.0
                )

                candidates.append(
                    {
                        "chunk_id": related_id,
                        "document": document,
                        "metadata": metadata,
                        "distance": None,
                        "rank": None,
                        "score": related_score,
                        "source": "related",
                    }
                )

            # ----------------------------------------------------
            # Rank related candidates
            # ----------------------------------------------------

            candidates.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

            # ----------------------------------------------------
            # Add limited related context
            # ----------------------------------------------------

            added = 0

            for candidate in candidates:

                if added >= max_related_per_chunk:
                    break

                candidate_id = (
                    candidate["chunk_id"]
                )

                if candidate_id in seen_ids:
                    continue

                related_results.append(
                    candidate
                )

                seen_ids.add(
                    candidate_id
                )

                added += 1

        # --------------------------------------------------------
        # CRITICAL:
        #
        # Primary results ALWAYS come first.
        # Related chunks are appended afterward.
        # --------------------------------------------------------

        return (
            primary_results
            + related_results
        )

    # ============================================================
    # COMPLETE MULTIMODAL RETRIEVAL
    # ============================================================

    def search_multimodal(
        self,
        query: str,
        top_k: int = 5,
        expansion_limit: int = 3,
    ):
        """
        Complete multimodal retrieval pipeline.

        Query
          ↓
        Text Retrieval
          ↓
        Image Retrieval
          ↓
        RRF Fusion
          ↓
        Primary Results
          ↓
        Related Context Expansion
          ↓
        Final Context
        """

        # --------------------------------------------------------
        # 1. Independent modality retrieval
        # --------------------------------------------------------

        results = self.search(
            query=query,
            top_k=top_k,
        )

        # --------------------------------------------------------
        # 2. Prepare text results
        # --------------------------------------------------------

        text_results = self._prepare_results(
            results["text"]
        )

        # --------------------------------------------------------
        # 3. Prepare image results
        # --------------------------------------------------------

        image_results = self._prepare_results(
            results["image"]
        )

        # --------------------------------------------------------
        # 4. Off-topic gate
        #
        # CLIP's text tower cannot tell a relevant query from an
        # unrelated one, so it always returns neighbours. BM25 is
        # the relevance signal: if the query shares no meaningful
        # term with the index, nothing is genuinely relevant.
        # --------------------------------------------------------

        lexical_results = self.lexical_index.search(
            query=query,
            top_k=top_k,
        )

        # The gate only makes sense when there is text to match
        # against. A corpus of images has no text chunks, so the
        # gate is skipped and CLIP drives retrieval on its own.
        if self.lexical_index.documents:
            # Non-emptiness is not enough: one common word makes an
            # unrelated question "match". Measured on this corpus, a
            # coverage floor removes the worst off-topic answers at no
            # cost in recall (see eval/BASELINE.md).
            relevance = self.lexical_index.keyword_relevance(query)

            if (
                not lexical_results
                or relevance["coverage"]
                < settings.retrieval_lexical_min_coverage
            ):
                return []

        # --------------------------------------------------------
        # 5. RRF fusion
        # --------------------------------------------------------

        primary_results = self._rrf_fusion(
            text_results=self._merge_text_rankings(
                text_results=text_results,
                lexical_results=lexical_results,
            ),
            image_results=image_results,
            top_k=top_k,
        )

        # --------------------------------------------------------
        # 6. Select strongest candidates for expansion
        # --------------------------------------------------------

        expansion_candidates = (
            primary_results[:expansion_limit]
        )

        # --------------------------------------------------------
        # 7. Expand related context
        # --------------------------------------------------------

        expanded_results = (
            self.expand_related_chunks(
                expansion_candidates,
                max_related_per_chunk=2,
            )
        )

        return expanded_results

    # ============================================================
    # HYBRID TEXT RANKING
    # ============================================================

    def _merge_text_rankings(
        self,
        text_results: list[dict],
        lexical_results: list[dict[str, Any]],
    ) -> list[dict]:
        """
        Combine CLIP text hits with BM25 hits for the same chunks.

        Chunks found by both signals keep their CLIP distance/metadata;
        chunks found only by BM25 are looked up in the vector store.
        """
        by_id = {
            result["chunk_id"]: dict(result)
            for result in text_results
        }

        for lexical in lexical_results:
            chunk_id = lexical["chunk_id"]

            if chunk_id in by_id:
                by_id[chunk_id]["lexical_score"] = lexical["score"]
                continue

            stored = self.vector_store.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )

            if not stored.get("ids"):
                continue

            by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "document": stored["documents"][0],
                "metadata": stored["metadatas"][0],
                "distance": None,
                "rank": None,
                "lexical_score": lexical["score"],
                "source": "lexical",
            }

        merged = list(by_id.values())

        # BM25 hits first (lexical precision), then CLIP-only hits.
        merged.sort(
            key=lambda item: (
                0 if item.get("lexical_score") else 1,
                -item.get("lexical_score", 0.0),
                item.get("distance") or 0.0,
            )
        )

        for rank, result in enumerate(merged, start=1):
            result["rank"] = rank
            result["score"] = result.get("lexical_score", 0.0)
            result.setdefault("source", "retrieved")
            if result["source"] == "lexical":
                result["source"] = "retrieved"

        return merged