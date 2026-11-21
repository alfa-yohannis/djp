"""Hybrid retrieval: dense vectors, BM25 keywords, fusion and reranking.

Neither retrieval method alone is sufficient for this corpus. Vector search
understands that "beban mengajar dosen" and "EWMP" describe the same thing, but
drifts on exact identifiers. Keyword search nails a decree number such as
``SK-023/PRA/XI/2024`` but misses every paraphrase. :class:`HybridRetriever`
runs both, merges their rankings with reciprocal rank fusion, and hands the
survivors to a cross-encoder for a final precision pass.

Fusion is done on *ranks* rather than raw scores deliberately: cosine
similarity and BM25 live on incomparable scales, so any weighted sum of the two
would need re-tuning whenever the corpus changes.
"""

from __future__ import annotations

import logging

from .config import SETTINGS, Settings
from .documents import RetrievedPassage
from .embeddings import TextEmbedder
from .reranking import CrossEncoderReranker
from .storage import DOCUMENT_CATALOGUE, VECTOR_INDEX, DocumentCatalogue, VectorIndex

LOGGER = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """Merges several ranked candidate lists into one consensus ranking.

    Each list contributes ``1 / (k + rank)`` to a passage's score, so a passage
    ranked highly by both retrieval methods outranks one that only a single
    method loved. The constant ``k`` damps the influence of the very top ranks,
    preventing one method from dominating the merged list.
    """

    def __init__(self, smoothing_constant: int) -> None:
        """Store the fusion constant, conventionally 60."""
        self._smoothing_constant = smoothing_constant

    def fuse(self, ranked_lists: list[list[RetrievedPassage]]) -> list[RetrievedPassage]:
        """Combine ranked lists into a single list ordered by fused score.

        Passages appearing in more than one list are merged into one object that
        keeps every score it was found with, so the interface can show whether a
        result came from semantic search, keyword search, or both.
        """
        merged_by_chunk_id: dict[int, RetrievedPassage] = {}

        for ranked_list in ranked_lists:
            for rank_position, passage in enumerate(ranked_list, start=1):
                contribution = 1.0 / (self._smoothing_constant + rank_position)
                existing = merged_by_chunk_id.get(passage.chunk_id)
                if existing is None:
                    passage.fused_score = contribution
                    merged_by_chunk_id[passage.chunk_id] = passage
                else:
                    existing.fused_score += contribution
                    self._merge_scores(existing, passage)

        fused_passages = list(merged_by_chunk_id.values())
        fused_passages.sort(key=self._sort_key, reverse=True)
        return fused_passages

    def _merge_scores(self, target: RetrievedPassage, other: RetrievedPassage) -> None:
        """Copy any score the target is missing from a duplicate hit."""
        if target.dense_score is None and other.dense_score is not None:
            target.dense_score = other.dense_score
        if target.sparse_score is None and other.sparse_score is not None:
            target.sparse_score = other.sparse_score

    def _sort_key(self, passage: RetrievedPassage) -> float:
        """Ordering key for the fused list: highest fused score first."""
        return passage.fused_score


class HybridRetriever:
    """Finds the passages most likely to answer a question.

    Retrieval runs in four stages — dense search, keyword search, rank fusion,
    cross-encoder reranking — and returns passages already numbered for
    citation, so the prompt builder and the interface agree on what "[3]" means.
    """

    def __init__(
        self,
        settings: Settings,
        catalogue: DocumentCatalogue,
        vector_index: VectorIndex,
        embedder: TextEmbedder,
        reranker: CrossEncoderReranker,
    ) -> None:
        """Wire the retriever to the stores and models it queries."""
        self._settings = settings
        self._catalogue = catalogue
        self._vector_index = vector_index
        self._embedder = embedder
        self._reranker = reranker
        self._fusion = ReciprocalRankFusion(settings.fusion_constant)

    def retrieve(
        self, question: str, folder_filter: str | None = None, passage_count: int | None = None
    ) -> list[RetrievedPassage]:
        """Return the best passages for ``question``, numbered for citation.

        Args:
            question: The user's question, in any language.
            folder_filter: When given, restricts results to one top-level folder.
            passage_count: Overrides the configured number of answer passages.
        """
        requested_count = passage_count or self._settings.answer_passage_count
        if not question.strip():
            return []

        allowed_chunk_ids = self._resolve_folder_restriction(folder_filter)
        if allowed_chunk_ids is not None and not allowed_chunk_ids:
            return []

        dense_results = self._search_dense(question, allowed_chunk_ids)
        sparse_results = self._catalogue.search_by_keyword(
            question, self._settings.sparse_candidate_count, folder_filter
        )
        LOGGER.debug(
            "Kandidat: %d semantik, %d kata kunci", len(dense_results), len(sparse_results)
        )

        fused_results = self._fusion.fuse([dense_results, sparse_results])
        rerank_input = fused_results[: self._settings.reranking_candidate_count]
        final_passages = self._reranker.rerank(question, rerank_input, requested_count)

        for citation_index, passage in enumerate(final_passages, start=1):
            passage.citation_number = citation_index
        return final_passages

    def _resolve_folder_restriction(self, folder_filter: str | None) -> set[int] | None:
        """Translate a folder name into the set of chunk ids it contains."""
        if not folder_filter:
            return None
        return self._catalogue.fetch_chunk_ids_for_folder(folder_filter)

    def _search_dense(
        self, question: str, allowed_chunk_ids: set[int] | None
    ) -> list[RetrievedPassage]:
        """Run vector search and hydrate the hits with their stored text."""
        if self._vector_index.size == 0:
            return []

        question_vector = self._embedder.encode_question(question)
        scored_ids = self._vector_index.search(
            question_vector, self._settings.dense_candidate_count, allowed_chunk_ids
        )
        if not scored_ids:
            return []

        chunk_ids = [chunk_id for chunk_id, _score in scored_ids]
        passages_by_id = self._catalogue.fetch_passages_by_id(chunk_ids)

        hydrated: list[RetrievedPassage] = []
        for chunk_id, similarity in scored_ids:
            passage = passages_by_id.get(chunk_id)
            if passage is None:
                # The vector index outlived its chunk row; a reindex will fix it.
                continue
            passage.dense_score = similarity
            hydrated.append(passage)
        return hydrated


#: Shared reranker and retriever bound to the process-wide stores and models.
CROSS_ENCODER_RERANKER = CrossEncoderReranker(SETTINGS)

#: Embeds questions while serving, on QUERY_EMBED_DEVICE (CPU by default) so
#: the GPU stays free for the language model, through OpenVINO int8 when that
#: model has been built. Indexing uses its own instance.
QUERY_EMBEDDER = TextEmbedder(
    SETTINGS,
    device=SETTINGS.query_embedding_device,
    backend=SETTINGS.query_embedding_backend,
    openvino_directory=SETTINGS.query_embedding_openvino_directory,
)
