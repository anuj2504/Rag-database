"""
Hybrid Search combining BM25, Dense, and ColPali retrieval.

Uses Reciprocal Rank Fusion (RRF) to combine results from multiple
retrieval methods, providing the best of both keyword and semantic search.
"""
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.query_analyzer import QueryModalityDetector
    from src.storage.vector_store import QdrantVisualElementStore
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RetrievalMethod(Enum):
    BM25 = "bm25"
    DENSE = "dense"
    COLPALI = "colpali"


@dataclass
class HybridResult:
    """Result from hybrid search."""
    id: str
    final_score: float
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Scores from individual methods
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    colpali_score: Optional[float] = None
    colpali_rank: Optional[int] = None


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining ranked lists.

    RRF Score = Σ 1 / (k + rank_i)

    Where:
    - k is a constant (typically 60) that dampens high-ranked items
    - rank_i is the position in each ranked list (1-indexed)

    RRF is preferred over score normalization because:
    - Works with different scoring scales
    - Doesn't require calibration
    - Robust to outliers
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF.

        Args:
            k: Ranking constant (higher = more weight to lower ranks)
        """
        self.k = k

    def fuse(
        self,
        ranked_lists: Dict[str, List[Tuple[str, float]]],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked lists using RRF.

        Args:
            ranked_lists: Dict of method_name -> [(id, score), ...]
            weights: Optional weights for each method

        Returns:
            Fused list of (id, score) sorted by score descending
        """
        if not ranked_lists:
            return []

        # Default equal weights
        if weights is None:
            weights = {method: 1.0 for method in ranked_lists}

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)

        for method, results in ranked_lists.items():
            weight = weights.get(method, 1.0)
            for rank, (doc_id, _) in enumerate(results, start=1):
                rrf_scores[doc_id] += weight * (1.0 / (self.k + rank))

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results


class HybridSearcher:
    """
    Hybrid search combining multiple retrieval methods with dynamic weight routing.

    Supports:
    - BM25 (keyword)
    - Dense embeddings (semantic)
    - ColPali (visual - full pages)
    - Visual Elements (tables, figures - cropped)

    Features:
    - Uses RRF for result fusion
    - Dynamic weight adjustment based on query modality
    - Visual element search for queries targeting tables/figures

    Based on ColPALI Meets DocLayNet architecture.
    """

    def __init__(
        self,
        bm25_store=None,
        dense_store=None,
        colpali_store=None,
        visual_element_store=None,  # NEW: For cropped tables/figures
        dense_embedder=None,
        colpali_embedder=None,
        query_analyzer=None,  # NEW: For dynamic weight routing
        rrf_k: int = 60,
        default_weights: Optional[Dict[str, float]] = None,
        enable_dynamic_weights: bool = True,  # NEW: Enable/disable dynamic routing
    ):
        """
        Initialize hybrid searcher with visual element support.

        Args:
            bm25_store: BM25Index instance
            dense_store: QdrantVectorStore for dense embeddings
            colpali_store: QdrantMultiVectorStore for ColPali (full pages)
            visual_element_store: QdrantVisualElementStore for tables/figures
            dense_embedder: Dense embedding model
            colpali_embedder: ColPali embedding model
            query_analyzer: QueryModalityDetector for dynamic weights
            rrf_k: RRF constant
            default_weights: Default weights for each method
            enable_dynamic_weights: Whether to use query-based weight adjustment
        """
        self.bm25_store = bm25_store
        self.dense_store = dense_store
        self.colpali_store = colpali_store
        self.visual_element_store = visual_element_store
        self.dense_embedder = dense_embedder
        self.colpali_embedder = colpali_embedder
        self.query_analyzer = query_analyzer
        self.enable_dynamic_weights = enable_dynamic_weights

        self.rrf = ReciprocalRankFusion(k=rrf_k)

        # Default weights (used when dynamic routing is disabled or no analyzer)
        self.default_weights = default_weights or {
            RetrievalMethod.BM25.value: 0.3,
            RetrievalMethod.DENSE.value: 0.5,
            RetrievalMethod.COLPALI.value: 0.2,
        }

    def _get_weights_for_query(self, query: str, custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Get weights for a query, optionally using dynamic routing.

        Args:
            query: Search query
            custom_weights: Override weights (skip dynamic routing)

        Returns:
            Dict with bm25, dense, colpali weights
        """
        # Custom weights take precedence
        if custom_weights is not None:
            return custom_weights

        # Use dynamic routing if enabled and analyzer available
        if self.enable_dynamic_weights and self.query_analyzer:
            analysis = self.query_analyzer.analyze(query)
            logger.info(
                f"Query analysis: modality={analysis.modality.value}, "
                f"visual_score={analysis.visual_score:.2f}, "
                f"weights={analysis.suggested_weights}"
            )
            return analysis.suggested_weights

        # Fall back to default weights
        return self.default_weights

    def search(
        self,
        query: str,
        limit: int = 10,
        methods: Optional[List[RetrievalMethod]] = None,
        weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_text: bool = True
    ) -> List[HybridResult]:
        """
        Perform hybrid search across multiple methods.

        Args:
            query: Search query
            limit: Number of results
            methods: Which methods to use (default: all available)
            weights: Custom weights for RRF
            filters: Metadata filters
            include_text: Whether to include text in results

        Returns:
            List of HybridResult sorted by fused score
        """
        # Determine which methods to use
        if methods is None:
            methods = []
            if self.bm25_store:
                methods.append(RetrievalMethod.BM25)
            if self.dense_store and self.dense_embedder:
                methods.append(RetrievalMethod.DENSE)
            if self.colpali_store and self.colpali_embedder:
                methods.append(RetrievalMethod.COLPALI)

        if not methods:
            raise ValueError("No search methods available")

        # Get more results from each method for fusion
        fetch_limit = limit * 3

        # Collect results from each method
        ranked_lists: Dict[str, List[Tuple[str, float]]] = {}
        all_results_by_id: Dict[str, Dict[str, Any]] = {}

        # BM25 search
        if RetrievalMethod.BM25 in methods and self.bm25_store:
            logger.info(f"BM25 search: query='{query[:50]}', filters={filters}")
            bm25_results = self.bm25_store.search(
                query=query,
                limit=fetch_limit,
                filters=filters
            )
            logger.info(f"BM25 returned {len(bm25_results)} results")
            ranked_lists[RetrievalMethod.BM25.value] = [
                (r.id, r.score) for r in bm25_results
            ]
            for rank, r in enumerate(bm25_results, 1):
                if r.id not in all_results_by_id:
                    all_results_by_id[r.id] = {
                        "text": r.text,
                        "metadata": r.metadata
                    }
                all_results_by_id[r.id]["bm25_score"] = r.score
                all_results_by_id[r.id]["bm25_rank"] = rank

        # Dense search
        if RetrievalMethod.DENSE in methods and self.dense_store and self.dense_embedder:
            query_embedding = self.dense_embedder.embed_query(query)
            dense_results = self.dense_store.search(
                query_embedding=query_embedding,
                limit=fetch_limit,
                filters=filters
            )
            # Use original_id from payload for fusion with BM25 (Qdrant converts IDs to UUIDs)
            logger.info(f"Dense returned {len(dense_results)} results")
            ranked_lists[RetrievalMethod.DENSE.value] = [
                (r.payload.get("original_id", r.id), r.score) for r in dense_results
            ]
            for rank, r in enumerate(dense_results, 1):
                # Use original_id for consistent ID across all methods
                doc_id = r.payload.get("original_id", r.id)
                if doc_id not in all_results_by_id:
                    all_results_by_id[doc_id] = {
                        "text": r.payload.get("text"),
                        "metadata": r.payload
                    }
                all_results_by_id[doc_id]["dense_score"] = r.score
                all_results_by_id[doc_id]["dense_rank"] = rank

        # ColPali search
        if RetrievalMethod.COLPALI in methods and self.colpali_store and self.colpali_embedder:
            query_embedding = self.colpali_embedder.embed_query(query)
            colpali_results = self.colpali_store.search(
                query_embedding=query_embedding,
                limit=fetch_limit,
                filters=filters
            )
            # Use original_id from payload for fusion (Qdrant converts IDs to UUIDs)
            ranked_lists[RetrievalMethod.COLPALI.value] = [
                (r.payload.get("original_id", r.id), r.score) for r in colpali_results
            ]
            for rank, r in enumerate(colpali_results, 1):
                # Use original_id for consistent ID across all methods
                doc_id = r.payload.get("original_id", r.id)
                if doc_id not in all_results_by_id:
                    all_results_by_id[doc_id] = {
                        "text": None,  # ColPali returns pages, not text chunks
                        "metadata": r.payload
                    }
                all_results_by_id[doc_id]["colpali_score"] = r.score
                all_results_by_id[doc_id]["colpali_rank"] = rank

        # Get weights (dynamic or custom)
        use_weights = self._get_weights_for_query(query, weights)

        # Fuse results using RRF
        fused_results = self.rrf.fuse(ranked_lists, use_weights)

        # Log fusion stats
        logger.info(
            f"RRF fusion: methods={list(ranked_lists.keys())}, "
            f"counts={[len(v) for v in ranked_lists.values()]}, "
            f"fused={len(fused_results)}"
        )

        # Build final results
        results = []
        for doc_id, final_score in fused_results[:limit]:
            result_data = all_results_by_id.get(doc_id, {})

            result = HybridResult(
                id=doc_id,
                final_score=final_score,
                text=result_data.get("text") if include_text else None,
                metadata=result_data.get("metadata", {}),
                bm25_score=result_data.get("bm25_score"),
                bm25_rank=result_data.get("bm25_rank"),
                dense_score=result_data.get("dense_score"),
                dense_rank=result_data.get("dense_rank"),
                colpali_score=result_data.get("colpali_score"),
                colpali_rank=result_data.get("colpali_rank"),
            )
            results.append(result)

        return results

    def search_bm25_only(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HybridResult]:
        """Search using BM25 only."""
        return self.search(
            query=query,
            limit=limit,
            methods=[RetrievalMethod.BM25],
            filters=filters
        )

    def search_dense_only(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HybridResult]:
        """Search using dense embeddings only."""
        return self.search(
            query=query,
            limit=limit,
            methods=[RetrievalMethod.DENSE],
            filters=filters
        )

    def search_visual(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HybridResult]:
        """Search using ColPali visual embeddings only."""
        return self.search(
            query=query,
            limit=limit,
            methods=[RetrievalMethod.COLPALI],
            filters=filters
        )

    def search_visual_elements(
        self,
        query: str,
        limit: int = 10,
        element_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HybridResult]:
        """
        Search visual elements (tables, figures) directly.

        This searches the visual_elements collection for cropped tables/figures,
        which is more precise than full-page ColPali search for visual queries.

        Args:
            query: Search query
            limit: Number of results
            element_types: Filter by element types (e.g., ["Table", "Image"])
            filters: Additional payload filters

        Returns:
            List of HybridResult with visual element matches
        """
        if not self.visual_element_store or not self.colpali_embedder:
            logger.warning("Visual element search requires visual_element_store and colpali_embedder")
            return []

        # Embed query with ColPali
        query_embedding = self.colpali_embedder.embed_query(query)

        # Search visual elements collection
        results = self.visual_element_store.search(
            query_embedding=query_embedding,
            limit=limit,
            element_types=element_types,
            filters=filters
        )

        # Convert to HybridResult
        hybrid_results = []
        for r in results:
            hybrid_results.append(HybridResult(
                id=r.payload.get("original_id", r.id),
                final_score=r.score,
                text=r.payload.get("text_content"),
                metadata=r.payload,
                colpali_score=r.score,
                colpali_rank=None,  # Not part of ranked fusion
            ))

        return hybrid_results

    def search_with_visual_elements(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_text: bool = True
    ) -> Tuple[List[HybridResult], List[HybridResult]]:
        """
        Search with automatic visual element detection.

        If the query targets visual content (tables, figures), also searches
        the visual_elements collection and returns both result sets.

        Args:
            query: Search query
            limit: Number of results
            filters: Metadata filters
            include_text: Whether to include text in results

        Returns:
            Tuple of (hybrid_results, visual_element_results)
        """
        # Always run hybrid search
        hybrid_results = self.search(
            query=query,
            limit=limit,
            filters=filters,
            include_text=include_text
        )

        # Check if we should also search visual elements
        visual_element_results = []
        if self.query_analyzer and self.visual_element_store:
            analysis = self.query_analyzer.analyze(query)
            if analysis.should_search_visual_elements():
                # Get target element types from query
                element_types = self.query_analyzer.get_target_element_types(query)
                visual_element_results = self.search_visual_elements(
                    query=query,
                    limit=limit // 2,  # Half the limit for visual elements
                    element_types=element_types,
                    filters=filters
                )
                logger.info(
                    f"Visual element search: found {len(visual_element_results)} elements "
                    f"(types={element_types})"
                )

        return hybrid_results, visual_element_results


class CrossEncoderReranker:
    """
    Optional: Cross-encoder reranking for improved precision.

    Cross-encoders are more accurate than bi-encoders but slower.
    Use as a final reranking step on top-k hybrid results.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: List[HybridResult],
        limit: Optional[int] = None
    ) -> List[HybridResult]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Original query
            results: Results to rerank
            limit: Max results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        # Get texts for reranking
        texts = [r.text or "" for r in results]
        pairs = [[query, text] for text in texts]

        # Score pairs
        scores = self.model.predict(pairs)

        # Sort by cross-encoder score
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Update scores and return
        reranked = []
        for result, ce_score in scored_results[:limit]:
            result.final_score = float(ce_score)
            reranked.append(result)

        return reranked


# Example usage
if __name__ == "__main__":
    from src.storage.bm25_store import BM25Index
    from src.storage.vector_store import QdrantVectorStore
    from src.embeddings.dense_embedder import get_embedder

    # Initialize stores
    bm25 = BM25Index()
    dense_store = QdrantVectorStore(
        collection_name="documents_dense",
        dimension=768
    )
    embedder = get_embedder("sentence-transformers")

    # Initialize hybrid searcher
    searcher = HybridSearcher(
        bm25_store=bm25,
        dense_store=dense_store,
        dense_embedder=embedder,
        default_weights={
            "bm25": 0.4,
            "dense": 0.6,
        }
    )

    # Search
    results = searcher.search(
        query="What are the payment terms?",
        limit=10,
        filters={"document_type": "contract"}
    )

    for r in results:
        print(f"ID: {r.id}")
        print(f"  Final Score: {r.final_score:.4f}")
        print(f"  BM25: rank={r.bm25_rank}, score={r.bm25_score}")
        print(f"  Dense: rank={r.dense_rank}, score={r.dense_score}")
        print()
