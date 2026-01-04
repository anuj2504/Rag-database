"""
Hybrid Search combining BM25, Dense, and ColPali retrieval.

Uses Reciprocal Rank Fusion (RRF) to combine results from multiple
retrieval methods, providing the best of both keyword and semantic search.
"""
from typing import List, Dict, Any, Optional, Tuple
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
    Hybrid search combining multiple retrieval methods.

    Supports:
    - BM25 (keyword)
    - Dense embeddings (semantic)
    - ColPali (visual)

    Uses RRF for result fusion.
    """

    def __init__(
        self,
        bm25_store=None,
        dense_store=None,
        colpali_store=None,
        dense_embedder=None,
        colpali_embedder=None,
        rrf_k: int = 60,
        default_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize hybrid searcher.

        Args:
            bm25_store: BM25Index instance
            dense_store: QdrantVectorStore for dense embeddings
            colpali_store: QdrantMultiVectorStore for ColPali
            dense_embedder: Dense embedding model
            colpali_embedder: ColPali embedding model
            rrf_k: RRF constant
            default_weights: Default weights for each method
        """
        self.bm25_store = bm25_store
        self.dense_store = dense_store
        self.colpali_store = colpali_store
        self.dense_embedder = dense_embedder
        self.colpali_embedder = colpali_embedder

        self.rrf = ReciprocalRankFusion(k=rrf_k)

        # Default weights (can be tuned based on your data)
        self.default_weights = default_weights or {
            RetrievalMethod.BM25.value: 0.3,
            RetrievalMethod.DENSE.value: 0.5,
            RetrievalMethod.COLPALI.value: 0.2,
        }

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

        # Fuse results using RRF
        use_weights = weights or self.default_weights
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
