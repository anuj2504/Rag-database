"""
Qdrant Vector Store for dense and ColPali embeddings.

Supports:
- Dense single-vector embeddings (text)
- Multi-vector embeddings (ColPali)
- Hybrid search with filtering
- MULTI-TENANT ISOLATION via organization_id filtering
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
import logging
import uuid
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchParams,
    OptimizersConfigDiff,
    HnswConfigDiff,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from vector store."""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[np.ndarray] = None


class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        payloads: List[Dict[str, Any]]
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        pass


class QdrantVectorStore(VectorStoreBase):
    """
    Qdrant vector store for dense embeddings.

    Production features:
    - HNSW indexing for fast search
    - Payload filtering
    - Batch operations
    - Collection management
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        host: str = "localhost",
        port: int = 6333,
        distance: Distance = Distance.COSINE,
        on_disk: bool = False,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            host: Qdrant host
            port: Qdrant port
            distance: Distance metric (COSINE, EUCLID, DOT)
            on_disk: Store vectors on disk for large datasets
            api_key: API key for Qdrant Cloud
            url: Full URL (for Qdrant Cloud)
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance = distance

        # Connect to Qdrant
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key)

        # Create collection if not exists
        self._ensure_collection(on_disk)

    def _ensure_collection(self, on_disk: bool = False):
        """Create collection with optimized settings if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            logger.info(f"Creating collection: {self.collection_name}")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=self.distance,
                    on_disk=on_disk,
                ),
                # Optimized HNSW settings for production
                hnsw_config=HnswConfigDiff(
                    m=16,  # Number of edges per node
                    ef_construct=100,  # Construction time/quality trade-off
                    full_scan_threshold=10000,
                ),
                # Optimizers for better indexing
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20000,
                    memmap_threshold=50000,
                ),
            )

            # Create payload indexes for common filter fields
            self._create_payload_indexes()

    def _create_payload_indexes(self):
        """Create indexes on payload fields for faster filtering."""
        index_fields = [
            # CRITICAL: Tenant isolation indexes (MUST be first for performance)
            ("organization_id", models.PayloadSchemaType.KEYWORD),
            ("workspace_id", models.PayloadSchemaType.KEYWORD),
            ("collection_id", models.PayloadSchemaType.KEYWORD),
            ("access_level", models.PayloadSchemaType.KEYWORD),
            # Document indexes
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("document_type", models.PayloadSchemaType.KEYWORD),
            ("page_number", models.PayloadSchemaType.INTEGER),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
        ]

        for field_name, field_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                logger.debug(f"Index {field_name} may already exist: {e}")

    def _string_to_uuid(self, string_id: str) -> str:
        """Convert a string ID to a deterministic UUID for Qdrant."""
        # Create a deterministic UUID from the string using MD5 hash
        hash_bytes = hashlib.md5(string_id.encode()).digest()
        return str(uuid.UUID(bytes=hash_bytes))

    def add_documents(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        payloads: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            ids: Document/chunk IDs (strings will be converted to UUIDs)
            embeddings: Embedding vectors (num_docs, dimension)
            payloads: Metadata for each document
            batch_size: Batch size for upsert
        """
        if len(ids) != len(embeddings) or len(ids) != len(payloads):
            raise ValueError("ids, embeddings, and payloads must have same length")

        # Process in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]

            points = []
            for idx, emb, payload in zip(batch_ids, batch_embeddings, batch_payloads):
                # Convert string ID to UUID for Qdrant compatibility
                qdrant_id = self._string_to_uuid(idx)
                # Store original ID in payload for retrieval
                payload["original_id"] = idx
                points.append(
                    PointStruct(
                        id=qdrant_id,
                        vector=emb.tolist(),
                        payload=payload
                    )
                )

            # Use upsert for idempotent operations
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # Wait for indexing
            )

        logger.info(f"Added {len(ids)} documents to {self.collection_name}")

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            limit: Number of results
            filters: Payload filters (e.g., {"document_type": "contract"})
            score_threshold: Minimum similarity score

        Returns:
            List of SearchResult objects
        """
        # Build filter
        qdrant_filter = self._build_filter(filters) if filters else None

        # Search using query_points (qdrant-client 1.7+)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            search_params=SearchParams(
                hnsw_ef=128,  # Search quality/speed trade-off
                exact=False   # Use HNSW index
            ),
            with_payload=True,
        )

        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {}
            )
            for r in results.points
        ]

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary."""
        conditions = []

        for key, value in filters.items():
            if isinstance(value, dict):
                # Range filter: {"page_number": {"gte": 1, "lte": 10}}
                range_params = {}
                if "gte" in value:
                    range_params["gte"] = value["gte"]
                if "lte" in value:
                    range_params["lte"] = value["lte"]
                if "gt" in value:
                    range_params["gt"] = value["gt"]
                if "lt" in value:
                    range_params["lt"] = value["lt"]
                conditions.append(
                    FieldCondition(key=key, range=Range(**range_params))
                )
            elif isinstance(value, list):
                # Match any in list
                for v in value:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=v))
                    )
            else:
                # Exact match
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=conditions)

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids)
        )
        logger.info(f"Deleted {len(ids)} documents from {self.collection_name}")

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted all chunks for document: {document_id}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "dimension": self.dimension,
        }


class QdrantMultiVectorStore:
    """
    Qdrant store for ColPali multi-vector embeddings.

    ColPali produces multiple vectors per page (patch embeddings).
    This store handles storage and MaxSim-based retrieval.
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int = 128,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.dimension = dimension

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key)

        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection for multi-vector storage."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            logger.info(f"Creating multi-vector collection: {self.collection_name}")

            # Use multivector configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "patches": VectorParams(
                        size=self.dimension,
                        distance=Distance.DOT,  # ColPali uses dot product
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
                },
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
            )

            # Create payload indexes
            self._create_payload_indexes()

    def _create_payload_indexes(self):
        """Create indexes for filtering."""
        index_fields = [
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("page_number", models.PayloadSchemaType.INTEGER),
        ]

        for field_name, field_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception:
                pass

    def _string_to_uuid(self, string_id: str) -> str:
        """Convert a string ID to a deterministic UUID for Qdrant."""
        hash_bytes = hashlib.md5(string_id.encode()).digest()
        return str(uuid.UUID(bytes=hash_bytes))

    def add_pages(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        payloads: List[Dict[str, Any]]
    ) -> None:
        """
        Add page embeddings (multi-vector per page).

        Args:
            ids: Page IDs (strings will be converted to UUIDs)
            embeddings: List of arrays, each (num_patches, 128)
            payloads: Metadata for each page
        """
        points = []
        for page_id, emb, payload in zip(ids, embeddings, payloads):
            # Convert string ID to UUID for Qdrant compatibility
            qdrant_id = self._string_to_uuid(page_id)
            # Store original ID in payload for retrieval
            payload["original_id"] = page_id
            points.append(
                PointStruct(
                    id=qdrant_id,
                    vector={"patches": emb.tolist()},
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        logger.info(f"Added {len(ids)} pages to {self.collection_name}")

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search using MaxSim (late interaction).

        Args:
            query_embedding: Query vectors (num_tokens, 128)
            limit: Number of results
            filters: Payload filters

        Returns:
            List of SearchResult with MaxSim scores
        """
        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            using="patches",
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {}
            )
            for r in results.points
        ]

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all pages for a document."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )


# Example usage
if __name__ == "__main__":
    # Dense vector store
    dense_store = QdrantVectorStore(
        collection_name="documents_dense",
        dimension=768,
        host="localhost",
        port=6333
    )

    # Add some documents
    ids = ["chunk_1", "chunk_2", "chunk_3"]
    embeddings = np.random.randn(3, 768).astype(np.float32)
    payloads = [
        {"document_id": "doc1", "document_type": "contract", "text": "Sample text 1"},
        {"document_id": "doc1", "document_type": "contract", "text": "Sample text 2"},
        {"document_id": "doc2", "document_type": "letter", "text": "Sample text 3"},
    ]

    dense_store.add_documents(ids, embeddings, payloads)

    # Search
    query = np.random.randn(768).astype(np.float32)
    results = dense_store.search(query, limit=5, filters={"document_type": "contract"})

    for r in results:
        print(f"ID: {r.id}, Score: {r.score:.4f}")
