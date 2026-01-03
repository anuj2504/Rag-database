"""
BM25 Index for keyword-based retrieval.

BM25 (Best Matching 25) is a probabilistic ranking function for text retrieval.
It excels at:
- Exact term matching (contract numbers, names, dates)
- Keyword search
- Cases where semantic search may miss exact terms

This implementation supports:
- In-memory BM25 with persistence
- Elasticsearch backend (for production scale)
- MULTI-TENANT ISOLATION via organization_id filtering
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """BM25 search result."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class TextPreprocessor:
    """Text preprocessing for BM25."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        stemming: bool = False,
        min_token_length: int = 2
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.min_token_length = min_token_length

        # English stopwords
        self.stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "can", "will", "just", "should", "now", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "doing", "would", "could", "might", "must", "shall", "this",
            "that", "these", "those", "i", "me", "my", "myself", "we", "our",
            "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
            "its", "itself", "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom",
        }

        # Optional: Porter stemmer
        self.stemmer = None
        if stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                logger.warning("NLTK not installed, stemming disabled")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize and preprocess text."""
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens


class BM25Index:
    """
    In-memory BM25 index with persistence.

    Good for:
    - Small to medium datasets (< 1M documents)
    - Development/testing
    - Single-server deployments
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        preprocessor: Optional[TextPreprocessor] = None,
        persist_path: Optional[str] = None
    ):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0.75 typical)
            preprocessor: Text preprocessor
            persist_path: Path to persist index
        """
        self.k1 = k1
        self.b = b
        self.preprocessor = preprocessor or TextPreprocessor()
        self.persist_path = persist_path

        # Index state
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []  # Original documents
        self.tokenized_corpus: List[List[str]] = []  # Tokenized texts

        # Load from disk if exists
        if persist_path and Path(persist_path).exists():
            self.load()

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        id_field: str = "id"
    ) -> None:
        """
        Add documents to the index.

        Args:
            documents: List of dicts with text and id
                      MUST include 'organization_id' for tenant isolation
            text_field: Field containing text to index
            id_field: Field containing document ID

        Note: For multi-tenant isolation, documents SHOULD include:
            - organization_id (required for filtering)
            - workspace_id (optional)
            - access_level (optional, defaults to 'internal')
        """
        for doc in documents:
            if text_field not in doc:
                raise ValueError(f"Document missing '{text_field}' field")
            if id_field not in doc:
                raise ValueError(f"Document missing '{id_field}' field")
            # Note: organization_id validation happens at pipeline level

            text = doc[text_field]
            tokens = self.preprocessor.tokenize(text)

            self.documents.append(doc)
            self.tokenized_corpus.append(tokens)

        # Rebuild BM25 index
        self._rebuild_index()

        logger.info(f"Added {len(documents)} documents, total: {len(self.documents)}")

    def _rebuild_index(self):
        """Rebuild BM25 index from tokenized corpus."""
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b
            )

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[BM25Result]:
        """
        Search the index.

        Args:
            query: Search query
            limit: Maximum results
            filters: Metadata filters to apply

        Returns:
            List of BM25Result
        """
        if not self.bm25:
            return []

        # Tokenize query
        query_tokens = self.preprocessor.tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Create (index, score) pairs
        scored_docs = list(enumerate(scores))

        # Apply filters
        if filters:
            scored_docs = [
                (idx, score) for idx, score in scored_docs
                if self._matches_filter(self.documents[idx], filters)
            ]

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Get top results
        results = []
        for idx, score in scored_docs[:limit]:
            if score > 0:  # Only include positive scores
                doc = self.documents[idx]
                results.append(BM25Result(
                    id=doc.get("id", str(idx)),
                    score=float(score),
                    text=doc.get("text", ""),
                    metadata={k: v for k, v in doc.items() if k not in ("id", "text")}
                ))

        return results

    def _matches_filter(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches filters."""
        for key, value in filters.items():
            doc_value = doc.get(key)
            if isinstance(value, list):
                if doc_value not in value:
                    return False
            elif doc_value != value:
                return False
        return True

    def delete_by_ids(self, ids: List[str]) -> int:
        """Delete documents by ID."""
        ids_set = set(ids)
        original_count = len(self.documents)

        # Filter out deleted documents
        filtered = [
            (doc, tokens) for doc, tokens in zip(self.documents, self.tokenized_corpus)
            if doc.get("id") not in ids_set
        ]

        if filtered:
            self.documents, self.tokenized_corpus = zip(*filtered)
            self.documents = list(self.documents)
            self.tokenized_corpus = list(self.tokenized_corpus)
        else:
            self.documents = []
            self.tokenized_corpus = []

        self._rebuild_index()

        deleted_count = original_count - len(self.documents)
        logger.info(f"Deleted {deleted_count} documents")
        return deleted_count

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks belonging to a document."""
        ids_to_delete = [
            doc["id"] for doc in self.documents
            if doc.get("document_id") == document_id
        ]
        return self.delete_by_ids(ids_to_delete)

    def save(self, path: Optional[str] = None) -> None:
        """Persist index to disk."""
        save_path = path or self.persist_path
        if not save_path:
            raise ValueError("No persist path specified")

        data = {
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus,
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved BM25 index to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load index from disk."""
        load_path = path or self.persist_path
        if not load_path:
            raise ValueError("No persist path specified")

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self.k1 = data["k1"]
        self.b = data["b"]
        self.documents = data["documents"]
        self.tokenized_corpus = data["tokenized_corpus"]

        self._rebuild_index()
        logger.info(f"Loaded BM25 index from {load_path} ({len(self.documents)} docs)")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_documents": len(self.documents),
            "k1": self.k1,
            "b": self.b,
            "avg_doc_length": np.mean([len(t) for t in self.tokenized_corpus]) if self.tokenized_corpus else 0,
            "vocabulary_size": len(set(t for tokens in self.tokenized_corpus for t in tokens)) if self.tokenized_corpus else 0,
        }


class ElasticsearchBM25:
    """
    Elasticsearch-based BM25 for production scale.

    Use when:
    - Dataset > 1M documents
    - Need distributed search
    - Need advanced features (fuzzy matching, highlighting)
    """

    def __init__(
        self,
        index_name: str,
        hosts: List[str] = ["http://localhost:9200"],
        api_key: Optional[str] = None,
    ):
        from elasticsearch import Elasticsearch

        self.index_name = index_name

        # Connect
        if api_key:
            self.client = Elasticsearch(hosts=hosts, api_key=api_key)
        else:
            self.client = Elasticsearch(hosts=hosts)

        # Create index if not exists
        self._ensure_index()

    def _ensure_index(self):
        """Create index with BM25 settings."""
        if not self.client.indices.exists(index=self.index_name):
            settings = {
                "settings": {
                    "index": {
                        "number_of_shards": 2,
                        "number_of_replicas": 1,
                    },
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": "standard",
                                "stopwords": "_english_"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text", "analyzer": "default"},
                        "document_id": {"type": "keyword"},
                        "document_type": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "chunk_index": {"type": "integer"},
                        "metadata": {"type": "object", "enabled": False},
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=settings)
            logger.info(f"Created Elasticsearch index: {self.index_name}")

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> None:
        """Bulk index documents."""
        from elasticsearch.helpers import bulk

        def generate_actions():
            for doc in documents:
                yield {
                    "_index": self.index_name,
                    "_id": doc["id"],
                    "_source": doc,
                }

        success, failed = bulk(
            self.client,
            generate_actions(),
            chunk_size=batch_size,
            raise_on_error=False
        )
        logger.info(f"Indexed {success} documents, {len(failed)} failed")

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[BM25Result]:
        """Search using Elasticsearch BM25."""
        # Build query
        es_query = {
            "bool": {
                "must": [
                    {"match": {"text": query}}
                ]
            }
        }

        # Add filters
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {key: value}})
                else:
                    filter_clauses.append({"term": {key: value}})
            es_query["bool"]["filter"] = filter_clauses

        response = self.client.search(
            index=self.index_name,
            body={
                "query": es_query,
                "size": limit,
            }
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append(BM25Result(
                id=hit["_id"],
                score=hit["_score"],
                text=hit["_source"].get("text", ""),
                metadata={k: v for k, v in hit["_source"].items() if k not in ("id", "text")}
            ))

        return results

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        self.client.delete_by_query(
            index=self.index_name,
            body={
                "query": {
                    "term": {"document_id": document_id}
                }
            }
        )


# Example usage
if __name__ == "__main__":
    # Create index
    bm25 = BM25Index(
        k1=1.5,
        b=0.75,
        persist_path="./data/bm25_index.pkl"
    )

    # Add documents
    docs = [
        {"id": "1", "text": "This contract is between Party A and Party B", "document_id": "doc1"},
        {"id": "2", "text": "Payment terms are net 30 days from invoice date", "document_id": "doc1"},
        {"id": "3", "text": "The agreement shall commence on January 1, 2024", "document_id": "doc2"},
    ]

    bm25.add_documents(docs)

    # Search
    results = bm25.search("payment terms invoice", limit=5)
    for r in results:
        print(f"ID: {r.id}, Score: {r.score:.4f}, Text: {r.text[:50]}...")

    # Save index
    bm25.save()

    # Stats
    print(f"Stats: {bm25.get_stats()}")
