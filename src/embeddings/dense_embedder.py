"""Dense text embeddings using sentence-transformers or OpenAI."""
from typing import List, Union, Optional
import numpy as np
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Base class for embedding models."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (may use different prompt)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embeddings using sentence-transformers.
    Good models:
    - BAAI/bge-base-en-v1.5 (768d, fast, good quality)
    - BAAI/bge-large-en-v1.5 (1024d, better quality)
    - intfloat/e5-large-v2 (1024d, excellent quality)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()

        # BGE models need specific prefixes for best results
        self.query_prefix = ""
        self.document_prefix = ""
        if "bge" in model_name.lower():
            self.query_prefix = "Represent this sentence for searching relevant passages: "

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed documents/chunks."""
        # Add prefix for document embeddings if needed
        prefixed_texts = [self.document_prefix + t for t in texts]
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True  # Important for cosine similarity
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query."""
        prefixed_query = self.query_prefix + query
        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True
        )
        return np.array(embedding)

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedder(BaseEmbedder):
    """
    Embeddings using OpenAI API.
    Models:
    - text-embedding-3-small (1536d, cheap)
    - text-embedding-3-large (3072d, best quality)
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100
    ):
        from openai import OpenAI
        import os

        self.model_name = model_name
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        # Dimensions by model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = self._dimensions.get(model_name, 1536)

    def _batch_embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts in batches."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed documents/chunks."""
        return self._batch_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=query
        )
        return np.array(response.data[0].embedding)

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedder(
    provider: str = "sentence-transformers",
    model_name: Optional[str] = None,
    **kwargs
) -> BaseEmbedder:
    """Factory function to create embedder."""
    if provider == "sentence-transformers":
        model = model_name or "BAAI/bge-base-en-v1.5"
        return SentenceTransformerEmbedder(model_name=model, **kwargs)
    elif provider == "openai":
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbedder(model_name=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Example usage
if __name__ == "__main__":
    # Using sentence-transformers (no API key needed)
    embedder = get_embedder("sentence-transformers")

    texts = [
        "This contract is between Party A and Party B",
        "The agreement shall commence on January 1, 2024",
        "Payment terms are net 30 days"
    ]

    embeddings = embedder.embed_texts(texts)
    print(f"Embedded {len(texts)} texts, shape: {embeddings.shape}")

    query_embedding = embedder.embed_query("What are the payment terms?")
    print(f"Query embedding shape: {query_embedding.shape}")
