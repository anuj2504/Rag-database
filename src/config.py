"""Configuration settings for the RAG system."""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    POSTGRES_URL: str = "postgresql://user:password@localhost:5432/rag_db"

    # Vector Store
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_DENSE: str = "documents_dense"
    QDRANT_COLLECTION_COLPALI: str = "documents_colpali"

    # Embeddings
    DENSE_MODEL: str = "BAAI/bge-base-en-v1.5"  # 768 dims, good quality
    DENSE_DIMENSION: int = 768
    OPENAI_API_KEY: Optional[str] = None

    # ColPali
    COLPALI_MODEL: str = "vidore/colpali-v1.2"
    COLPALI_DIMENSION: int = 128  # Per-patch dimension

    # Processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_IMAGE_SIZE: int = 1024  # For ColPali processing

    # BM25
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    class Config:
        env_file = ".env"


settings = Settings()
