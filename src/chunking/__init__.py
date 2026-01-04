"""
Chunking Module - Intelligent Document Chunking for NHAI/L&T RAG System.

This module provides unified chunking that combines:
- Chonkie (semantic/token-aware chunking)
- Hierarchical (structure-aware chunking)

Primary Exports:
- ChunkingService: Main service for chunking documents
- UnifiedChunk: Single chunk type used throughout the system
- ChunkLevel: Hierarchy levels (DOCUMENT, SECTION, PARAGRAPH, SENTENCE)
- ChunkingResult: Result of chunking operation

Usage:
    from src.chunking import ChunkingService, UnifiedChunk, ChunkLevel

    service = ChunkingService()
    result = service.chunk(
        text="...",
        document_id="doc_123",
        tenant_context=tenant_context,
        document_type="contract",
        quality_level="high",
    )

    for chunk in result.chunks:
        print(f"{chunk.id}: {chunk.text[:50]}...")
"""

# Primary exports - use these
from src.chunking.unified_chunk import (
    UnifiedChunk,
    ChunkLevel,
    ChunkingResult,
    DocumentType,
)

from src.chunking.chunking_service import (
    ChunkingService,
    ChunkStrategy,
    QualityLevel,
    create_chunking_service,
)

# Legacy exports - for backward compatibility only
# These are deprecated and will be removed in future versions
from src.chunking.hierarchical_chunker import (
    HierarchicalChunker,
    HierarchicalChunk,
    ChunkLevel as HierarchicalChunkLevel,
    LegalDocumentDetector,
    FinancialDocumentDetector,
    QueryComplexityAnalyzer,
)

from src.chunking.chonkie_chunker import (
    ChonkieChunkerWrapper,
    AdaptiveChunker,
    LegalDocumentChunker,
    EnterpriseChunk,
    ChunkStrategy as ChonkieChunkStrategy,
)

__all__ = [
    # Primary exports (USE THESE)
    "ChunkingService",
    "UnifiedChunk",
    "ChunkLevel",
    "ChunkingResult",
    "DocumentType",
    "ChunkStrategy",
    "QualityLevel",
    "create_chunking_service",

    # Legacy exports (DEPRECATED)
    "HierarchicalChunker",
    "HierarchicalChunk",
    "LegalDocumentDetector",
    "FinancialDocumentDetector",
    "QueryComplexityAnalyzer",
    "ChonkieChunkerWrapper",
    "AdaptiveChunker",
    "LegalDocumentChunker",
    "EnterpriseChunk",
]
