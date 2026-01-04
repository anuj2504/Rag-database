"""
Unified Chunk - Single chunk type for the entire RAG system.

NHAI/L&T Enterprise RAG System
==============================
This is the ONLY chunk type used throughout the system.
All chunking strategies output this format.

Key Features:
- Token-aware (from Chonkie)
- Structure-aware (from Hierarchical)
- Multi-tenant fields embedded at creation time
- Compatible with all storage backends
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid


class ChunkLevel(Enum):
    """Hierarchical chunk levels for retrieval optimization."""
    DOCUMENT = "document"      # Full document summary
    SECTION = "section"        # Major sections (Articles, Chapters)
    PARAGRAPH = "paragraph"    # Primary retrieval unit
    SENTENCE = "sentence"      # Precision queries
    TABLE = "table"            # Extracted tables
    FIGURE = "figure"          # Figures/diagrams (for ColPali)


class DocumentType(Enum):
    """
    Supported document types for NHAI/L&T.

    Each type has specific structure detection and metadata extraction.
    """
    # Legal/Contract Documents
    CONTRACT = "contract"
    AGREEMENT = "agreement"
    TENDER = "tender"
    AMENDMENT = "amendment"

    # Regulatory/Code Documents
    IRC_CODE = "irc_code"
    BUILDING_CODE = "building_code"
    SPECIFICATION = "specification"
    STANDARD = "standard"

    # Project Documents
    DPR = "dpr"  # Detailed Project Report
    FEASIBILITY = "feasibility"
    ESTIMATE = "estimate"
    BOQ = "boq"  # Bill of Quantities

    # Financial Documents
    FINANCIAL_REPORT = "financial_report"
    INVOICE = "invoice"
    BUDGET = "budget"

    # Technical Documents
    DRAWING = "drawing"
    MANUAL = "manual"
    SOP = "sop"  # Standard Operating Procedure

    # Correspondence
    LETTER = "letter"
    MEMO = "memo"
    CIRCULAR = "circular"

    # General
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class UnifiedChunk:
    """
    Single unified chunk type for the entire RAG system.

    This replaces:
    - ProcessedChunk (from document_processor.py)
    - HierarchicalChunk (from hierarchical_chunker.py)
    - EnterpriseChunk (from chonkie_chunker.py)

    All chunking strategies MUST output this format.
    All storage backends MUST accept this format.
    """

    # ========== IDENTITY ==========
    id: str                              # Unique chunk ID
    document_id: str                     # Parent document ID
    text: str                            # Chunk text content

    # ========== POSITION ==========
    chunk_index: int                     # Position in document
    page_number: Optional[int] = None    # Source page (if available)
    char_start: int = 0                  # Character offset start
    char_end: int = 0                    # Character offset end

    # ========== TOKEN INFO (from Chonkie) ==========
    token_count: int = 0                 # Token count for LLM limits

    # ========== HIERARCHY (from Hierarchical) ==========
    level: ChunkLevel = ChunkLevel.PARAGRAPH
    parent_id: Optional[str] = None      # Parent chunk ID
    children_ids: List[str] = field(default_factory=list)
    section_title: Optional[str] = None  # Section header
    section_number: Optional[str] = None # Section number (1.2.3)

    # ========== MULTI-TENANT (CRITICAL) ==========
    organization_id: str = ""            # REQUIRED - Tenant isolation
    workspace_id: Optional[str] = None   # Optional workspace
    collection_id: Optional[str] = None  # Optional collection
    access_level: str = "internal"       # public, internal, restricted, confidential

    # ========== DOCUMENT CONTEXT ==========
    document_type: str = "general"       # From DocumentType enum
    filename: Optional[str] = None       # Source filename
    element_type: Optional[str] = None   # Unstructured element type

    # ========== QUALITY & STRATEGY ==========
    quality_level: Optional[str] = None  # high, medium, low, garbage
    chunk_strategy: Optional[str] = None # sdpm, semantic, sentence, token

    # ========== FLEXIBLE METADATA ==========
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate required fields and set defaults."""
        if not self.id:
            self.id = self._generate_id()
        if not self.organization_id:
            raise ValueError("organization_id is REQUIRED for multi-tenant isolation")
        if self.char_end == 0 and self.text:
            self.char_end = len(self.text)

    def _generate_id(self) -> str:
        """Generate unique chunk ID."""
        content = f"{self.document_id}_{self.chunk_index}_{self.text[:50]}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"{self.document_id}_chunk_{self.chunk_index}_{hash_val}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            # Identity
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,

            # Position
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,

            # Token info
            "token_count": self.token_count,

            # Hierarchy
            "level": self.level.value if isinstance(self.level, ChunkLevel) else self.level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "section_title": self.section_title,
            "section_number": self.section_number,

            # Multi-tenant
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "collection_id": self.collection_id,
            "access_level": self.access_level,

            # Document context
            "document_type": self.document_type,
            "filename": self.filename,
            "element_type": self.element_type,

            # Quality
            "quality_level": self.quality_level,
            "chunk_strategy": self.chunk_strategy,

            # Metadata
            "metadata": self.metadata,
        }

    def to_vector_payload(self) -> Dict[str, Any]:
        """
        Convert to Qdrant vector payload.

        Includes only fields needed for search and filtering.
        Text is truncated to save space.
        """
        return {
            # Identity (for retrieval)
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text[:1000],  # Truncated for payload size

            # CRITICAL: Tenant isolation filters
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "collection_id": self.collection_id,
            "access_level": self.access_level,

            # Search filters
            "document_type": self.document_type,
            "level": self.level.value if isinstance(self.level, ChunkLevel) else self.level,
            "section_title": self.section_title,

            # Context
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "filename": self.filename,
        }

    def to_bm25_document(self) -> Dict[str, Any]:
        """Convert to BM25 index document."""
        return {
            "id": self.id,
            "text": self.text,
            "document_id": self.document_id,

            # CRITICAL: Tenant isolation
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "access_level": self.access_level,

            # Filters
            "document_type": self.document_type,
            "section_title": self.section_title,
        }

    def to_metadata_record(self) -> Dict[str, Any]:
        """Convert to PostgreSQL metadata record."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,

            # CRITICAL: Tenant isolation
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "access_level": self.access_level,

            # Hierarchy
            "level": self.level.value if isinstance(self.level, ChunkLevel) else self.level,
            "parent_id": self.parent_id,
            "section_title": self.section_title,

            # Element info
            "element_type": self.element_type or self.level.value,

            # Metadata as JSONB
            "metadata": {
                "token_count": self.token_count,
                "quality_level": self.quality_level,
                "chunk_strategy": self.chunk_strategy,
                "section_number": self.section_number,
                **self.metadata,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedChunk":
        """Create UnifiedChunk from dictionary."""
        level = data.get("level", "paragraph")
        if isinstance(level, str):
            try:
                level = ChunkLevel(level)
            except ValueError:
                level = ChunkLevel.PARAGRAPH

        return cls(
            id=data.get("id", ""),
            document_id=data["document_id"],
            text=data["text"],
            chunk_index=data.get("chunk_index", 0),
            page_number=data.get("page_number"),
            char_start=data.get("char_start", 0),
            char_end=data.get("char_end", 0),
            token_count=data.get("token_count", 0),
            level=level,
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            section_title=data.get("section_title"),
            section_number=data.get("section_number"),
            organization_id=data["organization_id"],
            workspace_id=data.get("workspace_id"),
            collection_id=data.get("collection_id"),
            access_level=data.get("access_level", "internal"),
            document_type=data.get("document_type", "general"),
            filename=data.get("filename"),
            element_type=data.get("element_type"),
            quality_level=data.get("quality_level"),
            chunk_strategy=data.get("chunk_strategy"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChunkingResult:
    """Result of chunking operation."""
    chunks: List[UnifiedChunk]
    document_id: str
    total_chunks: int
    total_tokens: int
    quality_level: str
    chunk_strategy: str
    sections_detected: int
    tables_detected: int
    processing_time_seconds: float

    def get_by_level(self, level: ChunkLevel) -> List[UnifiedChunk]:
        """Get chunks filtered by level."""
        return [c for c in self.chunks if c.level == level]

    def get_paragraph_chunks(self) -> List[UnifiedChunk]:
        """Get paragraph-level chunks (primary retrieval unit)."""
        return self.get_by_level(ChunkLevel.PARAGRAPH)
