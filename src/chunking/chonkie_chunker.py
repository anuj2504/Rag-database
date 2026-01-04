"""
Chonkie-based Intelligent Chunking.

==============================================================================
DEPRECATED: Use ChunkingService from src.chunking.chunking_service instead.

    from src.chunking import ChunkingService, UnifiedChunk

ChunkingService integrates Chonkie with structure detection for better results.
All Chonkie strategies (TOKEN, SENTENCE, SEMANTIC, SDPM) are available there.

This file is kept for backward compatibility only.
==============================================================================

Chonkie is a modern chunking library that provides:
- Semantic chunking (preserves meaning)
- Token-aware chunking (works with any tokenizer)
- Sentence-based chunking
- Recursive chunking for structured documents

This module integrates Chonkie with our hierarchical chunking strategy
for enterprise document processing.
"""
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

# Chonkie imports
from chonkie import (
    TokenChunker,
    SentenceChunker,
    SemanticChunker,  # With skip_window>0, provides SDPM behavior
    Chunk as ChonkieChunk,
)
from chonkie.embeddings import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)


class ChunkStrategy(Enum):
    """Chunking strategy selection."""
    TOKEN = "token"           # Fast, token-count based
    SENTENCE = "sentence"     # Sentence boundary aware
    SEMANTIC = "semantic"     # Embedding-based semantic coherence
    SDPM = "sdpm"            # Semantic Double-Pass Merge (best quality)


@dataclass
class EnterpriseChunk:
    """
    Enterprise-grade chunk with full context.

    Compatible with both Chonkie output and our hierarchical system.
    """
    id: str
    text: str

    # Position info
    document_id: str
    chunk_index: int

    # Token info (from Chonkie)
    token_count: int
    char_start: int = 0
    char_end: int = 0

    # Hierarchy info
    level: str = "paragraph"  # document, section, paragraph, sentence
    parent_id: Optional[str] = None
    section_title: Optional[str] = None
    section_number: Optional[str] = None

    # Multi-tenant fields
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    access_level: str = "internal"

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "text": self.text,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "level": self.level,
            "parent_id": self.parent_id,
            "section_title": self.section_title,
            "section_number": self.section_number,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "access_level": self.access_level,
            "metadata": self.metadata,
        }


class ChonkieChunkerWrapper:
    """
    Wrapper around Chonkie for enterprise document chunking.

    Provides:
    - Multiple chunking strategies
    - Automatic strategy selection based on document quality
    - Section-aware chunking for structured documents
    - Multi-tenant metadata propagation
    """

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.SENTENCE,
        chunk_size: int = 512,           # Target tokens per chunk
        chunk_overlap: int = 128,        # Overlap tokens
        min_chunk_size: int = 50,        # Minimum tokens
        embedding_model: str = "all-MiniLM-L6-v2",  # For semantic chunking
        tokenizer: str = "gpt2",         # Default tokenizer
    ):
        """
        Initialize Chonkie chunker.

        Args:
            strategy: Chunking strategy to use
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            embedding_model: Model for semantic chunking
            tokenizer: Tokenizer for token counting
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer

        # Initialize chunkers lazily
        self._token_chunker = None
        self._sentence_chunker = None
        self._semantic_chunker = None
        self._sdpm_chunker = None

    def _get_token_chunker(self) -> TokenChunker:
        """Get or create token chunker."""
        if self._token_chunker is None:
            self._token_chunker = TokenChunker(
                tokenizer=self.tokenizer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        return self._token_chunker

    def _get_sentence_chunker(self) -> SentenceChunker:
        """Get or create sentence chunker."""
        if self._sentence_chunker is None:
            self._sentence_chunker = SentenceChunker(
                tokenizer=self.tokenizer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                min_sentences_per_chunk=1,
            )
        return self._sentence_chunker

    def _get_semantic_chunker(self) -> SemanticChunker:
        """Get or create semantic chunker."""
        if self._semantic_chunker is None:
            embeddings = SentenceTransformerEmbeddings(self.embedding_model)
            self._semantic_chunker = SemanticChunker(
                embedding_model=embeddings,
                chunk_size=self.chunk_size,
                threshold=0.5,  # Semantic similarity threshold
            )
        return self._semantic_chunker

    def _get_sdpm_chunker(self) -> SemanticChunker:
        """Get or create SDPM chunker (SemanticChunker with skip_window for double-pass merging)."""
        if self._sdpm_chunker is None:
            embeddings = SentenceTransformerEmbeddings(self.embedding_model)
            self._sdpm_chunker = SemanticChunker(
                embedding_model=embeddings,
                chunk_size=self.chunk_size,
                threshold=0.5,  # similarity threshold
                skip_window=2,  # Enables SDPM-like double-pass merging
            )
        return self._sdpm_chunker

    def chunk(
        self,
        text: str,
        document_id: str,
        strategy: Optional[ChunkStrategy] = None,
        organization_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        access_level: str = "internal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EnterpriseChunk]:
        """
        Chunk text using specified or default strategy.

        Args:
            text: Document text to chunk
            document_id: Document identifier
            strategy: Override default strategy
            organization_id: Tenant organization ID
            workspace_id: Workspace within organization
            access_level: Access level (public, internal, restricted, confidential)
            metadata: Additional metadata to include

        Returns:
            List of EnterpriseChunk objects
        """
        use_strategy = strategy or self.strategy

        # Select chunker based on strategy
        if use_strategy == ChunkStrategy.TOKEN:
            chunker = self._get_token_chunker()
        elif use_strategy == ChunkStrategy.SENTENCE:
            chunker = self._get_sentence_chunker()
        elif use_strategy == ChunkStrategy.SEMANTIC:
            chunker = self._get_semantic_chunker()
        elif use_strategy == ChunkStrategy.SDPM:
            chunker = self._get_sdpm_chunker()
        else:
            chunker = self._get_sentence_chunker()

        # Perform chunking
        try:
            chonkie_chunks: List[ChonkieChunk] = chunker.chunk(text)
        except Exception as e:
            logger.warning(f"Chunking failed with {use_strategy}, falling back to token: {e}")
            chunker = self._get_token_chunker()
            chonkie_chunks = chunker.chunk(text)

        # Convert to enterprise chunks
        enterprise_chunks = []
        for i, chunk in enumerate(chonkie_chunks):
            # Skip chunks that are too small
            if chunk.token_count < self.min_chunk_size:
                continue

            enterprise_chunk = EnterpriseChunk(
                id=f"{document_id}_chunk_{i}",
                text=chunk.text,
                document_id=document_id,
                chunk_index=i,
                token_count=chunk.token_count,
                char_start=chunk.start_index if hasattr(chunk, 'start_index') else 0,
                char_end=chunk.end_index if hasattr(chunk, 'end_index') else len(chunk.text),
                level="paragraph",
                organization_id=organization_id,
                workspace_id=workspace_id,
                access_level=access_level,
                metadata=metadata or {},
            )
            enterprise_chunks.append(enterprise_chunk)

        logger.info(
            f"Created {len(enterprise_chunks)} chunks from document {document_id} "
            f"using {use_strategy.value} strategy"
        )

        return enterprise_chunks

    def chunk_with_sections(
        self,
        text: str,
        document_id: str,
        sections: List[Dict[str, Any]],
        organization_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        access_level: str = "internal",
    ) -> List[EnterpriseChunk]:
        """
        Chunk text with section awareness.

        Chunks within section boundaries while respecting hierarchy.

        Args:
            text: Full document text
            document_id: Document identifier
            sections: List of section dicts with 'start', 'end', 'title', 'number'
            organization_id: Tenant organization ID
            workspace_id: Workspace within organization
            access_level: Access level

        Returns:
            List of EnterpriseChunk with section context
        """
        if not sections:
            # No sections detected, chunk entire document
            return self.chunk(
                text, document_id,
                organization_id=organization_id,
                workspace_id=workspace_id,
                access_level=access_level,
            )

        all_chunks = []
        chunk_index = 0

        for section in sections:
            start = section.get('start', section.get('position', 0))
            end = section.get('end', len(text))
            section_text = text[start:end]

            if len(section_text.strip()) < 50:
                continue

            # Chunk this section
            section_chunks = self.chunk(
                section_text,
                document_id,
                organization_id=organization_id,
                workspace_id=workspace_id,
                access_level=access_level,
            )

            # Update with section context
            section_id = f"{document_id}_sec_{len(all_chunks)}"
            for chunk in section_chunks:
                chunk.id = f"{document_id}_chunk_{chunk_index}"
                chunk.chunk_index = chunk_index
                chunk.parent_id = section_id
                chunk.section_title = section.get('title')
                chunk.section_number = section.get('number')
                chunk.char_start += start
                chunk.char_end += start

                all_chunks.append(chunk)
                chunk_index += 1

        return all_chunks


class AdaptiveChunker:
    """
    Adaptive chunker that selects strategy based on document quality.

    Strategy selection:
    - HIGH quality → SDPM (best semantic coherence)
    - MEDIUM quality → Semantic (good balance)
    - LOW quality → Sentence (robust to noise)
    - GARBAGE quality → Token (just split by size)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.chonkie = ChonkieChunkerWrapper(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )

        self.quality_to_strategy = {
            "high": ChunkStrategy.SDPM,
            "medium": ChunkStrategy.SEMANTIC,
            "low": ChunkStrategy.SENTENCE,
            "garbage": ChunkStrategy.TOKEN,
        }

    def chunk(
        self,
        text: str,
        document_id: str,
        quality_level: str = "medium",
        sections: Optional[List[Dict[str, Any]]] = None,
        organization_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        access_level: str = "internal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EnterpriseChunk]:
        """
        Adaptively chunk based on document quality.

        Args:
            text: Document text
            document_id: Document identifier
            quality_level: Document quality (high, medium, low, garbage)
            sections: Optional section boundaries
            organization_id: Tenant organization ID
            workspace_id: Workspace within organization
            access_level: Access level
            metadata: Additional metadata

        Returns:
            List of EnterpriseChunk
        """
        strategy = self.quality_to_strategy.get(
            quality_level.lower(),
            ChunkStrategy.SENTENCE
        )

        logger.info(f"Using {strategy.value} strategy for {quality_level} quality document")

        if sections:
            chunks = self.chonkie.chunk_with_sections(
                text=text,
                document_id=document_id,
                sections=sections,
                organization_id=organization_id,
                workspace_id=workspace_id,
                access_level=access_level,
            )
        else:
            chunks = self.chonkie.chunk(
                text=text,
                document_id=document_id,
                strategy=strategy,
                organization_id=organization_id,
                workspace_id=workspace_id,
                access_level=access_level,
                metadata=metadata,
            )

        # Add quality info to metadata
        for chunk in chunks:
            chunk.metadata['quality_level'] = quality_level
            chunk.metadata['chunk_strategy'] = strategy.value

        return chunks


class LegalDocumentChunker:
    """
    Specialized chunker for legal documents.

    Handles:
    - Contract sections (Articles, Sections, Clauses)
    - Definitions blocks
    - Numbered paragraphs
    - Exhibits and schedules
    """

    # Legal section patterns
    SECTION_PATTERNS = [
        r'^(Article|ARTICLE)\s+(\d+|[IVXLC]+)[:\.]?\s*(.*)',
        r'^(Section|SECTION)\s+(\d+(?:\.\d+)*)[:\.]?\s*(.*)',
        r'^(\d+(?:\.\d+)*)\s+([A-Z][^.]+)',
        r'^(WHEREAS|RECITALS?|DEFINITIONS?)[:\s]*',
        r'^(Exhibit|EXHIBIT|Schedule|SCHEDULE)\s+([A-Z0-9]+)',
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        self.chonkie = ChonkieChunkerWrapper(
            strategy=ChunkStrategy.SENTENCE,  # Sentence-aware for legal
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.patterns = [re.compile(p, re.MULTILINE) for p in self.SECTION_PATTERNS]

    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect legal document sections."""
        sections = []
        lines = text.split('\n')
        current_pos = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            for pattern in self.patterns:
                match = pattern.match(line_stripped)
                if match:
                    section = {
                        'position': current_pos,
                        'line_number': i,
                        'raw_line': line,
                        'title': match.group(0),
                        'number': match.group(2) if len(match.groups()) > 1 else None,
                    }
                    sections.append(section)
                    break
            current_pos += len(line) + 1

        # Calculate end positions
        for i, section in enumerate(sections):
            if i + 1 < len(sections):
                section['end'] = sections[i + 1]['position']
            else:
                section['end'] = len(text)
            section['start'] = section['position']

        return sections

    def chunk(
        self,
        text: str,
        document_id: str,
        organization_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        access_level: str = "internal",
    ) -> List[EnterpriseChunk]:
        """Chunk legal document with section awareness."""
        sections = self.detect_sections(text)

        return self.chonkie.chunk_with_sections(
            text=text,
            document_id=document_id,
            sections=sections,
            organization_id=organization_id,
            workspace_id=workspace_id,
            access_level=access_level,
        )


# Example usage
if __name__ == "__main__":
    # Basic chunking
    chunker = ChonkieChunkerWrapper(
        strategy=ChunkStrategy.SENTENCE,
        chunk_size=256,
        chunk_overlap=50,
    )

    sample_text = """
    SERVICES AGREEMENT

    This Agreement is entered into as of January 1, 2024, between ABC Corporation
    ("Company") and XYZ Services Inc. ("Provider").

    ARTICLE 1: DEFINITIONS

    1.1 "Services" means the consulting services described in Exhibit A attached hereto.

    1.2 "Term" means the period commencing on the Effective Date and continuing for
    twelve (12) months thereafter, unless earlier terminated.

    ARTICLE 2: SCOPE OF SERVICES

    2.1 Provider shall provide the Services to Company in accordance with the
    specifications set forth in Exhibit A.

    2.2 Provider represents and warrants that it has the expertise, qualifications,
    and resources necessary to perform the Services in a professional manner.

    ARTICLE 3: COMPENSATION

    3.1 Company shall pay Provider the fee of $50,000 per month for the Services.

    3.2 Payment is due within thirty (30) days of receipt of Provider's invoice.
    """

    # Test basic chunking
    chunks = chunker.chunk(
        text=sample_text,
        document_id="contract_001",
        organization_id="org_acme",
        workspace_id="ws_legal",
        access_level="confidential",
    )

    print(f"Created {len(chunks)} chunks\n")
    for chunk in chunks:
        print(f"ID: {chunk.id}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Org: {chunk.organization_id}")
        print(f"  Text: {chunk.text[:100]}...")
        print()

    # Test legal document chunker
    print("\n--- Legal Document Chunker ---\n")
    legal_chunker = LegalDocumentChunker()
    legal_chunks = legal_chunker.chunk(
        text=sample_text,
        document_id="contract_002",
        organization_id="org_acme",
    )

    print(f"Created {len(legal_chunks)} chunks with section awareness\n")
    for chunk in legal_chunks[:3]:
        print(f"ID: {chunk.id}")
        print(f"  Section: {chunk.section_title}")
        print(f"  Text: {chunk.text[:80]}...")
        print()
