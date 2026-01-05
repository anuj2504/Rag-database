"""
Unified Chunking Service - Merges Chonkie + Hierarchical Chunking.

NHAI/L&T Enterprise RAG System
==============================

This service provides intelligent document chunking that:
1. Detects document structure (sections, articles, clauses)
2. Selects chunking strategy based on document quality
3. Applies Chonkie algorithms within structural boundaries
4. Outputs UnifiedChunk with tenant fields embedded

Chunking Strategy Selection:
- HIGH quality   → SDPM (Semantic Double-Pass Merge) - best coherence
- MEDIUM quality → Semantic chunking - good balance
- LOW quality    → Sentence chunking - robust to noise
- GARBAGE        → Token chunking - just split by size

Document Type Handling:
- Contracts/Agreements → Legal structure detection (Articles, Sections)
- DPR/Specifications   → Technical structure detection
- Financial Reports    → Financial structure detection
- IRC/Building Codes   → Code section detection (§, CFR, etc.)
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
import re

from src.chunking.unified_chunk import UnifiedChunk, ChunkLevel, ChunkingResult, DocumentType
from src.metadata.tenant_schema import TenantContext

# Chonkie imports
try:
    from chonkie import TokenChunker, SentenceChunker, SemanticChunker
    from chonkie.embeddings import SentenceTransformerEmbeddings
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    logging.warning("Chonkie not available. Install with: pip install chonkie")

logger = logging.getLogger(__name__)


class ChunkStrategy(Enum):
    """Chunking strategy selection."""
    TOKEN = "token"           # Fast, fixed token count
    SENTENCE = "sentence"     # Sentence boundary aware
    SEMANTIC = "semantic"     # Embedding-based coherence
    SDPM = "sdpm"             # Semantic Double-Pass Merge (best)


class QualityLevel(Enum):
    """Document quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    GARBAGE = "garbage"


# Quality to strategy mapping
QUALITY_STRATEGY_MAP = {
    QualityLevel.HIGH: ChunkStrategy.SDPM,
    QualityLevel.MEDIUM: ChunkStrategy.SEMANTIC,
    QualityLevel.LOW: ChunkStrategy.SENTENCE,
    QualityLevel.GARBAGE: ChunkStrategy.TOKEN,
}


@dataclass
class DetectedSection:
    """Detected document section."""
    start: int
    end: int
    title: str
    number: Optional[str]
    level: int  # Nesting level (1, 2, 3...)
    line_number: int


class StructureDetector:
    """
    Detects document structure for NHAI/L&T document types.

    Handles:
    - Legal documents (contracts, agreements, tenders)
    - Technical documents (DPR, specifications, standards)
    - Regulatory documents (IRC, building codes, CFR)
    - Financial documents (reports, BOQ, estimates)
    """

    # Legal document patterns
    LEGAL_PATTERNS = [
        r'^(Article|ARTICLE)\s+([\dIVXLC]+)[:\.]?\s*(.*)',
        r'^(Section|SECTION)\s+(\d+(?:\.\d+)*)[:\.]?\s*(.*)',
        r'^(Clause|CLAUSE)\s+(\d+(?:\.\d+)*)[:\.]?\s*(.*)',
        r'^(\d+(?:\.\d+)*)\s+([A-Z][^.]+)',
        r'^\(([a-z])\)\s+(.+)',
        r'^(WHEREAS|RECITALS?|DEFINITIONS?|PREAMBLE)[:\s]*',
        r'^(Exhibit|EXHIBIT|Schedule|SCHEDULE|Annexure|ANNEXURE)\s+([A-Z0-9]+)',
    ]

    # Technical/DPR patterns
    TECHNICAL_PATTERNS = [
        r'^(Chapter|CHAPTER)\s+(\d+)[:\.]?\s*(.*)',
        r'^(\d+(?:\.\d+)*)\s+(SCOPE|DESIGN|SPECIFICATIONS?|REQUIREMENTS?)',
        r'^(Part|PART)\s+([A-Z0-9]+)[:\.]?\s*(.*)',
        r'^(Drawing|DRAWING)\s+No\.?\s*(\S+)',
    ]

    # IRC/Building Code patterns
    CODE_PATTERNS = [
        r'^§\s*(\d+(?:\.\d+)*)\s+(.*)',
        r'^(\d+)\s+CFR\s+(\d+(?:\.\d+)*)',
        r'^IRC\s+(\d+(?:\.\d+)*)',
        r'^IBC\s+(\d+(?:\.\d+)*)',
        r'^IS\s+(\d+)[:\s]',  # Indian Standards
        r'^NBC\s+(\d+(?:\.\d+)*)',  # National Building Code
    ]

    # Financial patterns
    FINANCIAL_PATTERNS = [
        r'^(Executive Summary|Management Discussion)',
        r'^(Balance Sheet|Income Statement|Cash Flow)',
        r'^(Notes to Financial Statements)',
        r'^(Q[1-4]\s+\d{4}|FY\s*\d{4})',
        r'^(Bill of Quantities|BOQ|Schedule of Rates)',
        r'^(Item|Sl\.?\s*No\.?)\s+(\d+)',
    ]

    def __init__(self, document_type: str = "general"):
        self.document_type = document_type
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns based on document type."""
        patterns = []

        if self.document_type in ["contract", "agreement", "tender", "amendment"]:
            patterns.extend(self.LEGAL_PATTERNS)
        elif self.document_type in ["dpr", "specification", "standard", "manual"]:
            patterns.extend(self.TECHNICAL_PATTERNS)
        elif self.document_type in ["irc_code", "building_code"]:
            patterns.extend(self.CODE_PATTERNS)
        elif self.document_type in ["financial_report", "boq", "estimate"]:
            patterns.extend(self.FINANCIAL_PATTERNS)
        else:
            # Use all patterns for unknown types
            patterns.extend(self.LEGAL_PATTERNS)
            patterns.extend(self.TECHNICAL_PATTERNS)

        self.patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in patterns]

    def detect_sections(self, text: str) -> List[DetectedSection]:
        """Detect document sections."""
        sections = []
        lines = text.split('\n')
        current_pos = 0

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                current_pos += len(line) + 1
                continue

            for pattern in self.patterns:
                match = pattern.match(line_stripped)
                if match:
                    groups = match.groups()
                    section = DetectedSection(
                        start=current_pos,
                        end=0,  # Will be set later
                        title=match.group(0),
                        number=groups[1] if len(groups) > 1 else groups[0],
                        level=self._determine_level(match.group(0)),
                        line_number=line_num,
                    )
                    sections.append(section)
                    break

            current_pos += len(line) + 1

        # Set end positions
        for i, section in enumerate(sections):
            if i + 1 < len(sections):
                section.end = sections[i + 1].start
            else:
                section.end = len(text)

        return sections

    def _determine_level(self, header: str) -> int:
        """Determine nesting level of section."""
        # Count dots for numbered sections
        dots = header.count('.')
        if dots >= 2:
            return 3
        elif dots == 1:
            return 2
        # Check for sub-patterns
        if re.match(r'^\([a-z]\)', header):
            return 3
        if re.match(r'^(Article|Chapter|Part)', header, re.IGNORECASE):
            return 1
        return 2


class ChunkingService:
    """
    Unified chunking service for NHAI/L&T Enterprise RAG.

    Merges Chonkie (semantic/token-aware) with Hierarchical (structure-aware)
    chunking into a single service.

    Usage:
        service = ChunkingService()
        result = service.chunk(
            text="...",
            document_id="doc_123",
            tenant_context=TenantContext(organization_id="nhai", ...),
            document_type="contract",
            quality_level="high",
        )
        chunks = result.chunks
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        tokenizer: str = "gpt2",
    ):
        """
        Initialize chunking service.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size in tokens
            embedding_model: Model for semantic chunking
            tokenizer: Tokenizer for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer

        # Lazy-loaded chunkers
        self._token_chunker = None
        self._sentence_chunker = None
        self._semantic_chunker = None
        self._sdpm_chunker = None

    # ========== CHONKIE CHUNKER GETTERS ==========

    def _get_token_chunker(self):
        """Get or create token chunker."""
        if not CHONKIE_AVAILABLE:
            return None
        if self._token_chunker is None:
            self._token_chunker = TokenChunker(
                tokenizer=self.tokenizer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        return self._token_chunker

    def _get_sentence_chunker(self):
        """Get or create sentence chunker."""
        if not CHONKIE_AVAILABLE:
            return None
        if self._sentence_chunker is None:
            self._sentence_chunker = SentenceChunker(
                tokenizer=self.tokenizer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                min_sentences_per_chunk=1,
            )
        return self._sentence_chunker

    def _get_semantic_chunker(self):
        """Get or create semantic chunker."""
        if not CHONKIE_AVAILABLE:
            return None
        if self._semantic_chunker is None:
            embeddings = SentenceTransformerEmbeddings(self.embedding_model)
            self._semantic_chunker = SemanticChunker(
                embedding_model=embeddings,
                chunk_size=self.chunk_size,
                threshold=0.5,  # semantic similarity threshold
            )
        return self._semantic_chunker

    def _get_sdpm_chunker(self):
        """Get or create SDPM chunker (SemanticChunker with skip_window for double-pass merging)."""
        if not CHONKIE_AVAILABLE:
            return None
        if self._sdpm_chunker is None:
            embeddings = SentenceTransformerEmbeddings(self.embedding_model)
            # In chonkie 1.5+, SDPM behavior is achieved via SemanticChunker with skip_window > 0
            self._sdpm_chunker = SemanticChunker(
                embedding_model=embeddings,
                chunk_size=self.chunk_size,
                threshold=0.5,
                skip_window=2,  # Enables SDPM-like double-pass merging
            )
        return self._sdpm_chunker

    def _get_chunker_for_strategy(self, strategy: ChunkStrategy):
        """Get appropriate chunker for strategy."""
        chunkers = {
            ChunkStrategy.TOKEN: self._get_token_chunker,
            ChunkStrategy.SENTENCE: self._get_sentence_chunker,
            ChunkStrategy.SEMANTIC: self._get_semantic_chunker,
            ChunkStrategy.SDPM: self._get_sdpm_chunker,
        }
        return chunkers.get(strategy, self._get_sentence_chunker)()

    # ========== MAIN CHUNKING METHOD ==========

    def chunk(
        self,
        text: str,
        document_id: str,
        tenant_context: TenantContext,
        document_type: str = "general",
        quality_level: str = "medium",
        filename: Optional[str] = None,
        detect_structure: bool = True,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> ChunkingResult:
        """
        Chunk document with intelligent strategy selection.

        Args:
            text: Document text to chunk
            document_id: Unique document identifier
            tenant_context: REQUIRED - Tenant context for isolation
            document_type: Type of document (contract, dpr, etc.)
            quality_level: Document quality (high, medium, low, garbage)
            filename: Source filename
            detect_structure: Whether to detect document structure
            custom_metadata: Additional metadata to include

        Returns:
            ChunkingResult with list of UnifiedChunk
        """
        start_time = datetime.now()

        # Validate tenant context
        if not tenant_context or not tenant_context.organization_id:
            raise ValueError("TenantContext with organization_id is REQUIRED")

        # Select strategy based on quality
        try:
            quality = QualityLevel(quality_level.lower())
        except ValueError:
            quality = QualityLevel.MEDIUM

        strategy = QUALITY_STRATEGY_MAP.get(quality, ChunkStrategy.SENTENCE)

        logger.info(
            f"Chunking document {document_id} "
            f"(type={document_type}, quality={quality.value}, strategy={strategy.value})"
        )

        # Detect structure if enabled
        sections = []
        if detect_structure:
            detector = StructureDetector(document_type)
            sections = detector.detect_sections(text)
            logger.info(f"Detected {len(sections)} sections")

        # Chunk the document
        if sections:
            chunks = self._chunk_with_structure(
                text=text,
                document_id=document_id,
                sections=sections,
                strategy=strategy,
                tenant_context=tenant_context,
                document_type=document_type,
                filename=filename,
                quality_level=quality.value,
                custom_metadata=custom_metadata,
            )
        else:
            chunks = self._chunk_flat(
                text=text,
                document_id=document_id,
                strategy=strategy,
                tenant_context=tenant_context,
                document_type=document_type,
                filename=filename,
                quality_level=quality.value,
                custom_metadata=custom_metadata,
            )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Calculate totals
        total_tokens = sum(c.token_count for c in chunks)

        result = ChunkingResult(
            chunks=chunks,
            document_id=document_id,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            quality_level=quality.value,
            chunk_strategy=strategy.value,
            sections_detected=len(sections),
            tables_detected=0,  # TODO: Integrate table detection
            processing_time_seconds=processing_time,
        )

        logger.info(
            f"Created {len(chunks)} chunks, {total_tokens} tokens "
            f"in {processing_time:.2f}s"
        )

        return result

    def _chunk_with_structure(
        self,
        text: str,
        document_id: str,
        sections: List[DetectedSection],
        strategy: ChunkStrategy,
        tenant_context: TenantContext,
        document_type: str,
        filename: Optional[str],
        quality_level: str,
        custom_metadata: Optional[Dict[str, Any]],
    ) -> List[UnifiedChunk]:
        """Chunk document respecting section boundaries."""
        all_chunks = []
        chunk_index = 0

        # Create document-level chunk (summary)
        doc_chunk = UnifiedChunk(
            id=f"{document_id}_doc",
            document_id=document_id,
            text=text[:2000] + "..." if len(text) > 2000 else text,
            chunk_index=0,
            level=ChunkLevel.DOCUMENT,
            organization_id=tenant_context.organization_id,
            workspace_id=tenant_context.workspace_id,
            collection_id=tenant_context.collection_id,
            access_level=tenant_context.access_level.value if hasattr(tenant_context.access_level, 'value') else str(tenant_context.access_level),
            document_type=document_type,
            filename=filename,
            quality_level=quality_level,
            chunk_strategy=strategy.value,
            metadata=custom_metadata or {},
        )
        all_chunks.append(doc_chunk)
        chunk_index += 1

        # Process each section
        for sec_idx, section in enumerate(sections):
            section_text = text[section.start:section.end].strip()

            if len(section_text) < self.min_chunk_size:
                continue

            # Create section-level chunk
            section_id = f"{document_id}_sec_{sec_idx}"
            section_chunk = UnifiedChunk(
                id=section_id,
                document_id=document_id,
                text=section_text[:2000] if len(section_text) > 2000 else section_text,
                chunk_index=chunk_index,
                char_start=section.start,
                char_end=section.end,
                level=ChunkLevel.SECTION,
                parent_id=doc_chunk.id,
                section_title=section.title,
                section_number=section.number,
                organization_id=tenant_context.organization_id,
                workspace_id=tenant_context.workspace_id,
                collection_id=tenant_context.collection_id,
                access_level=tenant_context.access_level.value if hasattr(tenant_context.access_level, 'value') else str(tenant_context.access_level),
                document_type=document_type,
                filename=filename,
                quality_level=quality_level,
                chunk_strategy=strategy.value,
                metadata=custom_metadata or {},
            )
            all_chunks.append(section_chunk)
            chunk_index += 1

            # Chunk within section using Chonkie
            para_chunks = self._apply_chonkie(
                text=section_text,
                document_id=document_id,
                strategy=strategy,
                start_index=chunk_index,
                parent_id=section_id,
                section_title=section.title,
                section_number=section.number,
                char_offset=section.start,
                tenant_context=tenant_context,
                document_type=document_type,
                filename=filename,
                quality_level=quality_level,
                custom_metadata=custom_metadata,
            )

            all_chunks.extend(para_chunks)
            chunk_index += len(para_chunks)

            # Update section's children
            section_chunk.children_ids = [c.id for c in para_chunks]

        # Update document's children
        doc_chunk.children_ids = [
            c.id for c in all_chunks
            if c.level == ChunkLevel.SECTION
        ]

        return all_chunks

    def _chunk_flat(
        self,
        text: str,
        document_id: str,
        strategy: ChunkStrategy,
        tenant_context: TenantContext,
        document_type: str,
        filename: Optional[str],
        quality_level: str,
        custom_metadata: Optional[Dict[str, Any]],
    ) -> List[UnifiedChunk]:
        """Chunk document without structure detection."""
        return self._apply_chonkie(
            text=text,
            document_id=document_id,
            strategy=strategy,
            start_index=0,
            parent_id=None,
            section_title=None,
            section_number=None,
            char_offset=0,
            tenant_context=tenant_context,
            document_type=document_type,
            filename=filename,
            quality_level=quality_level,
            custom_metadata=custom_metadata,
        )

    def _apply_chonkie(
        self,
        text: str,
        document_id: str,
        strategy: ChunkStrategy,
        start_index: int,
        parent_id: Optional[str],
        section_title: Optional[str],
        section_number: Optional[str],
        char_offset: int,
        tenant_context: TenantContext,
        document_type: str,
        filename: Optional[str],
        quality_level: str,
        custom_metadata: Optional[Dict[str, Any]],
    ) -> List[UnifiedChunk]:
        """Apply Chonkie chunking and convert to UnifiedChunk."""
        chunks = []

        if CHONKIE_AVAILABLE:
            chunker = self._get_chunker_for_strategy(strategy)
            if chunker:
                try:
                    chonkie_chunks = chunker.chunk(text)

                    kept_count = 0  # Track actual kept chunks to avoid ID collisions
                    for cc in chonkie_chunks:
                        if cc.token_count < self.min_chunk_size:
                            continue

                        chunk_idx = start_index + kept_count
                        chunk = UnifiedChunk(
                            id=f"{document_id}_chunk_{chunk_idx}",
                            document_id=document_id,
                            text=cc.text,
                            chunk_index=chunk_idx,
                            char_start=char_offset + (cc.start_index if hasattr(cc, 'start_index') else 0),
                            char_end=char_offset + (cc.end_index if hasattr(cc, 'end_index') else len(cc.text)),
                            token_count=cc.token_count,
                            level=ChunkLevel.PARAGRAPH,
                            parent_id=parent_id,
                            section_title=section_title,
                            section_number=section_number,
                            organization_id=tenant_context.organization_id,
                            workspace_id=tenant_context.workspace_id,
                            collection_id=tenant_context.collection_id,
                            access_level=tenant_context.access_level.value if hasattr(tenant_context.access_level, 'value') else str(tenant_context.access_level),
                            document_type=document_type,
                            filename=filename,
                            quality_level=quality_level,
                            chunk_strategy=strategy.value,
                            metadata=custom_metadata or {},
                        )
                        chunks.append(chunk)
                        kept_count += 1

                    return chunks

                except Exception as e:
                    logger.warning(f"Chonkie chunking failed, falling back: {e}")

        # Fallback: Simple paragraph-based chunking
        return self._fallback_chunk(
            text=text,
            document_id=document_id,
            start_index=start_index,
            parent_id=parent_id,
            section_title=section_title,
            section_number=section_number,
            char_offset=char_offset,
            tenant_context=tenant_context,
            document_type=document_type,
            filename=filename,
            quality_level=quality_level,
            custom_metadata=custom_metadata,
        )

    def _fallback_chunk(
        self,
        text: str,
        document_id: str,
        start_index: int,
        parent_id: Optional[str],
        section_title: Optional[str],
        section_number: Optional[str],
        char_offset: int,
        tenant_context: TenantContext,
        document_type: str,
        filename: Optional[str],
        quality_level: str,
        custom_metadata: Optional[Dict[str, Any]],
    ) -> List[UnifiedChunk]:
        """Fallback paragraph-based chunking when Chonkie unavailable."""
        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk = ""
        chunk_index = start_index
        char_pos = char_offset

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds target size, save current
            if len(current_chunk.split()) + len(para.split()) > self.chunk_size and current_chunk:
                chunks.append(self._create_fallback_chunk(
                    text=current_chunk,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    char_start=char_pos,
                    parent_id=parent_id,
                    section_title=section_title,
                    section_number=section_number,
                    tenant_context=tenant_context,
                    document_type=document_type,
                    filename=filename,
                    quality_level=quality_level,
                    custom_metadata=custom_metadata,
                ))
                chunk_index += 1
                char_pos += len(current_chunk) + 2
                current_chunk = para
            else:
                current_chunk = (current_chunk + "\n\n" + para).strip()

        # Don't forget last chunk
        if current_chunk:
            chunks.append(self._create_fallback_chunk(
                text=current_chunk,
                document_id=document_id,
                chunk_index=chunk_index,
                char_start=char_pos,
                parent_id=parent_id,
                section_title=section_title,
                section_number=section_number,
                tenant_context=tenant_context,
                document_type=document_type,
                filename=filename,
                quality_level=quality_level,
                custom_metadata=custom_metadata,
            ))

        return chunks

    def _create_fallback_chunk(
        self,
        text: str,
        document_id: str,
        chunk_index: int,
        char_start: int,
        parent_id: Optional[str],
        section_title: Optional[str],
        section_number: Optional[str],
        tenant_context: TenantContext,
        document_type: str,
        filename: Optional[str],
        quality_level: str,
        custom_metadata: Optional[Dict[str, Any]],
    ) -> UnifiedChunk:
        """Create a fallback chunk."""
        return UnifiedChunk(
            id=f"{document_id}_chunk_{chunk_index}",
            document_id=document_id,
            text=text,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_start + len(text),
            token_count=len(text.split()),  # Approximate
            level=ChunkLevel.PARAGRAPH,
            parent_id=parent_id,
            section_title=section_title,
            section_number=section_number,
            organization_id=tenant_context.organization_id,
            workspace_id=tenant_context.workspace_id,
            collection_id=tenant_context.collection_id,
            access_level=tenant_context.access_level.value if hasattr(tenant_context.access_level, 'value') else str(tenant_context.access_level),
            document_type=document_type,
            filename=filename,
            quality_level=quality_level,
            chunk_strategy="fallback",
            metadata=custom_metadata or {},
        )


# Convenience function
def create_chunking_service(
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> ChunkingService:
    """Create a configured chunking service."""
    return ChunkingService(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
