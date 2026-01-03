"""
Hierarchical Chunking with Document Structure Preservation.

Key insight from enterprise RAG:
- Documents have structure (sections, paragraphs, sentences)
- Fixed-size chunking destroys this structure
- Query complexity should determine retrieval level

This module provides:
- Multi-level chunking (document → section → paragraph → sentence)
- Structure detection for different document types
- Query-complexity-based retrieval level selection
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ChunkLevel(Enum):
    """Hierarchical chunk levels."""
    DOCUMENT = "document"    # Full document metadata
    SECTION = "section"      # Major sections (chapters, articles)
    PARAGRAPH = "paragraph"  # Individual paragraphs (primary retrieval unit)
    SENTENCE = "sentence"    # Sentence-level for precision queries


@dataclass
class HierarchicalChunk:
    """A chunk with hierarchical context."""
    id: str
    text: str
    level: ChunkLevel

    # Position info
    document_id: str
    section_id: Optional[str] = None
    paragraph_index: int = 0
    sentence_index: Optional[int] = None

    # Hierarchy context (for retrieval context)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Section context
    section_title: Optional[str] = None
    section_number: Optional[str] = None

    # Metadata
    char_start: int = 0
    char_end: int = 0
    word_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentStructure:
    """Parsed document structure."""
    title: Optional[str] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructureDetector(ABC):
    """Abstract base for document structure detection."""

    @abstractmethod
    def detect_structure(self, text: str) -> DocumentStructure:
        pass


class LegalDocumentDetector(StructureDetector):
    """
    Structure detection for legal documents.
    Handles: Contracts, Agreements, IRC codes, Building codes, Regulations
    """

    # Legal document section patterns
    SECTION_PATTERNS = [
        # Article/Section numbering
        r'^(Article|Section|ARTICLE|SECTION)\s+(\d+(?:\.\d+)*)[:\.]?\s*(.*)$',
        # Roman numeral sections
        r'^(I{1,3}|IV|VI{0,3}|IX|X{1,3})\.\s+(.+)$',
        # Numbered sections (1., 1.1, 1.1.1)
        r'^(\d+(?:\.\d+)*)\s+([A-Z][^.]+)$',
        # Lettered sections ((a), (b), etc.)
        r'^\(([a-z])\)\s+(.+)$',
        # Definition sections
        r'^"([^"]+)"\s+means\s+',
        # WHEREAS clauses
        r'^WHEREAS[,:]?\s+',
        # RECITALS
        r'^RECITALS?:?\s*$',
    ]

    # IRC/Building code specific patterns
    CODE_PATTERNS = [
        r'^§\s*(\d+(?:\.\d+)*)\s+(.+)$',  # § 123.45 Section Title
        r'^(\d+)\s+CFR\s+(\d+(?:\.\d+)*)',  # Federal regulations
        r'^IRC\s+(\d+(?:\.\d+)*)',  # IRC references
        r'^IBC\s+(\d+(?:\.\d+)*)',  # International Building Code
    ]

    def __init__(self):
        self.section_patterns = [re.compile(p, re.MULTILINE) for p in self.SECTION_PATTERNS]
        self.code_patterns = [re.compile(p, re.MULTILINE) for p in self.CODE_PATTERNS]

    def detect_structure(self, text: str) -> DocumentStructure:
        """Detect legal document structure."""
        structure = DocumentStructure()
        lines = text.split('\n')

        # Detect title (usually first non-empty line in caps or specific format)
        structure.title = self._detect_title(lines)

        # Detect sections
        structure.sections = self._detect_sections(text, lines)

        # Detect if this is a code document
        structure.metadata['is_code'] = self._is_code_document(text)
        structure.metadata['document_subtype'] = self._detect_subtype(text)

        return structure

    def _detect_title(self, lines: List[str]) -> Optional[str]:
        """Detect document title."""
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue

            # Title patterns
            if re.match(r'^[A-Z][A-Z\s]+$', line):  # ALL CAPS
                return line
            if re.match(r'^(Agreement|Contract|Amendment|Exhibit)', line, re.IGNORECASE):
                return line

        return None

    def _detect_sections(self, text: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect document sections."""
        sections = []
        current_position = 0

        for i, line in enumerate(lines):
            for pattern in self.section_patterns + self.code_patterns:
                match = pattern.match(line.strip())
                if match:
                    section = {
                        'line_number': i,
                        'position': current_position,
                        'raw_line': line,
                        'level': self._determine_section_level(match.group(0)),
                    }

                    # Extract section number and title
                    groups = match.groups()
                    if len(groups) >= 2:
                        section['number'] = groups[0]
                        section['title'] = groups[1] if len(groups) > 1 else None

                    sections.append(section)
                    break

            current_position += len(line) + 1

        return sections

    def _determine_section_level(self, header: str) -> int:
        """Determine nesting level of section."""
        # Count dots in section number for level
        dots = header.count('.')
        if dots >= 2:
            return 3
        elif dots == 1:
            return 2
        else:
            return 1

    def _is_code_document(self, text: str) -> bool:
        """Check if document is a code/regulation."""
        code_indicators = ['IRC', 'IBC', 'CFR', '§', 'Code', 'Regulation']
        text_lower = text.lower()
        return sum(1 for ind in code_indicators if ind.lower() in text_lower) >= 2

    def _detect_subtype(self, text: str) -> str:
        """Detect specific document subtype."""
        text_lower = text.lower()

        if 'irc' in text_lower or 'internal revenue' in text_lower:
            return 'irc_code'
        elif 'ibc' in text_lower or 'building code' in text_lower:
            return 'building_code'
        elif 'cfr' in text_lower or 'federal regulation' in text_lower:
            return 'federal_regulation'
        elif 'agreement' in text_lower or 'contract' in text_lower:
            return 'contract'
        elif 'whereas' in text_lower:
            return 'legal_agreement'
        else:
            return 'general_legal'


class FinancialDocumentDetector(StructureDetector):
    """
    Structure detection for financial documents.
    Handles: Financial reports, Audits, Budgets, Forecasts
    """

    SECTION_PATTERNS = [
        r'^(Executive Summary|Management Discussion)',
        r'^(Financial Highlights|Key Metrics)',
        r'^(Balance Sheet|Income Statement|Cash Flow)',
        r'^(Notes to Financial Statements)',
        r'^(Q[1-4]\s+\d{4}|FY\s*\d{4})',
        r'^(Revenue|Expenses|Assets|Liabilities)',
        r'^(Appendix|Exhibit)\s+[A-Z0-9]+',
    ]

    def __init__(self):
        self.section_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE)
                                  for p in self.SECTION_PATTERNS]

    def detect_structure(self, text: str) -> DocumentStructure:
        """Detect financial document structure."""
        structure = DocumentStructure()
        lines = text.split('\n')

        structure.title = self._detect_title(lines)
        structure.sections = self._detect_sections(text)

        # Financial-specific metadata
        structure.metadata['fiscal_periods'] = self._extract_fiscal_periods(text)
        structure.metadata['has_tables'] = self._has_financial_tables(text)

        return structure

    def _detect_title(self, lines: List[str]) -> Optional[str]:
        """Detect financial document title."""
        for line in lines[:15]:
            line = line.strip()
            if any(term in line.lower() for term in
                   ['annual report', 'quarterly report', 'financial statement',
                    'earnings', 'budget', 'forecast']):
                return line
        return None

    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect financial document sections."""
        sections = []
        for pattern in self.section_patterns:
            for match in pattern.finditer(text):
                sections.append({
                    'position': match.start(),
                    'title': match.group(0),
                    'level': 1
                })
        return sorted(sections, key=lambda x: x['position'])

    def _extract_fiscal_periods(self, text: str) -> List[str]:
        """Extract mentioned fiscal periods."""
        periods = []
        # Q1 2023, FY2022, etc.
        pattern = r'(Q[1-4]\s*\d{4}|FY\s*\d{4}|\d{4}\s*Q[1-4])'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(set(matches))

    def _has_financial_tables(self, text: str) -> bool:
        """Check for financial table indicators."""
        table_terms = ['$', 'revenue', 'total', 'subtotal', 'balance']
        return sum(1 for term in table_terms if term in text.lower()) >= 3


class HierarchicalChunker:
    """
    Creates hierarchical chunks from documents with structure preservation.

    Key features:
    - Maintains parent-child relationships between chunks
    - Preserves section context for each chunk
    - Enables query-complexity-based retrieval
    """

    def __init__(
        self,
        paragraph_size: int = 400,
        paragraph_overlap: int = 50,
        sentence_size: int = 100,
        min_section_size: int = 100,
        detector: Optional[StructureDetector] = None
    ):
        self.paragraph_size = paragraph_size
        self.paragraph_overlap = paragraph_overlap
        self.sentence_size = sentence_size
        self.min_section_size = min_section_size
        self.detector = detector or LegalDocumentDetector()

        # Sentence splitting pattern
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=[A-Z0-9])'
        )

    def chunk(
        self,
        text: str,
        document_id: str,
        detect_structure: bool = True
    ) -> List[HierarchicalChunk]:
        """
        Create hierarchical chunks from document.

        Args:
            text: Document text
            document_id: Unique document ID
            detect_structure: Whether to detect document structure

        Returns:
            List of HierarchicalChunk at all levels
        """
        chunks = []

        # Detect structure
        if detect_structure:
            structure = self.detector.detect_structure(text)
        else:
            structure = DocumentStructure()

        # Create document-level chunk
        doc_chunk = HierarchicalChunk(
            id=f"{document_id}_doc",
            text=text[:2000] + "..." if len(text) > 2000 else text,  # Summary
            level=ChunkLevel.DOCUMENT,
            document_id=document_id,
            word_count=len(text.split()),
            metadata={
                'title': structure.title,
                'structure': structure.metadata
            }
        )
        chunks.append(doc_chunk)

        # Create section-level chunks
        section_chunks = self._create_section_chunks(
            text, document_id, structure, doc_chunk.id
        )
        doc_chunk.children_ids = [c.id for c in section_chunks]
        chunks.extend(section_chunks)

        # Create paragraph-level chunks within each section
        for section_chunk in section_chunks:
            para_chunks = self._create_paragraph_chunks(
                section_chunk.text, document_id,
                section_chunk.id, section_chunk.section_title
            )
            section_chunk.children_ids = [c.id for c in para_chunks]
            chunks.extend(para_chunks)

            # Create sentence-level chunks for precision queries
            for para_chunk in para_chunks:
                sent_chunks = self._create_sentence_chunks(
                    para_chunk.text, document_id, para_chunk.id
                )
                para_chunk.children_ids = [c.id for c in sent_chunks]
                chunks.extend(sent_chunks)

        return chunks

    def _create_section_chunks(
        self,
        text: str,
        document_id: str,
        structure: DocumentStructure,
        parent_id: str
    ) -> List[HierarchicalChunk]:
        """Create section-level chunks."""
        chunks = []

        if not structure.sections:
            # No sections detected - treat whole document as one section
            chunk = HierarchicalChunk(
                id=f"{document_id}_sec_0",
                text=text,
                level=ChunkLevel.SECTION,
                document_id=document_id,
                section_id=f"{document_id}_sec_0",
                parent_id=parent_id,
                section_title="Document Content",
                word_count=len(text.split())
            )
            return [chunk]

        # Create chunks based on detected sections
        sections = structure.sections
        for i, section in enumerate(sections):
            # Get section text (from this section to next, or end)
            start = section.get('position', 0)
            if i + 1 < len(sections):
                end = sections[i + 1].get('position', len(text))
            else:
                end = len(text)

            section_text = text[start:end].strip()

            if len(section_text) < self.min_section_size:
                continue

            chunk = HierarchicalChunk(
                id=f"{document_id}_sec_{i}",
                text=section_text,
                level=ChunkLevel.SECTION,
                document_id=document_id,
                section_id=f"{document_id}_sec_{i}",
                parent_id=parent_id,
                section_title=section.get('title'),
                section_number=section.get('number'),
                char_start=start,
                char_end=end,
                word_count=len(section_text.split())
            )
            chunks.append(chunk)

        return chunks

    def _create_paragraph_chunks(
        self,
        text: str,
        document_id: str,
        parent_id: str,
        section_title: Optional[str]
    ) -> List[HierarchicalChunk]:
        """Create paragraph-level chunks with overlap."""
        chunks = []

        # Split by double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)

        # Merge small paragraphs, split large ones
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds size, save current and start new
            if len(current_chunk) + len(para) > self.paragraph_size and current_chunk:
                chunk = self._create_para_chunk(
                    current_chunk, document_id, parent_id,
                    section_title, chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1

                # Keep overlap from end of previous chunk
                if self.paragraph_overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-self.paragraph_overlap:] if len(words) > self.paragraph_overlap else words
                    current_chunk = ' '.join(overlap_words) + ' ' + para
                else:
                    current_chunk = para
            else:
                current_chunk = (current_chunk + '\n\n' + para).strip()

        # Don't forget the last chunk
        if current_chunk:
            chunk = self._create_para_chunk(
                current_chunk, document_id, parent_id,
                section_title, chunk_index
            )
            chunks.append(chunk)

        return chunks

    def _create_para_chunk(
        self,
        text: str,
        document_id: str,
        parent_id: str,
        section_title: Optional[str],
        index: int
    ) -> HierarchicalChunk:
        """Helper to create a paragraph chunk."""
        return HierarchicalChunk(
            id=f"{parent_id}_para_{index}",
            text=text,
            level=ChunkLevel.PARAGRAPH,
            document_id=document_id,
            parent_id=parent_id,
            paragraph_index=index,
            section_title=section_title,
            word_count=len(text.split())
        )

    def _create_sentence_chunks(
        self,
        text: str,
        document_id: str,
        parent_id: str
    ) -> List[HierarchicalChunk]:
        """Create sentence-level chunks for precision queries."""
        chunks = []

        # Split into sentences
        sentences = self.sentence_pattern.split(text)

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short fragments
                continue

            chunk = HierarchicalChunk(
                id=f"{parent_id}_sent_{i}",
                text=sentence,
                level=ChunkLevel.SENTENCE,
                document_id=document_id,
                parent_id=parent_id,
                sentence_index=i,
                word_count=len(sentence.split())
            )
            chunks.append(chunk)

        return chunks

    def get_chunks_by_level(
        self,
        chunks: List[HierarchicalChunk],
        level: ChunkLevel
    ) -> List[HierarchicalChunk]:
        """Filter chunks by level."""
        return [c for c in chunks if c.level == level]


class QueryComplexityAnalyzer:
    """
    Analyzes query complexity to determine optimal retrieval level.

    Key insight: Different queries need different chunk granularities
    - Broad: "What does this contract cover?" → paragraph level
    - Precise: "What is the exact value in Table 3?" → sentence level
    """

    # Keywords indicating precision queries
    PRECISION_KEYWORDS = [
        'exact', 'specific', 'precisely', 'exactly',
        'table', 'figure', 'section', 'article', 'clause',
        'number', 'amount', 'value', 'date', 'deadline',
        'line', 'paragraph', 'sentence',
        'what is the', 'what was the',
    ]

    # Keywords indicating broad queries
    BROAD_KEYWORDS = [
        'overview', 'summary', 'about', 'describe',
        'explain', 'what does', 'how does', 'why does',
        'generally', 'overall', 'main', 'key points',
    ]

    def analyze(self, query: str) -> Tuple[ChunkLevel, float]:
        """
        Analyze query to determine optimal chunk level.

        Returns:
            Tuple of (recommended_level, confidence)
        """
        query_lower = query.lower()

        # Count precision vs broad indicators
        precision_score = sum(
            1 for kw in self.PRECISION_KEYWORDS
            if kw in query_lower
        )
        broad_score = sum(
            1 for kw in self.BROAD_KEYWORDS
            if kw in query_lower
        )

        # Question length is also an indicator
        # Longer, more specific questions often need precision
        word_count = len(query.split())

        if precision_score > broad_score:
            if precision_score >= 2:
                return ChunkLevel.SENTENCE, 0.9
            else:
                return ChunkLevel.SENTENCE, 0.7
        elif broad_score > precision_score:
            if broad_score >= 2:
                return ChunkLevel.SECTION, 0.9
            else:
                return ChunkLevel.PARAGRAPH, 0.7
        else:
            # Default to paragraph for balanced queries
            return ChunkLevel.PARAGRAPH, 0.5


# Example usage
if __name__ == "__main__":
    # Sample contract text
    contract_text = """
    SERVICES AGREEMENT

    This Agreement is entered into as of January 1, 2024.

    ARTICLE 1: DEFINITIONS

    1.1 "Company" means ABC Corporation, a Delaware corporation.

    1.2 "Services" means the consulting services described in Exhibit A.

    1.3 "Term" means the period from the Effective Date until termination.

    ARTICLE 2: SCOPE OF SERVICES

    2.1 The Company shall provide Services to Client as described herein.

    2.2 Services shall be performed in a professional manner consistent
    with industry standards. The Company represents that it has the
    expertise and qualifications necessary to perform the Services.

    ARTICLE 3: COMPENSATION

    3.1 Client shall pay Company the fee of $50,000 per month.

    3.2 Payment is due within 30 days of invoice receipt.
    """

    # Create chunker with legal detector
    chunker = HierarchicalChunker(
        detector=LegalDocumentDetector(),
        paragraph_size=300
    )

    # Create chunks
    chunks = chunker.chunk(contract_text, "contract_001")

    # Print hierarchy
    print("Document Structure:")
    for chunk in chunks:
        indent = "  " * (
            0 if chunk.level == ChunkLevel.DOCUMENT else
            1 if chunk.level == ChunkLevel.SECTION else
            2 if chunk.level == ChunkLevel.PARAGRAPH else 3
        )
        preview = chunk.text[:50].replace('\n', ' ') + "..."
        print(f"{indent}[{chunk.level.value}] {chunk.id}: {preview}")

    # Test query complexity
    analyzer = QueryComplexityAnalyzer()

    queries = [
        "What is this contract about?",
        "What is the exact monthly fee in Article 3?",
        "Explain the scope of services",
    ]

    print("\nQuery Analysis:")
    for query in queries:
        level, confidence = analyzer.analyze(query)
        print(f"  '{query}' → {level.value} (confidence: {confidence})")
