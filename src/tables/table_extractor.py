"""
Table Extraction Pipeline.

Key insight from enterprise RAG:
- Tables contain CRITICAL information (financial data, compliance matrices)
- Standard RAG ignores tables or extracts as unstructured text
- Tables need DEDICATED processing to preserve relationships

This module provides:
- Table detection in documents
- Structure-preserving extraction
- Dual embedding (structured + semantic description)
- Table-specific querying
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TableType(Enum):
    """Types of tables commonly found in enterprise documents."""
    FINANCIAL = "financial"        # Numbers, currency, percentages
    COMPLIANCE = "compliance"      # Checkmarks, yes/no, status
    SCHEDULE = "schedule"          # Dates, deadlines, milestones
    REFERENCE = "reference"        # Cross-references, codes
    DATA = "data"                  # General data tables
    COMPARISON = "comparison"      # Side-by-side comparisons


@dataclass
class TableCell:
    """Individual table cell."""
    value: str
    row: int
    col: int
    is_header: bool = False
    data_type: str = "text"  # text, number, currency, percentage, date


@dataclass
class ExtractedTable:
    """Extracted and processed table."""
    id: str
    document_id: str
    page_number: Optional[int]

    # Table structure
    rows: List[List[TableCell]]
    num_rows: int
    num_cols: int
    headers: List[str]

    # Table metadata
    table_type: TableType
    title: Optional[str] = None
    caption: Optional[str] = None

    # Position in document
    char_start: int = 0
    char_end: int = 0

    # For embedding
    structured_text: str = ""      # CSV-like structured representation
    semantic_description: str = "" # Natural language description

    # Raw data for querying
    raw_data: Dict[str, Any] = field(default_factory=dict)


class TableDetector:
    """
    Detects tables in document text.

    Uses heuristics since LLM detection is expensive and inconsistent.
    """

    # Patterns indicating table presence
    TABLE_INDICATORS = [
        r'\|\s*\w+\s*\|',           # Markdown-style tables
        r'\t{2,}',                   # Multiple tabs (common in text tables)
        r'\s{4,}\S+\s{4,}',         # Spaced columns
        r'Table\s+\d+',             # Table references
        r'^\s*[-|+]+\s*$',          # Table borders
    ]

    # Column delimiter patterns
    DELIMITER_PATTERNS = [
        r'\|',                       # Pipe
        r'\t',                       # Tab
        r'\s{3,}',                   # Multiple spaces
    ]

    def __init__(self):
        self.table_patterns = [re.compile(p, re.MULTILINE) for p in self.TABLE_INDICATORS]
        self.delimiter_patterns = [re.compile(p) for p in self.DELIMITER_PATTERNS]

    def detect_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect potential tables in text.

        Returns:
            List of dicts with 'start', 'end', 'text', 'confidence'
        """
        tables = []

        # Method 1: Look for table indicators
        for pattern in self.table_patterns:
            for match in pattern.finditer(text):
                # Expand to find table boundaries
                table_region = self._expand_table_region(text, match.start(), match.end())
                if table_region:
                    tables.append(table_region)

        # Method 2: Look for consistent column patterns
        line_tables = self._detect_by_line_analysis(text)
        tables.extend(line_tables)

        # Deduplicate overlapping detections
        tables = self._deduplicate_tables(tables)

        return tables

    def _expand_table_region(
        self,
        text: str,
        start: int,
        end: int
    ) -> Optional[Dict[str, Any]]:
        """Expand from match to full table boundaries."""
        lines = text.split('\n')

        # Find which line contains the match
        char_count = 0
        start_line = 0
        for i, line in enumerate(lines):
            if char_count <= start < char_count + len(line) + 1:
                start_line = i
                break
            char_count += len(line) + 1

        # Expand upward to find table start
        table_start_line = start_line
        for i in range(start_line - 1, max(0, start_line - 20), -1):
            if self._is_table_line(lines[i]):
                table_start_line = i
            elif lines[i].strip() and not self._is_table_line(lines[i]):
                break

        # Expand downward to find table end
        table_end_line = start_line
        for i in range(start_line + 1, min(len(lines), start_line + 50)):
            if self._is_table_line(lines[i]):
                table_end_line = i
            elif lines[i].strip() and not self._is_table_line(lines[i]):
                break

        # Extract table text
        if table_end_line - table_start_line >= 2:  # At least 3 rows
            table_lines = lines[table_start_line:table_end_line + 1]
            table_text = '\n'.join(table_lines)

            # Calculate character positions
            table_start_char = sum(len(lines[i]) + 1 for i in range(table_start_line))
            table_end_char = sum(len(lines[i]) + 1 for i in range(table_end_line + 1))

            return {
                'start': table_start_char,
                'end': table_end_char,
                'text': table_text,
                'start_line': table_start_line,
                'end_line': table_end_line,
                'confidence': self._calculate_table_confidence(table_text),
            }

        return None

    def _is_table_line(self, line: str) -> bool:
        """Check if a line looks like part of a table."""
        if not line.strip():
            return False

        # Check for delimiters
        if '|' in line:
            return True
        if '\t' in line and line.count('\t') >= 2:
            return True

        # Check for consistent spacing
        parts = re.split(r'\s{3,}', line.strip())
        if len(parts) >= 3:
            return True

        # Check for border characters
        if re.match(r'^[\s\-|+]+$', line):
            return True

        return False

    def _detect_by_line_analysis(self, text: str) -> List[Dict[str, Any]]:
        """Detect tables by analyzing line structure."""
        tables = []
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            # Look for sequence of lines with consistent column count
            if self._is_table_line(lines[i]):
                start_line = i
                col_counts = []

                while i < len(lines) and (self._is_table_line(lines[i]) or not lines[i].strip()):
                    if lines[i].strip():
                        col_count = self._count_columns(lines[i])
                        col_counts.append(col_count)
                    i += 1

                # If we have consistent columns for multiple lines, it's a table
                if len(col_counts) >= 3:
                    avg_cols = sum(col_counts) / len(col_counts)
                    variance = sum((c - avg_cols) ** 2 for c in col_counts) / len(col_counts)

                    if variance < 2:  # Low variance = consistent table
                        table_text = '\n'.join(lines[start_line:i])
                        table_start = sum(len(lines[j]) + 1 for j in range(start_line))

                        tables.append({
                            'start': table_start,
                            'end': table_start + len(table_text),
                            'text': table_text,
                            'start_line': start_line,
                            'end_line': i - 1,
                            'confidence': min(0.9, 0.5 + len(col_counts) * 0.05),
                        })
            else:
                i += 1

        return tables

    def _count_columns(self, line: str) -> int:
        """Count number of columns in a line."""
        if '|' in line:
            return len([p for p in line.split('|') if p.strip()])
        elif '\t' in line:
            return len([p for p in line.split('\t') if p.strip()])
        else:
            return len([p for p in re.split(r'\s{2,}', line.strip()) if p])

    def _calculate_table_confidence(self, text: str) -> float:
        """Calculate confidence that text is actually a table."""
        score = 0.5

        # More rows = higher confidence
        row_count = len(text.strip().split('\n'))
        score += min(0.2, row_count * 0.02)

        # Consistent delimiters = higher confidence
        if '|' in text and text.count('|') > row_count:
            score += 0.15

        # Numbers and data-like content
        if re.search(r'\$[\d,]+|\d+%|\d{1,2}/\d{1,2}/\d{2,4}', text):
            score += 0.1

        return min(1.0, score)

    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove overlapping table detections."""
        if not tables:
            return []

        # Sort by start position
        sorted_tables = sorted(tables, key=lambda x: x['start'])

        result = [sorted_tables[0]]
        for table in sorted_tables[1:]:
            last = result[-1]
            # If not overlapping, add it
            if table['start'] >= last['end']:
                result.append(table)
            # If overlapping but higher confidence, replace
            elif table['confidence'] > last['confidence']:
                result[-1] = table

        return result


class TableParser:
    """
    Parses detected table text into structured format.
    """

    def parse(
        self,
        table_text: str,
        document_id: str,
        table_index: int,
        page_number: int = None
    ) -> ExtractedTable:
        """
        Parse table text into structured ExtractedTable.

        Args:
            table_text: Raw table text
            document_id: Parent document ID
            table_index: Table index in document
            page_number: Optional page number

        Returns:
            ExtractedTable with structured data
        """
        # Detect delimiter
        delimiter = self._detect_delimiter(table_text)

        # Parse into rows and cells
        rows = self._parse_rows(table_text, delimiter)

        # Identify headers
        headers, data_rows = self._identify_headers(rows)

        # Detect data types
        rows_with_types = self._detect_data_types(rows)

        # Determine table type
        table_type = self._classify_table(rows_with_types, headers)

        # Extract title/caption
        title = self._extract_title(table_text)

        # Create structured representations
        structured_text = self._to_structured_text(rows, headers)
        semantic_desc = self._to_semantic_description(rows, headers, table_type)

        # Create raw data dict for querying
        raw_data = self._to_dict(rows, headers)

        return ExtractedTable(
            id=f"{document_id}_table_{table_index}",
            document_id=document_id,
            page_number=page_number,
            rows=rows_with_types,
            num_rows=len(rows),
            num_cols=max(len(r) for r in rows) if rows else 0,
            headers=headers,
            table_type=table_type,
            title=title,
            structured_text=structured_text,
            semantic_description=semantic_desc,
            raw_data=raw_data
        )

    def _detect_delimiter(self, text: str) -> str:
        """Detect the most likely column delimiter."""
        pipe_count = text.count('|')
        tab_count = text.count('\t')
        lines = text.strip().split('\n')

        if pipe_count > len(lines):
            return '|'
        elif tab_count > len(lines):
            return '\t'
        else:
            return r'\s{2,}'  # Regex for multiple spaces

    def _parse_rows(
        self,
        text: str,
        delimiter: str
    ) -> List[List[TableCell]]:
        """Parse text into rows of cells."""
        rows = []
        lines = text.strip().split('\n')

        for row_idx, line in enumerate(lines):
            # Skip border lines
            if re.match(r'^[\s\-|+]+$', line):
                continue

            if delimiter in ['|', '\t']:
                parts = [p.strip() for p in line.split(delimiter) if p.strip()]
            else:
                parts = [p.strip() for p in re.split(delimiter, line) if p.strip()]

            if parts:
                cells = [
                    TableCell(
                        value=part,
                        row=row_idx,
                        col=col_idx,
                        is_header=(row_idx == 0)
                    )
                    for col_idx, part in enumerate(parts)
                ]
                rows.append(cells)

        return rows

    def _identify_headers(
        self,
        rows: List[List[TableCell]]
    ) -> Tuple[List[str], List[List[TableCell]]]:
        """Identify header row."""
        if not rows:
            return [], []

        # First row is usually header
        headers = [cell.value for cell in rows[0]]

        # Mark first row as headers
        for cell in rows[0]:
            cell.is_header = True

        return headers, rows[1:]

    def _detect_data_types(
        self,
        rows: List[List[TableCell]]
    ) -> List[List[TableCell]]:
        """Detect data type of each cell."""
        for row in rows:
            for cell in row:
                cell.data_type = self._classify_cell(cell.value)
        return rows

    def _classify_cell(self, value: str) -> str:
        """Classify a cell's data type."""
        value = value.strip()

        if re.match(r'^\$[\d,]+(?:\.\d{2})?$', value):
            return 'currency'
        elif re.match(r'^[\d,]+(?:\.\d+)?%$', value):
            return 'percentage'
        elif re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', value):
            return 'date'
        elif re.match(r'^[\d,]+(?:\.\d+)?$', value):
            return 'number'
        elif value.lower() in ['yes', 'no', 'true', 'false', '✓', '✗', 'x']:
            return 'boolean'
        else:
            return 'text'

    def _classify_table(
        self,
        rows: List[List[TableCell]],
        headers: List[str]
    ) -> TableType:
        """Classify table type based on content."""
        all_cells = [cell for row in rows for cell in row]

        # Count data types
        type_counts = {}
        for cell in all_cells:
            type_counts[cell.data_type] = type_counts.get(cell.data_type, 0) + 1

        total_cells = len(all_cells)
        if total_cells == 0:
            return TableType.DATA

        # Financial: mostly currency/numbers
        if (type_counts.get('currency', 0) + type_counts.get('number', 0)) / total_cells > 0.5:
            return TableType.FINANCIAL

        # Compliance: has boolean values
        if type_counts.get('boolean', 0) / total_cells > 0.2:
            return TableType.COMPLIANCE

        # Schedule: has dates
        if type_counts.get('date', 0) / total_cells > 0.2:
            return TableType.SCHEDULE

        # Check headers for hints
        headers_lower = [h.lower() for h in headers]
        if any(h in headers_lower for h in ['amount', 'revenue', 'cost', 'price', 'total']):
            return TableType.FINANCIAL
        if any(h in headers_lower for h in ['date', 'deadline', 'due', 'schedule']):
            return TableType.SCHEDULE
        if any(h in headers_lower for h in ['status', 'complete', 'approved']):
            return TableType.COMPLIANCE

        return TableType.DATA

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract table title if present."""
        # Look for "Table X:" or "Table X." patterns before table
        match = re.search(r'(Table\s+\d+[:.]\s*[^\n]+)', text[:200], re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _to_structured_text(
        self,
        rows: List[List[TableCell]],
        headers: List[str]
    ) -> str:
        """Convert to CSV-like structured text."""
        lines = []

        if headers:
            lines.append(','.join(f'"{h}"' for h in headers))

        for row in rows:
            values = [f'"{cell.value}"' for cell in row]
            lines.append(','.join(values))

        return '\n'.join(lines)

    def _to_semantic_description(
        self,
        rows: List[List[TableCell]],
        headers: List[str],
        table_type: TableType
    ) -> str:
        """
        Create natural language description for semantic embedding.

        This allows the table to be found via semantic search.
        """
        descriptions = []

        # Table type description
        type_desc = {
            TableType.FINANCIAL: "financial data table with monetary values",
            TableType.COMPLIANCE: "compliance or status tracking table",
            TableType.SCHEDULE: "schedule or timeline table with dates",
            TableType.REFERENCE: "reference table with codes or identifiers",
            TableType.COMPARISON: "comparison table",
            TableType.DATA: "data table",
        }
        descriptions.append(f"This is a {type_desc.get(table_type, 'data table')}.")

        # Column descriptions
        if headers:
            descriptions.append(f"Columns: {', '.join(headers)}.")

        # Row count
        data_rows = [r for r in rows if not (r and r[0].is_header)]
        descriptions.append(f"Contains {len(data_rows)} data rows.")

        # Sample data description
        if data_rows and headers:
            sample_row = data_rows[0]
            sample_desc = []
            for i, cell in enumerate(sample_row):
                if i < len(headers):
                    sample_desc.append(f"{headers[i]}: {cell.value}")
            if sample_desc:
                descriptions.append(f"Sample row: {', '.join(sample_desc[:4])}.")

        return ' '.join(descriptions)

    def _to_dict(
        self,
        rows: List[List[TableCell]],
        headers: List[str]
    ) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage and querying."""
        data = {
            'headers': headers,
            'rows': []
        }

        for row in rows:
            if row and not row[0].is_header:
                row_dict = {}
                for i, cell in enumerate(row):
                    key = headers[i] if i < len(headers) else f'col_{i}'
                    row_dict[key] = {
                        'value': cell.value,
                        'type': cell.data_type
                    }
                data['rows'].append(row_dict)

        return data


class TableExtractor:
    """
    Main class for table extraction pipeline.

    Combines detection and parsing.
    """

    def __init__(self):
        self.detector = TableDetector()
        self.parser = TableParser()

    def extract_tables(
        self,
        text: str,
        document_id: str,
        min_confidence: float = 0.5
    ) -> List[ExtractedTable]:
        """
        Extract all tables from document.

        Args:
            text: Document text
            document_id: Parent document ID
            min_confidence: Minimum detection confidence

        Returns:
            List of ExtractedTable
        """
        # Detect tables
        detected = self.detector.detect_tables(text)

        # Filter by confidence
        detected = [t for t in detected if t['confidence'] >= min_confidence]

        # Parse each table
        tables = []
        for i, table_info in enumerate(detected):
            try:
                table = self.parser.parse(
                    table_text=table_info['text'],
                    document_id=document_id,
                    table_index=i
                )
                table.char_start = table_info['start']
                table.char_end = table_info['end']
                tables.append(table)

            except Exception as e:
                logger.warning(f"Failed to parse table {i}: {e}")

        return tables

    def get_table_chunks(
        self,
        tables: List[ExtractedTable]
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from tables for embedding.

        Each table produces TWO chunks:
        1. Structured text (for precise queries)
        2. Semantic description (for conceptual queries)
        """
        chunks = []

        for table in tables:
            # Structured chunk
            chunks.append({
                'id': f"{table.id}_structured",
                'text': table.structured_text,
                'type': 'table_structured',
                'table_id': table.id,
                'document_id': table.document_id,
                'metadata': {
                    'table_type': table.table_type.value,
                    'headers': table.headers,
                    'num_rows': table.num_rows,
                    'title': table.title,
                }
            })

            # Semantic chunk
            chunks.append({
                'id': f"{table.id}_semantic",
                'text': table.semantic_description,
                'type': 'table_semantic',
                'table_id': table.id,
                'document_id': table.document_id,
                'metadata': {
                    'table_type': table.table_type.value,
                    'headers': table.headers,
                }
            })

        return chunks


# Example usage
if __name__ == "__main__":
    # Sample table text
    sample_text = """
    Financial Summary for Q4 2023

    Table 1: Revenue by Segment

    | Segment          | Q4 2023    | Q4 2022    | Change   |
    |------------------|------------|------------|----------|
    | Software         | $2,500,000 | $2,100,000 | 19%      |
    | Services         | $1,800,000 | $1,650,000 | 9%       |
    | Hardware         | $800,000   | $950,000   | -16%     |
    | Total            | $5,100,000 | $4,700,000 | 8.5%     |

    The results show strong growth in software segment.
    """

    extractor = TableExtractor()
    tables = extractor.extract_tables(sample_text, "doc_001")

    for table in tables:
        print(f"\nTable: {table.id}")
        print(f"Type: {table.table_type.value}")
        print(f"Headers: {table.headers}")
        print(f"Rows: {table.num_rows}")
        print(f"\nStructured:\n{table.structured_text}")
        print(f"\nSemantic: {table.semantic_description}")
