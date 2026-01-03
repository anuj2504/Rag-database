"""
Document Quality Detection and Scoring System.

Based on real enterprise RAG challenges:
- Enterprise documents range from perfect PDFs to garbage scanned copies
- Different quality levels need different processing pipelines
- Quality detection BEFORE processing prevents downstream failures

Quality Tiers:
- HIGH (0.8-1.0): Clean digital PDFs → Full hierarchical processing
- MEDIUM (0.5-0.8): Some OCR artifacts → Basic chunking with cleanup
- LOW (0.2-0.5): Poor scans → Simple fixed chunks + manual review flags
- GARBAGE (<0.2): Unprocessable → Flag for manual handling
"""
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class QualityTier(Enum):
    """Document quality tiers for routing."""
    HIGH = "high"           # Clean digital docs
    MEDIUM = "medium"       # Some issues but usable
    LOW = "low"             # Significant issues
    GARBAGE = "garbage"     # Needs manual review


@dataclass
class QualityReport:
    """Detailed quality assessment report."""
    overall_score: float  # 0.0 to 1.0
    tier: QualityTier

    # Individual scores
    text_extraction_score: float
    ocr_artifact_score: float
    formatting_score: float
    structure_score: float
    content_coherence_score: float

    # Issues found
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Recommendations
    recommended_pipeline: str = "standard"
    manual_review_required: bool = False

    # Raw metrics
    metrics: Dict[str, Any] = field(default_factory=dict)


class DocumentQualityAnalyzer:
    """
    Analyzes document quality to route to appropriate processing pipeline.

    This is the FIRST step before any chunking or embedding.
    Poor quality detection = poor retrieval results downstream.
    """

    # Common OCR error patterns
    OCR_ERROR_PATTERNS = [
        r'[Il1]{3,}',           # Confused I/l/1
        r'[O0]{3,}',            # Confused O/0
        r'[^\x00-\x7F]{5,}',    # Long non-ASCII sequences
        r'\b[a-z]+[A-Z]+[a-z]+\b',  # Mixed case in middle of words
        r'[.,]{3,}',            # Repeated punctuation (scan artifacts)
        r'\s{5,}',              # Excessive whitespace
        r'[|]{2,}',             # Table line artifacts
        r'[_]{5,}',             # Underline artifacts
        r'\b\w{20,}\b',         # Unusually long "words" (merged text)
    ]

    # Patterns indicating good structure
    STRUCTURE_PATTERNS = [
        r'^#{1,6}\s+\w+',           # Markdown headers
        r'^\d+\.\s+\w+',            # Numbered lists
        r'^[A-Z][A-Z\s]+:',         # Section headers (caps)
        r'^\s*•\s+\w+',             # Bullet points
        r'^(Section|Article|Chapter)\s+\d+', # Legal structure
        r'^Table\s+\d+',            # Table references
        r'^Figure\s+\d+',           # Figure references
    ]

    # Gibberish detection patterns
    GIBBERISH_PATTERNS = [
        r'[bcdfghjklmnpqrstvwxz]{5,}',  # Too many consonants
        r'[aeiou]{5,}',                  # Too many vowels
        r'(.)\1{4,}',                    # Repeated characters
    ]

    def __init__(
        self,
        high_threshold: float = 0.8,
        medium_threshold: float = 0.5,
        low_threshold: float = 0.2
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold

        # Compile patterns
        self.ocr_patterns = [re.compile(p) for p in self.OCR_ERROR_PATTERNS]
        self.structure_patterns = [re.compile(p, re.MULTILINE) for p in self.STRUCTURE_PATTERNS]
        self.gibberish_patterns = [re.compile(p, re.IGNORECASE) for p in self.GIBBERISH_PATTERNS]

    def analyze(
        self,
        text: str,
        filename: Optional[str] = None,
        file_metadata: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """
        Analyze document quality and return detailed report.

        Args:
            text: Extracted text from document
            filename: Original filename (for context)
            file_metadata: Additional metadata (page count, file size, etc.)

        Returns:
            QualityReport with scores and recommendations
        """
        if not text or len(text.strip()) < 50:
            return QualityReport(
                overall_score=0.0,
                tier=QualityTier.GARBAGE,
                text_extraction_score=0.0,
                ocr_artifact_score=0.0,
                formatting_score=0.0,
                structure_score=0.0,
                content_coherence_score=0.0,
                issues=["Text extraction failed or document is nearly empty"],
                manual_review_required=True,
                recommended_pipeline="manual"
            )

        # Calculate individual scores
        text_score = self._score_text_extraction(text)
        ocr_score = self._score_ocr_quality(text)
        format_score = self._score_formatting(text)
        structure_score = self._score_structure(text)
        coherence_score = self._score_coherence(text)

        # Weighted overall score
        # OCR and text extraction are most important
        weights = {
            'text': 0.25,
            'ocr': 0.30,
            'format': 0.15,
            'structure': 0.15,
            'coherence': 0.15
        }

        overall_score = (
            weights['text'] * text_score +
            weights['ocr'] * ocr_score +
            weights['format'] * format_score +
            weights['structure'] * structure_score +
            weights['coherence'] * coherence_score
        )

        # Determine tier
        tier = self._determine_tier(overall_score)

        # Collect issues and warnings
        issues, warnings = self._collect_issues(
            text, text_score, ocr_score, format_score,
            structure_score, coherence_score
        )

        # Determine recommended pipeline
        pipeline, manual_review = self._recommend_pipeline(tier, issues)

        # Collect metrics
        metrics = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'line_count': text.count('\n') + 1,
            'avg_word_length': self._avg_word_length(text),
            'unique_word_ratio': self._unique_word_ratio(text),
            'uppercase_ratio': self._uppercase_ratio(text),
            'digit_ratio': self._digit_ratio(text),
            'special_char_ratio': self._special_char_ratio(text),
        }

        return QualityReport(
            overall_score=round(overall_score, 3),
            tier=tier,
            text_extraction_score=round(text_score, 3),
            ocr_artifact_score=round(ocr_score, 3),
            formatting_score=round(format_score, 3),
            structure_score=round(structure_score, 3),
            content_coherence_score=round(coherence_score, 3),
            issues=issues,
            warnings=warnings,
            recommended_pipeline=pipeline,
            manual_review_required=manual_review,
            metrics=metrics
        )

    def _score_text_extraction(self, text: str) -> float:
        """Score based on basic text extraction quality."""
        score = 1.0

        # Check for empty or very short text
        word_count = len(text.split())
        if word_count < 10:
            return 0.1
        if word_count < 50:
            score -= 0.3

        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.3:  # Less than 30% letters is suspicious
            score -= 0.3
        elif alpha_ratio < 0.5:
            score -= 0.1

        # Check for null bytes or control characters
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        if control_chars > 0:
            score -= min(0.3, control_chars / len(text) * 10)

        return max(0.0, score)

    def _score_ocr_quality(self, text: str) -> float:
        """Score based on OCR artifact detection."""
        score = 1.0
        text_length = len(text)

        # Count OCR error patterns
        total_errors = 0
        for pattern in self.ocr_patterns:
            matches = pattern.findall(text)
            total_errors += len(matches)

        # Calculate error density
        error_density = total_errors / (text_length / 1000)  # errors per 1000 chars

        if error_density > 10:
            score -= 0.5
        elif error_density > 5:
            score -= 0.3
        elif error_density > 2:
            score -= 0.1

        # Check for gibberish sequences
        gibberish_count = 0
        for pattern in self.gibberish_patterns:
            gibberish_count += len(pattern.findall(text))

        gibberish_density = gibberish_count / (text_length / 1000)
        if gibberish_density > 5:
            score -= 0.3
        elif gibberish_density > 2:
            score -= 0.15

        return max(0.0, score)

    def _score_formatting(self, text: str) -> float:
        """Score based on text formatting consistency."""
        score = 1.0
        lines = text.split('\n')

        if not lines:
            return 0.5

        # Check line length consistency
        line_lengths = [len(line) for line in lines if line.strip()]
        if line_lengths:
            avg_length = sum(line_lengths) / len(line_lengths)
            variance = sum((l - avg_length) ** 2 for l in line_lengths) / len(line_lengths)

            # Very high variance suggests formatting issues
            if variance > 10000:
                score -= 0.2

        # Check for excessive blank lines
        blank_ratio = sum(1 for line in lines if not line.strip()) / len(lines)
        if blank_ratio > 0.5:
            score -= 0.2

        # Check for consistent indentation patterns
        indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        if indents:
            unique_indents = len(set(indents))
            if unique_indents > 20:  # Too many indent levels suggests noise
                score -= 0.1

        return max(0.0, score)

    def _score_structure(self, text: str) -> float:
        """Score based on document structure detection."""
        score = 0.5  # Start at neutral

        # Look for structure indicators
        structure_count = 0
        for pattern in self.structure_patterns:
            matches = pattern.findall(text)
            structure_count += len(matches)

        # More structure = higher score
        if structure_count > 20:
            score = 1.0
        elif structure_count > 10:
            score = 0.85
        elif structure_count > 5:
            score = 0.7
        elif structure_count > 0:
            score = 0.6

        return score

    def _score_coherence(self, text: str) -> float:
        """Score based on content coherence (real language vs noise)."""
        score = 1.0
        words = text.lower().split()

        if len(words) < 20:
            return 0.5

        # Check word length distribution
        word_lengths = [len(w) for w in words]
        avg_word_length = sum(word_lengths) / len(word_lengths)

        # English avg word length is ~4.5. Too high or low is suspicious
        if avg_word_length < 2 or avg_word_length > 12:
            score -= 0.3
        elif avg_word_length < 3 or avg_word_length > 8:
            score -= 0.1

        # Check for reasonable vocabulary diversity
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.1:  # Too repetitive
            score -= 0.3
        elif unique_ratio > 0.95:  # No repetition at all is suspicious for long text
            if len(words) > 500:
                score -= 0.1

        # Simple language detection - check for common English words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'to', 'and', 'in', 'for', 'on', 'with'}
        common_count = sum(1 for w in words if w in common_words)
        common_ratio = common_count / len(words)

        if common_ratio < 0.01:  # Suspiciously low common words
            score -= 0.2

        return max(0.0, score)

    def _determine_tier(self, score: float) -> QualityTier:
        """Determine quality tier from overall score."""
        if score >= self.high_threshold:
            return QualityTier.HIGH
        elif score >= self.medium_threshold:
            return QualityTier.MEDIUM
        elif score >= self.low_threshold:
            return QualityTier.LOW
        else:
            return QualityTier.GARBAGE

    def _collect_issues(
        self,
        text: str,
        text_score: float,
        ocr_score: float,
        format_score: float,
        structure_score: float,
        coherence_score: float
    ) -> Tuple[List[str], List[str]]:
        """Collect specific issues and warnings."""
        issues = []
        warnings = []

        if text_score < 0.5:
            issues.append("Poor text extraction quality")
        elif text_score < 0.7:
            warnings.append("Text extraction has some issues")

        if ocr_score < 0.5:
            issues.append("Significant OCR artifacts detected")
        elif ocr_score < 0.7:
            warnings.append("Some OCR artifacts present")

        if format_score < 0.5:
            issues.append("Document formatting is inconsistent")
        elif format_score < 0.7:
            warnings.append("Minor formatting inconsistencies")

        if structure_score < 0.4:
            warnings.append("Limited document structure detected")

        if coherence_score < 0.5:
            issues.append("Text coherence is poor - possible garbage content")
        elif coherence_score < 0.7:
            warnings.append("Some coherence issues detected")

        return issues, warnings

    def _recommend_pipeline(
        self,
        tier: QualityTier,
        issues: List[str]
    ) -> Tuple[str, bool]:
        """Recommend processing pipeline based on quality."""
        if tier == QualityTier.HIGH:
            return "hierarchical", False
        elif tier == QualityTier.MEDIUM:
            return "standard", False
        elif tier == QualityTier.LOW:
            return "simple", True
        else:  # GARBAGE
            return "manual", True

    # Helper methods for metrics
    def _avg_word_length(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0
        return sum(len(w) for w in words) / len(words)

    def _unique_word_ratio(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0
        return len(set(words)) / len(words)

    def _uppercase_ratio(self, text: str) -> float:
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return 0
        return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

    def _digit_ratio(self, text: str) -> float:
        if not text:
            return 0
        return sum(1 for c in text if c.isdigit()) / len(text)

    def _special_char_ratio(self, text: str) -> float:
        if not text:
            return 0
        return sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)


class QualityBasedRouter:
    """
    Routes documents to appropriate processing pipelines based on quality.

    This prevents the common mistake of applying the same processing
    to all documents regardless of their actual quality.
    """

    def __init__(self, analyzer: Optional[DocumentQualityAnalyzer] = None):
        self.analyzer = analyzer or DocumentQualityAnalyzer()

    def route(self, text: str, filename: str = None) -> Dict[str, Any]:
        """
        Analyze and route document to appropriate pipeline.

        Returns:
            Dict with 'report', 'pipeline', and 'config' for processing
        """
        report = self.analyzer.analyze(text, filename)

        # Get pipeline-specific configuration
        config = self._get_pipeline_config(report.tier, report)

        return {
            "report": report,
            "pipeline": report.recommended_pipeline,
            "config": config
        }

    def _get_pipeline_config(
        self,
        tier: QualityTier,
        report: QualityReport
    ) -> Dict[str, Any]:
        """Get processing configuration for quality tier."""

        if tier == QualityTier.HIGH:
            return {
                "chunking_strategy": "hierarchical",
                "chunk_sizes": {
                    "document": None,  # Full document
                    "section": 2000,
                    "paragraph": 400,
                    "sentence": 100
                },
                "enable_table_extraction": True,
                "enable_metadata_extraction": True,
                "enable_structure_detection": True,
                "cleanup_level": "minimal",
            }

        elif tier == QualityTier.MEDIUM:
            return {
                "chunking_strategy": "standard",
                "chunk_sizes": {
                    "paragraph": 500,
                },
                "enable_table_extraction": True,
                "enable_metadata_extraction": True,
                "enable_structure_detection": False,
                "cleanup_level": "moderate",
                "ocr_cleanup": True,
            }

        elif tier == QualityTier.LOW:
            return {
                "chunking_strategy": "simple",
                "chunk_sizes": {
                    "fixed": 400,
                },
                "chunk_overlap": 50,
                "enable_table_extraction": False,
                "enable_metadata_extraction": False,
                "enable_structure_detection": False,
                "cleanup_level": "aggressive",
                "ocr_cleanup": True,
                "flag_for_review": True,
            }

        else:  # GARBAGE
            return {
                "chunking_strategy": "minimal",
                "chunk_sizes": {
                    "fixed": 300,
                },
                "enable_table_extraction": False,
                "enable_metadata_extraction": False,
                "cleanup_level": "aggressive",
                "flag_for_review": True,
                "skip_embedding": True,  # Don't waste compute on garbage
            }


# Example usage
if __name__ == "__main__":
    analyzer = DocumentQualityAnalyzer()

    # Test with sample text
    good_text = """
    # Contract Agreement

    This Agreement is entered into as of January 1, 2024.

    ## Section 1: Definitions

    1.1 "Company" means ABC Corporation.
    1.2 "Services" means the consulting services described in Exhibit A.

    ## Section 2: Payment Terms

    The Client shall pay the Company according to the following schedule:
    - Initial payment: $10,000 upon signing
    - Monthly retainer: $5,000 per month
    """

    bad_text = """
    Th1s ls s0me p00rly sc4nned t3xt w1th l0ts 0f 0CR err0rs
    and  m1ss1ng   characters    everywheeeeere
    aaaabbbbcccc ||||||||||| ________
    xyzqwrtp mjnbvcx asdfghjkl
    """

    print("Good document:")
    report = analyzer.analyze(good_text)
    print(f"  Score: {report.overall_score}, Tier: {report.tier.value}")
    print(f"  Pipeline: {report.recommended_pipeline}")

    print("\nBad document:")
    report = analyzer.analyze(bad_text)
    print(f"  Score: {report.overall_score}, Tier: {report.tier.value}")
    print(f"  Pipeline: {report.recommended_pipeline}")
    print(f"  Issues: {report.issues}")
