"""
Enhanced Hybrid Search with Enterprise Features.

Key improvements over basic hybrid search:
1. Query complexity detection → adaptive retrieval level
2. Precision query triggers → rule-based fallbacks
3. Acronym expansion → domain-aware query enhancement
4. Graph augmentation → related document discovery
5. Failure detection → automatic fallback strategies
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from collections import defaultdict

from src.retrieval.hybrid_search import (
    HybridSearcher,
    HybridResult,
    RetrievalMethod,
    ReciprocalRankFusion
)
from src.chunking.hierarchical_chunker import ChunkLevel, QueryComplexityAnalyzer
from src.terminology.acronym_database import AcronymDatabase, QueryEnhancer, Domain
from src.graph.document_graph import DocumentGraph, GraphAugmentedRetrieval, RelationType

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for routing."""
    BROAD = "broad"           # General overview questions
    PRECISE = "precise"       # Specific data points
    REFERENCE = "reference"   # Looking for specific section/table
    COMPARATIVE = "comparative"  # Comparing entities
    TEMPORAL = "temporal"     # Time-based queries


@dataclass
class EnhancedSearchResult(HybridResult):
    """Extended result with enterprise features."""
    # Augmentation info
    is_graph_augmented: bool = False
    augmentation_source: Optional[str] = None
    relation_path: List[Dict] = field(default_factory=list)

    # Query analysis
    query_type: Optional[str] = None
    retrieval_level: Optional[str] = None
    expansions_applied: List[str] = field(default_factory=list)

    # Confidence signals
    retrieval_confidence: float = 1.0
    potential_issues: List[str] = field(default_factory=list)


@dataclass
class SearchAnalytics:
    """Analytics for search performance monitoring."""
    query: str
    query_type: QueryType
    detected_domain: str
    retrieval_methods_used: List[str]
    total_results: int
    graph_augmented_count: int
    fallback_triggered: bool
    processing_time_ms: float
    expansions: List[str]


class PrecisionQueryDetector:
    """
    Detects precision queries that need special handling.

    Precision queries often fail with pure semantic search.
    They need rule-based or exact match fallbacks.
    """

    # Patterns indicating precision queries
    PRECISION_PATTERNS = [
        # Table/figure references
        (r'(?:in\s+)?table\s+(\d+|[A-Z])', 'table_reference'),
        (r'(?:in\s+)?figure\s+(\d+|[A-Z])', 'figure_reference'),
        (r'(?:in\s+)?exhibit\s+([A-Z]|\d+)', 'exhibit_reference'),

        # Section references
        (r'(?:in\s+)?section\s+(\d+(?:\.\d+)*)', 'section_reference'),
        (r'(?:in\s+)?article\s+(\d+|[IVXLC]+)', 'article_reference'),
        (r'(?:in\s+)?clause\s+(\d+(?:\.\d+)*)', 'clause_reference'),

        # Specific value queries
        (r'(?:exact|specific|precise)\s+(?:amount|value|number|date)', 'exact_value'),
        (r'what\s+(?:is|was|are|were)\s+the\s+(?:exact|specific)', 'exact_value'),

        # Line/paragraph references
        (r'(?:on\s+)?line\s+(\d+)', 'line_reference'),
        (r'(?:in\s+)?paragraph\s+(\d+)', 'paragraph_reference'),
    ]

    def __init__(self):
        self.patterns = [
            (re.compile(p, re.IGNORECASE), ptype)
            for p, ptype in self.PRECISION_PATTERNS
        ]

    def detect(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect if query is a precision query.

        Returns:
            Tuple of (is_precision, pattern_type, extracted_reference)
        """
        for pattern, ptype in self.patterns:
            match = pattern.search(query)
            if match:
                reference = match.group(1) if match.groups() else None
                return True, ptype, reference

        return False, None, None


class FailureDetector:
    """
    Detects potential retrieval failures.

    Helps identify when results might be unreliable
    and fallback strategies should be considered.
    """

    def analyze_results(
        self,
        query: str,
        results: List[HybridResult],
        expected_min_score: float = 0.3
    ) -> Dict[str, Any]:
        """
        Analyze results for potential failures.

        Returns:
            Dict with failure signals and recommendations
        """
        signals = {
            'low_scores': False,
            'high_variance': False,
            'single_source': False,
            'missing_methods': [],
            'confidence': 1.0,
            'recommendations': []
        }

        if not results:
            signals['confidence'] = 0.0
            signals['recommendations'].append('no_results_fallback')
            return signals

        # Check for low scores
        avg_score = sum(r.final_score for r in results) / len(results)
        if avg_score < expected_min_score:
            signals['low_scores'] = True
            signals['confidence'] -= 0.2
            signals['recommendations'].append('try_keyword_search')

        # Check score variance (might indicate uncertainty)
        if len(results) > 1:
            scores = [r.final_score for r in results]
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            if variance > 0.1:
                signals['high_variance'] = True
                signals['confidence'] -= 0.1

        # Check if results come from single document
        doc_ids = set(r.metadata.get('document_id') for r in results)
        if len(doc_ids) == 1:
            signals['single_source'] = True
            signals['recommendations'].append('expand_search')

        # Check which methods contributed
        methods_with_results = set()
        for r in results:
            if r.bm25_rank:
                methods_with_results.add('bm25')
            if r.dense_rank:
                methods_with_results.add('dense')
            if r.colpali_rank:
                methods_with_results.add('colpali')

        expected_methods = {'bm25', 'dense'}
        missing = expected_methods - methods_with_results
        if missing:
            signals['missing_methods'] = list(missing)
            signals['confidence'] -= 0.1 * len(missing)

        signals['confidence'] = max(0.0, signals['confidence'])

        return signals


class EnhancedHybridSearcher:
    """
    Enterprise-grade hybrid search with all enhancements.

    Features:
    - Query analysis and routing
    - Acronym expansion
    - Precision query handling
    - Graph augmentation
    - Failure detection and fallbacks
    """

    def __init__(
        self,
        base_searcher: HybridSearcher,
        acronym_db: AcronymDatabase = None,
        document_graph: DocumentGraph = None,
        enable_graph_augmentation: bool = True,
        enable_precision_fallback: bool = True
    ):
        self.base_searcher = base_searcher
        self.acronym_db = acronym_db or AcronymDatabase()
        self.query_enhancer = QueryEnhancer(self.acronym_db)
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.precision_detector = PrecisionQueryDetector()
        self.failure_detector = FailureDetector()

        self.document_graph = document_graph
        self.graph_retrieval = (
            GraphAugmentedRetrieval(document_graph)
            if document_graph else None
        )

        self.enable_graph_augmentation = enable_graph_augmentation
        self.enable_precision_fallback = enable_precision_fallback

    def search(
        self,
        query: str,
        limit: int = 10,
        domain: Domain = None,
        filters: Optional[Dict[str, Any]] = None,
        expand_acronyms: bool = True,
        use_graph: bool = True
    ) -> Tuple[List[EnhancedSearchResult], SearchAnalytics]:
        """
        Perform enhanced hybrid search.

        Args:
            query: Search query
            limit: Number of results
            domain: Domain hint for acronym expansion
            filters: Metadata filters
            expand_acronyms: Whether to expand acronyms
            use_graph: Whether to use graph augmentation

        Returns:
            Tuple of (results, analytics)
        """
        import time
        start_time = time.time()

        # Step 1: Analyze query
        query_type = self._classify_query(query)
        retrieval_level, _ = self.complexity_analyzer.analyze(query)

        # Step 2: Detect domain and expand acronyms
        enhanced = self.query_enhancer.enhance(query, domain, expand_acronyms)
        search_query = enhanced['enhanced_query'] if expand_acronyms else query
        detected_domain = Domain(enhanced['detected_domain'])

        # Step 3: Check for precision query
        is_precision, precision_type, reference = self.precision_detector.detect(query)

        # Step 4: Perform base hybrid search
        base_results = self.base_searcher.search(
            query=search_query,
            limit=limit * 2,  # Get more for reranking
            filters=filters
        )

        # Step 5: Handle precision queries with fallback
        if is_precision and self.enable_precision_fallback:
            precision_results = self._precision_fallback(
                query, precision_type, reference, filters
            )
            # Merge precision results at top
            base_results = self._merge_results(precision_results, base_results)

        # Step 6: Analyze for failures
        failure_signals = self.failure_detector.analyze_results(query, base_results)

        # Step 7: Apply fallback strategies if needed
        fallback_triggered = False
        if failure_signals['confidence'] < 0.5:
            fallback_results = self._apply_fallbacks(
                query, filters, failure_signals['recommendations']
            )
            if fallback_results:
                base_results = self._merge_results(fallback_results, base_results)
                fallback_triggered = True

        # Step 8: Graph augmentation
        graph_augmented_count = 0
        if use_graph and self.graph_retrieval and self.enable_graph_augmentation:
            result_dicts = [
                {'id': r.id, 'document_id': r.metadata.get('document_id')}
                for r in base_results
            ]
            augmented = self.graph_retrieval.augment_results(result_dicts)
            graph_augmented_count = sum(1 for r in augmented if r.get('is_augmented'))

            # Add augmented results
            for aug in augmented:
                if aug.get('is_augmented'):
                    # Convert to EnhancedSearchResult
                    enhanced_result = EnhancedSearchResult(
                        id=aug['id'],
                        final_score=aug.get('augmentation_score', 0.5),
                        is_graph_augmented=True,
                        augmentation_source=aug.get('source_result_id'),
                        relation_path=aug.get('relation_path', [])
                    )
                    base_results.append(enhanced_result)

        # Step 9: Convert to enhanced results
        enhanced_results = []
        for result in base_results[:limit]:
            if isinstance(result, EnhancedSearchResult):
                enhanced_result = result
            else:
                enhanced_result = EnhancedSearchResult(
                    id=result.id,
                    final_score=result.final_score,
                    text=result.text,
                    metadata=result.metadata,
                    bm25_score=result.bm25_score,
                    bm25_rank=result.bm25_rank,
                    dense_score=result.dense_score,
                    dense_rank=result.dense_rank,
                    colpali_score=result.colpali_score,
                    colpali_rank=result.colpali_rank,
                )

            enhanced_result.query_type = query_type.value
            enhanced_result.retrieval_level = retrieval_level.value
            enhanced_result.expansions_applied = enhanced.get('expansions', [])
            enhanced_result.retrieval_confidence = failure_signals['confidence']
            enhanced_result.potential_issues = failure_signals.get('recommendations', [])

            enhanced_results.append(enhanced_result)

        # Step 10: Build analytics
        processing_time = (time.time() - start_time) * 1000

        analytics = SearchAnalytics(
            query=query,
            query_type=query_type,
            detected_domain=detected_domain.value,
            retrieval_methods_used=self._get_methods_used(enhanced_results),
            total_results=len(enhanced_results),
            graph_augmented_count=graph_augmented_count,
            fallback_triggered=fallback_triggered,
            processing_time_ms=processing_time,
            expansions=enhanced.get('expansions', [])
        )

        return enhanced_results, analytics

    def _classify_query(self, query: str) -> QueryType:
        """Classify query type for routing."""
        query_lower = query.lower()

        # Check for comparative
        if any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'between']):
            return QueryType.COMPARATIVE

        # Check for temporal
        if any(word in query_lower for word in ['when', 'date', 'before', 'after', 'during']):
            return QueryType.TEMPORAL

        # Check for reference
        if any(word in query_lower for word in ['table', 'figure', 'section', 'article', 'exhibit']):
            return QueryType.REFERENCE

        # Check for precise
        if any(word in query_lower for word in ['exact', 'specific', 'precise', 'what is the']):
            return QueryType.PRECISE

        return QueryType.BROAD

    def _precision_fallback(
        self,
        query: str,
        precision_type: str,
        reference: str,
        filters: Optional[Dict[str, Any]]
    ) -> List[HybridResult]:
        """
        Fallback for precision queries using exact matching.

        When someone asks for "Table 3" or "Section 5.2",
        semantic search often misses. Use keyword/BM25.
        """
        results = []

        # Use BM25-only search for precision queries
        if self.base_searcher.bm25_store:
            # Build specific query for the reference
            if precision_type == 'table_reference':
                search_terms = [f"Table {reference}", f"table {reference}"]
            elif precision_type == 'section_reference':
                search_terms = [f"Section {reference}", f"section {reference}"]
            elif precision_type == 'exhibit_reference':
                search_terms = [f"Exhibit {reference}", f"exhibit {reference}"]
            else:
                search_terms = [reference]

            for term in search_terms:
                bm25_results = self.base_searcher.bm25_store.search(
                    query=term,
                    limit=5,
                    filters=filters
                )
                for r in bm25_results:
                    results.append(HybridResult(
                        id=r.id,
                        final_score=r.score + 0.5,  # Boost precision results
                        text=r.text,
                        metadata=r.metadata,
                        bm25_score=r.score,
                        bm25_rank=1
                    ))

        return results

    def _apply_fallbacks(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        recommendations: List[str]
    ) -> List[HybridResult]:
        """Apply fallback strategies based on recommendations."""
        results = []

        if 'try_keyword_search' in recommendations:
            # Try pure BM25 search
            if self.base_searcher.bm25_store:
                bm25_results = self.base_searcher.bm25_store.search(
                    query=query,
                    limit=5,
                    filters=filters
                )
                results.extend([
                    HybridResult(
                        id=r.id,
                        final_score=r.score,
                        text=r.text,
                        metadata=r.metadata,
                        bm25_score=r.score
                    )
                    for r in bm25_results
                ])

        if 'expand_search' in recommendations:
            # Try without filters
            expanded = self.base_searcher.search(
                query=query,
                limit=5,
                filters=None  # Remove filters
            )
            results.extend(expanded)

        return results

    def _merge_results(
        self,
        priority_results: List[HybridResult],
        other_results: List[HybridResult]
    ) -> List[HybridResult]:
        """Merge result lists with priority ordering."""
        seen_ids = set()
        merged = []

        # Add priority results first
        for r in priority_results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                merged.append(r)

        # Add other results
        for r in other_results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                merged.append(r)

        return merged

    def _get_methods_used(self, results: List[EnhancedSearchResult]) -> List[str]:
        """Get list of retrieval methods that contributed."""
        methods = set()
        for r in results:
            if r.bm25_rank:
                methods.add('bm25')
            if r.dense_rank:
                methods.add('dense')
            if r.colpali_rank:
                methods.add('colpali')
            if r.is_graph_augmented:
                methods.add('graph')
        return list(methods)


# Example usage
if __name__ == "__main__":
    # This would normally use real stores
    print("Enhanced Hybrid Search module loaded.")
    print("Features:")
    print("  - Query complexity detection")
    print("  - Precision query fallbacks")
    print("  - Acronym expansion")
    print("  - Graph augmentation")
    print("  - Failure detection")
