"""
Query Modality Detection for Dynamic Retrieval Routing.

This module detects whether a query targets visual content (tables, figures, charts)
and adjusts retrieval weights accordingly.

Based on the ColPALI Meets DocLayNet approach:
- Visual queries → boost ColPali weight
- Text queries → boost Dense/BM25 weight
"""
import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryModality(Enum):
    """Query modality classification."""
    TEXT = "text"  # Pure text query
    VISUAL = "visual"  # Targets visual elements
    MIXED = "mixed"  # Both text and visual components


class VisualElementTarget(Enum):
    """Specific visual element types a query might target."""
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    DIAGRAM = "diagram"
    IMAGE = "image"
    GRAPH = "graph"
    FORMULA = "formula"


@dataclass
class QueryAnalysis:
    """Result of query modality analysis."""
    query: str
    modality: QueryModality
    visual_score: float  # 0.0 = pure text, 1.0 = highly visual
    target_element_types: List[VisualElementTarget]
    suggested_weights: Dict[str, float]
    detected_patterns: List[str]

    def is_visual(self) -> bool:
        return self.modality in (QueryModality.VISUAL, QueryModality.MIXED)

    def should_search_visual_elements(self) -> bool:
        """Whether to search the visual_elements collection."""
        return self.visual_score >= 0.3 or len(self.target_element_types) > 0


class QueryModalityDetector:
    """
    Detect if a query targets visual content and suggest retrieval weights.

    This enables dynamic routing:
    - "What does the table show?" → boost ColPali weight
    - "Summarize the contract terms" → boost Dense weight
    - "Find the chart showing revenue" → search visual_elements collection
    """

    # Patterns that indicate visual content queries
    VISUAL_PATTERNS = {
        # Direct references to visual elements
        r'\b(table|tables)\b': VisualElementTarget.TABLE,
        r'\b(figure|figures|fig\.?)\b': VisualElementTarget.FIGURE,
        r'\b(chart|charts)\b': VisualElementTarget.CHART,
        r'\b(diagram|diagrams)\b': VisualElementTarget.DIAGRAM,
        r'\b(image|images|picture|pictures|photo|photos)\b': VisualElementTarget.IMAGE,
        r'\b(graph|graphs)\b': VisualElementTarget.GRAPH,
        r'\b(formula|formulas|equation|equations)\b': VisualElementTarget.FORMULA,
    }

    # Patterns that indicate visual query intent
    VISUAL_INTENT_PATTERNS = [
        r'what\s+does\s+the\s+(table|figure|chart|diagram)\s+show',
        r'according\s+to\s+the\s+(table|figure|chart)',
        r'in\s+the\s+(table|figure|chart|diagram)',
        r'from\s+the\s+(table|figure|chart)',
        r'(show|display|visualize|illustrate)',
        r'(look\s+at|see|view)\s+the\s+(table|figure|chart)',
        r'(data|values|numbers)\s+in\s+the\s+table',
        r'table\s+(shows?|displays?|contains?|lists?)',
        r'figure\s+(shows?|displays?|illustrates?)',
    ]

    # Default weights for different modalities
    DEFAULT_WEIGHTS = {
        QueryModality.TEXT: {"bm25": 0.35, "dense": 0.55, "colpali": 0.10},
        QueryModality.VISUAL: {"bm25": 0.15, "dense": 0.25, "colpali": 0.60},
        QueryModality.MIXED: {"bm25": 0.25, "dense": 0.35, "colpali": 0.40},
    }

    def __init__(
        self,
        custom_weights: Optional[Dict[QueryModality, Dict[str, float]]] = None,
        visual_threshold: float = 0.3,
    ):
        """
        Initialize the query modality detector.

        Args:
            custom_weights: Override default weights for each modality
            visual_threshold: Score threshold to classify as visual query
        """
        self.weights = custom_weights or self.DEFAULT_WEIGHTS
        self.visual_threshold = visual_threshold

        # Compile patterns for efficiency
        self._visual_patterns = {
            re.compile(pattern, re.IGNORECASE): target
            for pattern, target in self.VISUAL_PATTERNS.items()
        }
        self._intent_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.VISUAL_INTENT_PATTERNS
        ]

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to determine its modality and suggest weights.

        Args:
            query: The search query

        Returns:
            QueryAnalysis with modality, suggested weights, and target elements
        """
        query_lower = query.lower()
        detected_patterns = []
        target_elements = []

        # Check for visual element references
        visual_score = 0.0
        for pattern, target in self._visual_patterns.items():
            if pattern.search(query_lower):
                visual_score += 0.3
                detected_patterns.append(pattern.pattern)
                if target not in target_elements:
                    target_elements.append(target)

        # Check for visual intent patterns
        for pattern in self._intent_patterns:
            if pattern.search(query_lower):
                visual_score += 0.25
                detected_patterns.append(pattern.pattern)

        # Normalize score
        visual_score = min(visual_score, 1.0)

        # Determine modality
        if visual_score >= 0.5:
            modality = QueryModality.VISUAL
        elif visual_score >= self.visual_threshold:
            modality = QueryModality.MIXED
        else:
            modality = QueryModality.TEXT

        # Get suggested weights
        suggested_weights = self.weights[modality].copy()

        # If targeting specific elements, further boost ColPali
        if len(target_elements) > 0:
            boost = min(0.1 * len(target_elements), 0.15)
            suggested_weights["colpali"] = min(suggested_weights["colpali"] + boost, 0.7)
            # Normalize
            total = sum(suggested_weights.values())
            suggested_weights = {k: v / total for k, v in suggested_weights.items()}

        logger.debug(
            f"Query analysis: modality={modality.value}, "
            f"visual_score={visual_score:.2f}, targets={[t.value for t in target_elements]}"
        )

        return QueryAnalysis(
            query=query,
            modality=modality,
            visual_score=visual_score,
            target_element_types=target_elements,
            suggested_weights=suggested_weights,
            detected_patterns=detected_patterns,
        )

    def get_weights_for_query(self, query: str) -> Dict[str, float]:
        """
        Convenience method to get suggested weights for a query.

        Args:
            query: The search query

        Returns:
            Dict with bm25, dense, colpali weights
        """
        analysis = self.analyze(query)
        return analysis.suggested_weights

    def should_search_visual_elements(self, query: str) -> bool:
        """
        Check if the query should search the visual_elements collection.

        Args:
            query: The search query

        Returns:
            True if visual elements collection should be searched
        """
        analysis = self.analyze(query)
        return analysis.should_search_visual_elements()

    def get_target_element_types(self, query: str) -> List[str]:
        """
        Get the visual element types targeted by a query.

        Args:
            query: The search query

        Returns:
            List of element type strings for filtering (e.g., ["Table", "Figure"])
        """
        analysis = self.analyze(query)

        # Map to Unstructured element types
        type_mapping = {
            VisualElementTarget.TABLE: "Table",
            VisualElementTarget.FIGURE: "Image",  # Figures often classified as Image
            VisualElementTarget.CHART: "Image",
            VisualElementTarget.DIAGRAM: "Image",
            VisualElementTarget.IMAGE: "Image",
            VisualElementTarget.GRAPH: "Image",
            VisualElementTarget.FORMULA: "Formula",
        }

        element_types = list(set(
            type_mapping.get(t, "Image") for t in analysis.target_element_types
        ))

        return element_types if element_types else None


# Singleton instance for convenience
_default_detector = None


def get_query_analyzer() -> QueryModalityDetector:
    """Get the default query analyzer instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = QueryModalityDetector()
    return _default_detector


# Example usage
if __name__ == "__main__":
    detector = QueryModalityDetector()

    test_queries = [
        "What does the table show about revenue?",
        "Summarize the contract terms",
        "According to Figure 3, what is the trend?",
        "Find the chart showing quarterly sales",
        "What are the termination clauses?",
        "In the diagram on page 5",
        "Show me the data table",
        "List all parties to the agreement",
    ]

    for query in test_queries:
        analysis = detector.analyze(query)
        print(f"\nQuery: {query}")
        print(f"  Modality: {analysis.modality.value}")
        print(f"  Visual Score: {analysis.visual_score:.2f}")
        print(f"  Target Elements: {[t.value for t in analysis.target_element_types]}")
        print(f"  Weights: {analysis.suggested_weights}")
        print(f"  Search Visual Elements: {analysis.should_search_visual_elements()}")
