"""
Document Relationship Graph.

Key insight from enterprise RAG:
- Documents reference other documents constantly
- Drug A study references Drug B interaction data
- Contracts reference amendments and exhibits
- Semantic search misses these relationship networks

This module provides:
- Automatic relationship extraction during ingestion
- Graph-based retrieval augmentation
- Cross-document reference resolution
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of document relationships."""
    # Hierarchical
    AMENDS = "amends"                    # Document A amends Document B
    SUPERSEDES = "supersedes"            # Document A replaces Document B
    EXHIBITS_TO = "exhibits_to"          # Exhibit attached to contract
    SUPPLEMENT_TO = "supplement_to"      # Supplementary document

    # References
    REFERENCES = "references"            # General reference
    CITES = "cites"                      # Legal citation
    INCORPORATES = "incorporates"        # Incorporated by reference

    # Semantic
    RELATED_TO = "related_to"            # Topically related
    CONTRADICTS = "contradicts"          # Conflicting information
    CLARIFIES = "clarifies"              # Provides clarification

    # Temporal
    PRECEDES = "precedes"                # Chronological relationship
    FOLLOWS = "follows"                  # Chronological relationship

    # Legal/Regulatory
    IMPLEMENTS = "implements"            # Implements regulation
    COMPLIES_WITH = "complies_with"      # Compliance relationship


@dataclass
class DocumentRelation:
    """A relationship between two documents."""
    source_id: str          # Document containing the reference
    target_id: str          # Referenced document
    relation_type: RelationType
    confidence: float       # 0.0 to 1.0
    context: str = ""       # Text where relationship was found
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DocumentNode:
    """Node in the document graph."""
    document_id: str
    title: Optional[str] = None
    document_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Relationships
    outgoing_relations: List[DocumentRelation] = field(default_factory=list)
    incoming_relations: List[DocumentRelation] = field(default_factory=list)


class RelationshipExtractor:
    """
    Extracts document relationships from text.

    Uses pattern matching (NOT LLM) for consistency.
    """

    # Patterns for different relationship types
    PATTERNS = {
        # Amendment patterns
        RelationType.AMENDS: [
            r'amend(?:s|ed|ment to)?\s+(?:the\s+)?(?:original\s+)?(?:agreement|contract)?\s*(?:dated\s+)?([A-Z][^.]+?(?:\d{4}|agreement|contract))',
            r'(?:first|second|third|\d+(?:st|nd|rd|th))\s+amendment\s+to\s+([^.]+)',
        ],

        # Supersedes patterns
        RelationType.SUPERSEDES: [
            r'supersedes?\s+(?:and replaces?\s+)?(?:all\s+)?(?:prior\s+)?([^.]+)',
            r'replaces?\s+(?:in\s+its\s+entirety\s+)?([^.]+)',
        ],

        # Exhibit patterns
        RelationType.EXHIBITS_TO: [
            r'exhibit\s+([A-Z]|\d+)\s+(?:to|attached)',
            r'(?:see|refer to)\s+exhibit\s+([A-Z]|\d+)',
        ],

        # General reference patterns
        RelationType.REFERENCES: [
            r'(?:pursuant to|in accordance with|as defined in|as set forth in)\s+([^.]+)',
            r'(?:see|refer to|reference to)\s+([^.]+?(?:agreement|contract|document|section|article))',
        ],

        # Legal citations (IRC, CFR, USC)
        RelationType.CITES: [
            r'(?:IRC|I\.R\.C\.)\s*(?:§|Section)\s*(\d+(?:\([a-z]\))?(?:\(\d+\))?)',
            r'(\d+)\s*(?:U\.S\.C\.|USC)\s*(?:§|Section)?\s*(\d+)',
            r'(\d+)\s*(?:C\.F\.R\.|CFR)\s*(?:§|Section)?\s*(\d+(?:\.\d+)?)',
        ],

        # Incorporation by reference
        RelationType.INCORPORATES: [
            r'incorporat(?:es?|ed|ing)\s+(?:herein\s+)?by\s+reference\s+([^.]+)',
            r'deemed\s+(?:to\s+be\s+)?incorporated\s+([^.]+)',
        ],
    }

    def __init__(self):
        self.compiled_patterns = {
            rel_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for rel_type, patterns in self.PATTERNS.items()
        }

        # Known document identifiers (populated during processing)
        self.known_documents: Dict[str, str] = {}  # identifier -> document_id

    def extract_relationships(
        self,
        text: str,
        source_document_id: str
    ) -> List[DocumentRelation]:
        """
        Extract relationships from document text.

        Args:
            text: Document text
            source_document_id: ID of document being processed

        Returns:
            List of extracted relationships
        """
        relations = []

        for rel_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Get the referenced document identifier
                    target_ref = match.group(1) if match.groups() else match.group(0)
                    target_ref = target_ref.strip()

                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_confidence(rel_type, match.group(0))

                    # Get surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]

                    # Try to resolve target to known document
                    target_id = self._resolve_reference(target_ref)

                    relation = DocumentRelation(
                        source_id=source_document_id,
                        target_id=target_id or f"unresolved:{target_ref}",
                        relation_type=rel_type,
                        confidence=confidence,
                        context=context,
                        metadata={
                            'raw_reference': target_ref,
                            'pattern_matched': pattern.pattern,
                        }
                    )
                    relations.append(relation)

        # Deduplicate similar relations
        relations = self._deduplicate_relations(relations)

        return relations

    def _calculate_confidence(self, rel_type: RelationType, matched_text: str) -> float:
        """Calculate confidence score for extraction."""
        confidence = 0.7  # Base confidence

        # Higher confidence for specific patterns
        if rel_type in [RelationType.CITES, RelationType.EXHIBITS_TO]:
            confidence = 0.9  # Very specific patterns

        # Lower confidence for vague references
        if len(matched_text) > 200:
            confidence -= 0.2  # Long matches are less precise

        if any(word in matched_text.lower() for word in ['may', 'might', 'similar']):
            confidence -= 0.1  # Hedging language

        return max(0.3, min(1.0, confidence))

    def _resolve_reference(self, reference: str) -> Optional[str]:
        """Try to resolve a reference to a known document ID."""
        reference_lower = reference.lower()

        # Check against known documents
        for identifier, doc_id in self.known_documents.items():
            if identifier.lower() in reference_lower:
                return doc_id

        return None

    def _deduplicate_relations(
        self,
        relations: List[DocumentRelation]
    ) -> List[DocumentRelation]:
        """Remove duplicate relations."""
        seen = set()
        unique = []

        for rel in relations:
            key = (rel.source_id, rel.target_id, rel.relation_type)
            if key not in seen:
                seen.add(key)
                unique.append(rel)

        return unique

    def register_document(self, document_id: str, identifiers: List[str]):
        """Register a document with its identifiers for resolution."""
        for identifier in identifiers:
            self.known_documents[identifier] = document_id


class DocumentGraph:
    """
    Graph structure for document relationships.

    Enables graph-based retrieval augmentation.
    """

    def __init__(self):
        self.nodes: Dict[str, DocumentNode] = {}
        self.extractor = RelationshipExtractor()

        # Indexes for efficient lookup
        self.relations_by_type: Dict[RelationType, List[DocumentRelation]] = defaultdict(list)
        self.unresolved_references: Dict[str, List[DocumentRelation]] = defaultdict(list)

    def add_document(
        self,
        document_id: str,
        title: str = None,
        document_type: str = None,
        identifiers: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> DocumentNode:
        """Add a document node to the graph."""
        node = DocumentNode(
            document_id=document_id,
            title=title,
            document_type=document_type,
            metadata=metadata or {}
        )
        self.nodes[document_id] = node

        # Register identifiers for reference resolution
        if identifiers:
            self.extractor.register_document(document_id, identifiers)

        # Try to resolve pending unresolved references
        self._resolve_pending_references(document_id, identifiers or [])

        return node

    def process_document(
        self,
        document_id: str,
        text: str,
        title: str = None,
        document_type: str = None,
        identifiers: List[str] = None
    ) -> List[DocumentRelation]:
        """
        Process a document, extracting relationships and adding to graph.

        Args:
            document_id: Document ID
            text: Document text
            title: Document title
            document_type: Type of document
            identifiers: Document identifiers (contract numbers, etc.)

        Returns:
            List of extracted relationships
        """
        # Ensure node exists
        if document_id not in self.nodes:
            self.add_document(document_id, title, document_type, identifiers)

        # Extract relationships
        relations = self.extractor.extract_relationships(text, document_id)

        # Add relationships to graph
        for relation in relations:
            self._add_relation(relation)

        return relations

    def _add_relation(self, relation: DocumentRelation):
        """Add a relationship to the graph."""
        source_node = self.nodes.get(relation.source_id)

        if source_node:
            source_node.outgoing_relations.append(relation)

        # If target is resolved, add incoming relation
        if not relation.target_id.startswith("unresolved:"):
            target_node = self.nodes.get(relation.target_id)
            if target_node:
                target_node.incoming_relations.append(relation)
        else:
            # Track unresolved for later resolution
            ref_key = relation.target_id.replace("unresolved:", "")
            self.unresolved_references[ref_key].append(relation)

        # Index by type
        self.relations_by_type[relation.relation_type].append(relation)

    def _resolve_pending_references(
        self,
        document_id: str,
        identifiers: List[str]
    ):
        """Try to resolve pending unresolved references."""
        for identifier in identifiers:
            if identifier in self.unresolved_references:
                for relation in self.unresolved_references[identifier]:
                    # Update relation
                    relation.target_id = document_id
                    relation.confidence = min(1.0, relation.confidence + 0.1)

                    # Add incoming relation to target
                    target_node = self.nodes.get(document_id)
                    if target_node:
                        target_node.incoming_relations.append(relation)

                del self.unresolved_references[identifier]

    def get_related_documents(
        self,
        document_id: str,
        relation_types: List[RelationType] = None,
        max_depth: int = 1,
        min_confidence: float = 0.5
    ) -> List[Tuple[str, List[DocumentRelation]]]:
        """
        Get documents related to the given document.

        Args:
            document_id: Starting document
            relation_types: Filter by relation types
            max_depth: How many hops to traverse
            min_confidence: Minimum relation confidence

        Returns:
            List of (document_id, path_of_relations) tuples
        """
        if document_id not in self.nodes:
            return []

        visited = {document_id}
        results = []
        queue = [(document_id, [], 0)]  # (doc_id, path, depth)

        while queue:
            current_id, path, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            node = self.nodes.get(current_id)
            if not node:
                continue

            # Check outgoing relations
            for relation in node.outgoing_relations:
                if relation.confidence < min_confidence:
                    continue
                if relation_types and relation.relation_type not in relation_types:
                    continue

                target_id = relation.target_id
                if target_id.startswith("unresolved:"):
                    continue
                if target_id in visited:
                    continue

                visited.add(target_id)
                new_path = path + [relation]
                results.append((target_id, new_path))

                if depth + 1 < max_depth:
                    queue.append((target_id, new_path, depth + 1))

            # Check incoming relations (reverse traversal)
            for relation in node.incoming_relations:
                if relation.confidence < min_confidence:
                    continue
                if relation_types and relation.relation_type not in relation_types:
                    continue

                source_id = relation.source_id
                if source_id in visited:
                    continue

                visited.add(source_id)
                new_path = path + [relation]
                results.append((source_id, new_path))

                if depth + 1 < max_depth:
                    queue.append((source_id, new_path, depth + 1))

        return results

    def get_document_context(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Get full context for a document including all relationships.

        Useful for enriching search results with relationship info.
        """
        if document_id not in self.nodes:
            return {}

        node = self.nodes[document_id]

        return {
            'document_id': document_id,
            'title': node.title,
            'document_type': node.document_type,
            'outgoing_relations': [
                {
                    'target': r.target_id,
                    'type': r.relation_type.value,
                    'confidence': r.confidence,
                }
                for r in node.outgoing_relations
            ],
            'incoming_relations': [
                {
                    'source': r.source_id,
                    'type': r.relation_type.value,
                    'confidence': r.confidence,
                }
                for r in node.incoming_relations
            ],
            'related_document_count': len(set(
                r.target_id for r in node.outgoing_relations
            ) | set(
                r.source_id for r in node.incoming_relations
            ))
        }

    def find_document_chain(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> Optional[List[DocumentRelation]]:
        """
        Find relationship chain between two documents.

        Uses BFS for shortest path.
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        if start_id == end_id:
            return []

        visited = {start_id}
        queue = [(start_id, [])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            node = self.nodes.get(current_id)
            if not node:
                continue

            for relation in node.outgoing_relations:
                target_id = relation.target_id
                if target_id.startswith("unresolved:"):
                    continue

                new_path = path + [relation]

                if target_id == end_id:
                    return new_path

                if target_id not in visited:
                    visited.add(target_id)
                    queue.append((target_id, new_path))

        return None

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the document graph."""
        total_relations = sum(len(rels) for rels in self.relations_by_type.values())

        return {
            'total_documents': len(self.nodes),
            'total_relations': total_relations,
            'unresolved_references': len(self.unresolved_references),
            'relations_by_type': {
                rel_type.value: len(rels)
                for rel_type, rels in self.relations_by_type.items()
            },
            'avg_relations_per_document': total_relations / len(self.nodes) if self.nodes else 0,
        }


class GraphAugmentedRetrieval:
    """
    Augments search results with graph-based relationships.

    After semantic search, checks if retrieved docs have related
    documents that might have better answers.
    """

    def __init__(self, graph: DocumentGraph):
        self.graph = graph

    def augment_results(
        self,
        search_results: List[Dict[str, Any]],
        relation_types: List[RelationType] = None,
        max_augmented: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Augment search results with related documents.

        Args:
            search_results: Initial search results (dicts with 'id' key)
            relation_types: Types of relations to follow
            max_augmented: Max additional documents to add

        Returns:
            Augmented results list
        """
        augmented = list(search_results)
        seen_ids = {r.get('id') for r in search_results}

        # For each result, find related documents
        candidates = []

        for result in search_results[:5]:  # Check top 5 results
            doc_id = result.get('document_id') or result.get('id')
            if not doc_id:
                continue

            related = self.graph.get_related_documents(
                doc_id,
                relation_types=relation_types,
                max_depth=2
            )

            for related_id, path in related:
                if related_id in seen_ids:
                    continue

                # Score based on path strength
                path_score = sum(r.confidence for r in path) / len(path) if path else 0

                candidates.append({
                    'id': related_id,
                    'source_result_id': doc_id,
                    'relation_path': [
                        {'type': r.relation_type.value, 'target': r.target_id}
                        for r in path
                    ],
                    'augmentation_score': path_score,
                    'is_augmented': True,
                })

        # Sort candidates by score and add top ones
        candidates.sort(key=lambda x: x['augmentation_score'], reverse=True)

        for candidate in candidates[:max_augmented]:
            if candidate['id'] not in seen_ids:
                seen_ids.add(candidate['id'])
                augmented.append(candidate)

        return augmented


# Example usage
if __name__ == "__main__":
    # Create graph
    graph = DocumentGraph()

    # Add some documents
    graph.add_document(
        "contract_001",
        title="Master Services Agreement",
        document_type="contract",
        identifiers=["MSA-2024-001", "Master Services Agreement dated January 1, 2024"]
    )

    graph.add_document(
        "amendment_001",
        title="First Amendment to MSA",
        document_type="amendment",
        identifiers=["AMD-2024-001"]
    )

    # Process document to extract relationships
    amendment_text = """
    FIRST AMENDMENT TO MASTER SERVICES AGREEMENT

    This First Amendment amends the Master Services Agreement dated January 1, 2024
    between ABC Corp and XYZ Inc.

    Pursuant to Section 15.2 of the original agreement, the parties hereby agree
    to modify the payment terms as set forth in Exhibit A attached hereto.

    This Amendment incorporates by reference all terms of the original MSA
    except as specifically modified herein.

    This Amendment shall be governed by IRC Section 7701 for tax purposes.
    """

    relations = graph.process_document(
        "amendment_001",
        amendment_text,
        title="First Amendment"
    )

    print("Extracted relationships:")
    for rel in relations:
        print(f"  {rel.relation_type.value}: {rel.source_id} -> {rel.target_id}")
        print(f"    Confidence: {rel.confidence}")

    # Get related documents
    print("\nRelated documents for amendment_001:")
    related = graph.get_related_documents("amendment_001", max_depth=2)
    for doc_id, path in related:
        print(f"  {doc_id} via {[r.relation_type.value for r in path]}")

    # Get graph stats
    print("\nGraph statistics:")
    stats = graph.get_graph_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
