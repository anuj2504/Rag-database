"""
Enhanced Ingestion Pipeline with Enterprise Features.

Key improvements:
1. Quality-based routing to different processing pipelines
2. Hierarchical chunking with structure preservation
3. Domain-specific metadata extraction
4. Table extraction and dual embedding
5. Document relationship graph building
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, field

from src.quality.document_quality import (
    DocumentQualityAnalyzer,
    QualityBasedRouter,
    QualityTier,
    QualityReport
)
from src.chunking.hierarchical_chunker import (
    HierarchicalChunker,
    ChunkLevel,
    HierarchicalChunk,
    LegalDocumentDetector,
    FinancialDocumentDetector
)
from src.metadata.domain_schemas import (
    UnifiedMetadataExtractor,
    ExtractedMetadata
)
from src.tables.table_extractor import (
    TableExtractor,
    ExtractedTable
)
from src.graph.document_graph import (
    DocumentGraph,
    DocumentRelation
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedIngestionResult:
    """Detailed result of document ingestion."""
    document_id: str
    filename: str
    status: str  # 'success', 'partial', 'failed'

    # Quality info
    quality_tier: str
    quality_score: float
    quality_issues: List[str] = field(default_factory=list)

    # Processing stats
    chunks_created: int = 0
    tables_extracted: int = 0
    relationships_found: int = 0

    # Indexing stats
    chunks_indexed: int = 0
    pages_indexed: int = 0

    # Metadata
    document_type: Optional[str] = None
    extracted_metadata: Dict[str, Any] = field(default_factory=dict)

    # Errors
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0


class EnhancedIngestionPipeline:
    """
    Enterprise-grade document ingestion pipeline.

    Flow:
    1. Document processing (Unstructured.io)
    2. Quality assessment and routing
    3. Structure-aware chunking
    4. Domain-specific metadata extraction
    5. Table extraction
    6. Relationship graph building
    7. Embedding and indexing
    """

    def __init__(
        self,
        # Core processors
        document_processor,
        dense_embedder,
        colpali_embedder=None,
        # Stores
        metadata_store=None,
        dense_vector_store=None,
        colpali_vector_store=None,
        bm25_store=None,
        # Enhanced components
        document_graph: DocumentGraph = None,
        # Config
        batch_size: int = 32,
        enable_tables: bool = True,
        enable_graph: bool = True,
        enable_colpali: bool = True,
    ):
        # Core components
        self.document_processor = document_processor
        self.dense_embedder = dense_embedder
        self.colpali_embedder = colpali_embedder

        # Stores
        self.metadata_store = metadata_store
        self.dense_vector_store = dense_vector_store
        self.colpali_vector_store = colpali_vector_store
        self.bm25_store = bm25_store

        # Enhanced components
        self.quality_analyzer = DocumentQualityAnalyzer()
        self.quality_router = QualityBasedRouter(self.quality_analyzer)
        self.metadata_extractor = UnifiedMetadataExtractor()
        self.table_extractor = TableExtractor()
        self.document_graph = document_graph or DocumentGraph()

        # Chunkers for different document types
        self.legal_chunker = HierarchicalChunker(
            detector=LegalDocumentDetector(),
            paragraph_size=400
        )
        self.financial_chunker = HierarchicalChunker(
            detector=FinancialDocumentDetector(),
            paragraph_size=500
        )
        self.default_chunker = HierarchicalChunker(paragraph_size=400)

        # Config
        self.batch_size = batch_size
        self.enable_tables = enable_tables
        self.enable_graph = enable_graph
        self.enable_colpali = enable_colpali and colpali_embedder is not None

    def ingest_document(
        self,
        file_path: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> EnhancedIngestionResult:
        """
        Ingest a document with full enterprise processing.

        Args:
            file_path: Path to document
            custom_metadata: Additional metadata

        Returns:
            EnhancedIngestionResult with detailed stats
        """
        start_time = datetime.now()
        filename = Path(file_path).name

        try:
            logger.info(f"Starting enhanced ingestion: {filename}")

            # Step 1: Initial document processing
            logger.info("[1/8] Processing document with Unstructured.io")
            processed_doc = self.document_processor.process_document(
                file_path=file_path,
                custom_metadata=custom_metadata
            )
            document_id = processed_doc.document_id

            # Step 2: Quality assessment and routing
            logger.info("[2/8] Assessing document quality")
            full_text = '\n'.join(chunk.text for chunk in processed_doc.chunks)
            routing = self.quality_router.route(full_text, filename)
            quality_report: QualityReport = routing['report']
            pipeline_config = routing['config']

            # Check if document is too low quality
            if quality_report.tier == QualityTier.GARBAGE:
                if pipeline_config.get('skip_embedding'):
                    logger.warning(f"Document quality too low, flagging for manual review: {filename}")
                    return EnhancedIngestionResult(
                        document_id=document_id,
                        filename=filename,
                        status='partial',
                        quality_tier=quality_report.tier.value,
                        quality_score=quality_report.overall_score,
                        quality_issues=quality_report.issues,
                        error_message="Document flagged for manual review due to low quality"
                    )

            # Step 3: Domain-specific metadata extraction
            logger.info("[3/8] Extracting domain-specific metadata")
            extracted_metadata = self.metadata_extractor.extract(
                full_text, filename
            )

            # Step 4: Structure-aware chunking based on document type
            logger.info("[4/8] Creating hierarchical chunks")
            chunks = self._create_chunks(
                full_text, document_id,
                extracted_metadata.document_type,
                pipeline_config
            )

            # Step 5: Table extraction
            tables = []
            if self.enable_tables and pipeline_config.get('enable_table_extraction', True):
                logger.info("[5/8] Extracting tables")
                tables = self.table_extractor.extract_tables(full_text, document_id)
                logger.info(f"  Found {len(tables)} tables")

            # Step 6: Relationship extraction
            relationships = []
            if self.enable_graph:
                logger.info("[6/8] Extracting document relationships")
                relationships = self.document_graph.process_document(
                    document_id=document_id,
                    text=full_text,
                    title=extracted_metadata.title,
                    document_type=extracted_metadata.document_type,
                    identifiers=self._extract_identifiers(extracted_metadata)
                )
                logger.info(f"  Found {len(relationships)} relationships")

            # Step 7: Store metadata
            if self.metadata_store:
                logger.info("[7/8] Storing metadata")
                self._store_metadata(
                    processed_doc, extracted_metadata,
                    quality_report, chunks, tables
                )

            # Step 8: Create embeddings and index
            logger.info("[8/8] Creating embeddings and indexing")
            chunks_indexed = self._index_chunks(chunks, document_id, extracted_metadata)
            tables_indexed = self._index_tables(tables)
            pages_indexed = 0

            if self.enable_colpali and processed_doc.page_images:
                pages_indexed = self._index_colpali(processed_doc)

            # Index in BM25
            self._index_bm25(chunks, tables, document_id, extracted_metadata)

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Completed enhanced ingestion: {filename} "
                f"(quality: {quality_report.tier.value}, "
                f"chunks: {chunks_indexed}, tables: {len(tables)}, "
                f"relationships: {len(relationships)}, "
                f"time: {processing_time:.2f}s)"
            )

            return EnhancedIngestionResult(
                document_id=document_id,
                filename=filename,
                status='success',
                quality_tier=quality_report.tier.value,
                quality_score=quality_report.overall_score,
                quality_issues=quality_report.issues,
                chunks_created=len(chunks),
                tables_extracted=len(tables),
                relationships_found=len(relationships),
                chunks_indexed=chunks_indexed,
                pages_indexed=pages_indexed,
                document_type=extracted_metadata.document_type,
                extracted_metadata=extracted_metadata.filter_tags,
                processing_time_seconds=processing_time
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to ingest {filename}: {e}", exc_info=True)

            return EnhancedIngestionResult(
                document_id=locals().get('document_id', 'unknown'),
                filename=filename,
                status='failed',
                quality_tier='unknown',
                quality_score=0.0,
                error_message=str(e),
                processing_time_seconds=processing_time
            )

    def _create_chunks(
        self,
        text: str,
        document_id: str,
        document_type: str,
        config: Dict[str, Any]
    ) -> List[HierarchicalChunk]:
        """Create chunks based on document type and quality."""
        chunking_strategy = config.get('chunking_strategy', 'standard')

        if chunking_strategy == 'hierarchical':
            # Use structure-aware chunking
            if document_type in ['contract', 'irc_code', 'building_code']:
                chunker = self.legal_chunker
            elif document_type == 'financial_report':
                chunker = self.financial_chunker
            else:
                chunker = self.default_chunker

            return chunker.chunk(text, document_id, detect_structure=True)

        elif chunking_strategy == 'simple':
            # Simple fixed-size chunking for low-quality docs
            chunk_size = config.get('chunk_sizes', {}).get('fixed', 400)
            overlap = config.get('chunk_overlap', 50)
            return self._simple_chunk(text, document_id, chunk_size, overlap)

        else:
            # Standard chunking
            return self.default_chunker.chunk(text, document_id, detect_structure=False)

    def _simple_chunk(
        self,
        text: str,
        document_id: str,
        chunk_size: int,
        overlap: int
    ) -> List[HierarchicalChunk]:
        """Simple fixed-size chunking for low-quality documents."""
        chunks = []
        words = text.split()

        i = 0
        chunk_index = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunk = HierarchicalChunk(
                id=f"{document_id}_chunk_{chunk_index}",
                text=chunk_text,
                level=ChunkLevel.PARAGRAPH,
                document_id=document_id,
                paragraph_index=chunk_index,
                word_count=len(chunk_words)
            )
            chunks.append(chunk)

            i += chunk_size - overlap
            chunk_index += 1

        return chunks

    def _extract_identifiers(self, metadata: ExtractedMetadata) -> List[str]:
        """Extract document identifiers for graph resolution."""
        identifiers = []

        if metadata.title:
            identifiers.append(metadata.title)

        # Add any references found
        identifiers.extend(metadata.references[:10])

        return identifiers

    def _store_metadata(
        self,
        processed_doc,
        extracted_metadata: ExtractedMetadata,
        quality_report: QualityReport,
        chunks: List[HierarchicalChunk],
        tables: List[ExtractedTable]
    ):
        """Store all metadata in PostgreSQL."""
        # Create/update document record
        self.metadata_store.create_document(
            document_id=processed_doc.document_id,
            filename=processed_doc.filename,
            file_path=processed_doc.file_path,
            document_type=extracted_metadata.document_type,
            metadata={
                'quality_score': quality_report.overall_score,
                'quality_tier': quality_report.tier.value,
                'quality_issues': quality_report.issues,
                'extracted_metadata': extracted_metadata.filter_tags,
                'parties': extracted_metadata.parties,
                'key_terms': extracted_metadata.key_terms,
            }
        )

        # Create chunk records
        chunk_records = []
        for chunk in chunks:
            chunk_records.append({
                'id': chunk.id,
                'document_id': processed_doc.document_id,
                'text': chunk.text,
                'chunk_index': chunk.paragraph_index,
                'page_number': None,
                'element_type': chunk.level.value,
                'metadata': {
                    'section_title': chunk.section_title,
                    'parent_id': chunk.parent_id,
                    'word_count': chunk.word_count,
                }
            })
        self.metadata_store.create_chunks(chunk_records)

    def _index_chunks(
        self,
        chunks: List[HierarchicalChunk],
        document_id: str,
        metadata: ExtractedMetadata
    ) -> int:
        """Create embeddings and index chunks."""
        if not self.dense_vector_store:
            return 0

        # Get paragraph-level chunks (primary retrieval unit)
        para_chunks = [c for c in chunks if c.level == ChunkLevel.PARAGRAPH]

        if not para_chunks:
            return 0

        # Create embeddings
        texts = [c.text for c in para_chunks]
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.dense_embedder.embed_texts(batch)
            embeddings.append(batch_embeddings)

        import numpy as np
        all_embeddings = np.vstack(embeddings)

        # Prepare payloads with enriched metadata
        ids = [c.id for c in para_chunks]
        payloads = [
            {
                'document_id': document_id,
                'document_type': metadata.document_type,
                'chunk_level': c.level.value,
                'chunk_index': c.paragraph_index,
                'section_title': c.section_title,
                'text': c.text[:1000],
                # Add filter tags for search
                **metadata.filter_tags
            }
            for c in para_chunks
        ]

        # Store in vector database
        self.dense_vector_store.add_documents(ids, all_embeddings, payloads)

        # Mark as indexed
        if self.metadata_store:
            self.metadata_store.mark_chunks_indexed(ids)

        return len(para_chunks)

    def _index_tables(self, tables: List[ExtractedTable]) -> int:
        """Index extracted tables with dual embedding."""
        if not self.dense_vector_store or not tables:
            return 0

        # Get table chunks (both structured and semantic)
        table_chunks = self.table_extractor.get_table_chunks(tables)

        if not table_chunks:
            return 0

        texts = [c['text'] for c in table_chunks]
        embeddings = self.dense_embedder.embed_texts(texts)

        ids = [c['id'] for c in table_chunks]
        payloads = [
            {
                'document_id': c['document_id'],
                'table_id': c['table_id'],
                'chunk_type': c['type'],
                'text': c['text'][:1000],
                **c['metadata']
            }
            for c in table_chunks
        ]

        self.dense_vector_store.add_documents(ids, embeddings, payloads)

        return len(tables)

    def _index_colpali(self, processed_doc) -> int:
        """Index page images with ColPali."""
        if not self.colpali_vector_store or not processed_doc.page_images:
            return 0

        # Create page embeddings
        page_embeddings = self.colpali_embedder.embed_images(processed_doc.page_images)

        # Prepare data
        ids = [
            f"{processed_doc.document_id}_page_{i}"
            for i in range(len(processed_doc.page_images))
        ]
        payloads = [
            {
                'document_id': processed_doc.document_id,
                'page_number': i + 1,
                'filename': processed_doc.filename,
            }
            for i in range(len(processed_doc.page_images))
        ]

        self.colpali_vector_store.add_pages(ids, page_embeddings, payloads)

        return len(page_embeddings)

    def _index_bm25(
        self,
        chunks: List[HierarchicalChunk],
        tables: List[ExtractedTable],
        document_id: str,
        metadata: ExtractedMetadata
    ):
        """Index in BM25 for keyword search."""
        if not self.bm25_store:
            return

        documents = []

        # Add chunks
        for chunk in chunks:
            if chunk.level == ChunkLevel.PARAGRAPH:
                documents.append({
                    'id': chunk.id,
                    'text': chunk.text,
                    'document_id': document_id,
                    'document_type': metadata.document_type,
                    'section_title': chunk.section_title,
                })

        # Add table semantic descriptions
        for table in tables:
            documents.append({
                'id': f"{table.id}_semantic",
                'text': table.semantic_description,
                'document_id': document_id,
                'document_type': metadata.document_type,
                'is_table': True,
            })

        self.bm25_store.add_documents(documents)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get document graph statistics."""
        return self.document_graph.get_graph_stats()


def create_enhanced_pipeline(
    postgres_url: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    dense_model: str = "BAAI/bge-base-en-v1.5",
    enable_colpali: bool = False,
    enable_tables: bool = True,
    enable_graph: bool = True,
) -> EnhancedIngestionPipeline:
    """
    Factory function to create fully configured enhanced pipeline.
    """
    from src.ingestion.document_processor import DocumentProcessor
    from src.embeddings.dense_embedder import get_embedder
    from src.storage.metadata_store import MetadataStore
    from src.storage.vector_store import QdrantVectorStore, QdrantMultiVectorStore
    from src.storage.bm25_store import BM25Index

    # Initialize components
    document_processor = DocumentProcessor(
        chunk_size=512,
        chunk_overlap=50,
        extract_images=enable_colpali
    )

    dense_embedder = get_embedder("sentence-transformers", model_name=dense_model)

    colpali_embedder = None
    if enable_colpali:
        try:
            from src.embeddings.colpali_embedder import ColPaliEmbedder
            colpali_embedder = ColPaliEmbedder()
        except Exception as e:
            logger.warning(f"ColPali not available: {e}")

    # Initialize stores
    metadata_store = MetadataStore(postgres_url)
    metadata_store.create_tables()

    dense_vector_store = QdrantVectorStore(
        collection_name="documents_enhanced",
        dimension=dense_embedder.dimension,
        host=qdrant_host,
        port=qdrant_port
    )

    colpali_vector_store = None
    if colpali_embedder:
        colpali_vector_store = QdrantMultiVectorStore(
            collection_name="documents_colpali",
            dimension=128,
            host=qdrant_host,
            port=qdrant_port
        )

    bm25_store = BM25Index(persist_path="./data/bm25_enhanced.pkl")

    # Create document graph
    document_graph = DocumentGraph()

    return EnhancedIngestionPipeline(
        document_processor=document_processor,
        dense_embedder=dense_embedder,
        colpali_embedder=colpali_embedder,
        metadata_store=metadata_store,
        dense_vector_store=dense_vector_store,
        colpali_vector_store=colpali_vector_store,
        bm25_store=bm25_store,
        document_graph=document_graph,
        enable_tables=enable_tables,
        enable_graph=enable_graph,
        enable_colpali=enable_colpali
    )


# Example usage
if __name__ == "__main__":
    print("Enhanced Ingestion Pipeline")
    print("Features:")
    print("  - Quality-based routing")
    print("  - Hierarchical chunking")
    print("  - Domain-specific metadata")
    print("  - Table extraction")
    print("  - Relationship graph")
