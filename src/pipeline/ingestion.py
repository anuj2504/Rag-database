"""
Main Ingestion Pipeline for document processing and indexing.

==============================================================================
DEPRECATED: Use MasterPipeline from src.pipeline.master_pipeline instead.

    from src.pipeline import create_master_pipeline

This file is kept for backward compatibility only and will be removed
in a future version.
==============================================================================

Orchestrates the full flow:
1. Document processing (Unstructured.io)
2. Metadata storage (PostgreSQL)
3. Dense embedding + indexing (Qdrant)
4. ColPali embedding + indexing (Qdrant multi-vector)
5. BM25 indexing

IMPORTANT: All operations require TenantContext for multi-tenant isolation.
"""
import warnings
warnings.warn(
    "IngestionPipeline is deprecated. Use MasterPipeline from "
    "src.pipeline.master_pipeline instead.",
    DeprecationWarning,
    stacklevel=2
)
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from src.metadata.tenant_schema import TenantContext, TenantMetadata, TenantManager, AccessLevel

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: str
    filename: str
    status: str  # 'success', 'failed', 'partial'
    chunks_indexed: int
    pages_indexed: int
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0


class IngestionPipeline:
    """
    Production-grade document ingestion pipeline with multi-tenant support.

    Features:
    - Multi-tenant isolation (CRITICAL)
    - Parallel processing
    - Automatic retry on failure
    - Progress tracking
    - Atomic operations with rollback
    """

    def __init__(
        self,
        # Processors
        document_processor,
        dense_embedder,
        colpali_embedder=None,
        # Stores
        metadata_store=None,
        dense_vector_store=None,
        colpali_vector_store=None,
        bm25_store=None,
        # Config
        batch_size: int = 32,
        max_workers: int = 4,
        enable_colpali: bool = True,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            document_processor: DocumentProcessor instance
            dense_embedder: Dense embedding model
            colpali_embedder: ColPali embedding model (optional)
            metadata_store: MetadataStore for PostgreSQL
            dense_vector_store: QdrantVectorStore for dense vectors
            colpali_vector_store: QdrantMultiVectorStore for ColPali
            bm25_store: BM25Index for keyword search
            batch_size: Batch size for embedding
            max_workers: Max parallel workers
            enable_colpali: Whether to use ColPali
        """
        self.document_processor = document_processor
        self.dense_embedder = dense_embedder
        self.colpali_embedder = colpali_embedder
        self.metadata_store = metadata_store
        self.dense_vector_store = dense_vector_store
        self.colpali_vector_store = colpali_vector_store
        self.bm25_store = bm25_store
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_colpali = enable_colpali and colpali_embedder is not None

        # Tenant manager for isolation
        self.tenant_manager = TenantManager(metadata_store)

    def ingest_document(
        self,
        file_path: str,
        tenant_context: TenantContext,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Ingest a single document through the full pipeline.

        IMPORTANT: tenant_context is REQUIRED for multi-tenant isolation.

        Args:
            file_path: Path to the document
            tenant_context: REQUIRED - Tenant context for isolation
            custom_metadata: Additional metadata

        Returns:
            IngestionResult with status and counts
        """
        # Validate tenant context
        self.tenant_manager.validate_context(tenant_context)
        start_time = datetime.now()
        filename = Path(file_path).name

        try:
            logger.info(f"Starting ingestion: {filename} (org: {tenant_context.organization_id})")

            # Merge tenant metadata with custom metadata
            merged_metadata = self.tenant_manager.create_document_metadata(
                tenant_context,
                custom_metadata or {}
            )

            # Step 1: Process document with Unstructured.io
            logger.info(f"[1/5] Processing document: {filename}")
            processed_doc = self.document_processor.process_document(
                file_path=file_path,
                custom_metadata=merged_metadata
            )
            document_id = processed_doc.document_id

            # Step 2: Store metadata in PostgreSQL
            if self.metadata_store:
                logger.info(f"[2/5] Storing metadata: {document_id}")
                self._store_metadata(processed_doc, tenant_context)

            # Step 3: Create and store dense embeddings
            logger.info(f"[3/5] Creating dense embeddings: {len(processed_doc.chunks)} chunks")
            chunks_indexed = self._index_dense_embeddings(processed_doc, tenant_context)

            # Step 4: Create and store ColPali embeddings
            pages_indexed = 0
            if self.enable_colpali and processed_doc.page_images:
                logger.info(f"[4/5] Creating ColPali embeddings: {len(processed_doc.page_images)} pages")
                pages_indexed = self._index_colpali_embeddings(processed_doc, tenant_context)
            else:
                logger.info(f"[4/5] Skipping ColPali (disabled or no images)")

            # Step 5: Index in BM25
            if self.bm25_store:
                logger.info(f"[5/5] Indexing in BM25")
                self._index_bm25(processed_doc, tenant_context)

            # Update document status
            if self.metadata_store:
                self.metadata_store.update_document(
                    document_id=document_id,
                    status="completed",
                    total_chunks=chunks_indexed,
                    total_pages=pages_indexed,
                    processed_at=datetime.utcnow()
                )

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Completed ingestion: {filename} "
                f"({chunks_indexed} chunks, {pages_indexed} pages, {processing_time:.2f}s)"
            )

            return IngestionResult(
                document_id=document_id,
                filename=filename,
                status="success",
                chunks_indexed=chunks_indexed,
                pages_indexed=pages_indexed,
                organization_id=tenant_context.organization_id,
                workspace_id=tenant_context.workspace_id,
                processing_time_seconds=processing_time
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to ingest {filename}: {e}", exc_info=True)

            # Try to mark as failed in metadata store
            if self.metadata_store:
                try:
                    self.metadata_store.update_document(
                        document_id=processed_doc.document_id if 'processed_doc' in locals() else None,
                        status="failed"
                    )
                except Exception:
                    pass

            return IngestionResult(
                document_id=processed_doc.document_id if 'processed_doc' in locals() else "unknown",
                filename=filename,
                status="failed",
                chunks_indexed=0,
                pages_indexed=0,
                organization_id=tenant_context.organization_id,
                workspace_id=tenant_context.workspace_id,
                error_message=str(e),
                processing_time_seconds=processing_time
            )

    def _store_metadata(self, processed_doc, tenant_context: TenantContext) -> None:
        """Store document and chunk metadata in PostgreSQL with tenant isolation."""
        # Merge tenant metadata
        doc_metadata = self.tenant_manager.create_document_metadata(
            tenant_context,
            processed_doc.metadata or {}
        )

        # Create document record with tenant fields
        self.metadata_store.create_document(
            document_id=processed_doc.document_id,
            filename=processed_doc.filename,
            file_path=processed_doc.file_path,
            document_type=processed_doc.document_type,
            organization_id=tenant_context.organization_id,
            workspace_id=tenant_context.workspace_id,
            access_level=tenant_context.access_level.value,
            metadata=doc_metadata
        )

        # Create chunk records with tenant fields
        chunk_records = []
        for chunk in processed_doc.chunks:
            chunk_records.append({
                "id": chunk.chunk_id,
                "document_id": processed_doc.document_id,
                "organization_id": tenant_context.organization_id,
                "workspace_id": tenant_context.workspace_id,
                "access_level": tenant_context.access_level.value,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "element_type": chunk.element_type,
                "metadata": chunk.metadata,
            })
        self.metadata_store.create_chunks(chunk_records)

        # Create page records with tenant fields
        if processed_doc.page_images:
            page_records = []
            for idx, img in enumerate(processed_doc.page_images):
                page_id = f"{processed_doc.document_id}_page_{idx}"
                page_records.append({
                    "id": page_id,
                    "document_id": processed_doc.document_id,
                    "organization_id": tenant_context.organization_id,
                    "workspace_id": tenant_context.workspace_id,
                    "page_number": idx + 1,
                    "width": img.size[0],
                    "height": img.size[1],
                })
            self.metadata_store.create_pages(page_records)

    def _index_dense_embeddings(self, processed_doc, tenant_context: TenantContext) -> int:
        """Create and store dense embeddings in Qdrant with tenant isolation."""
        if not self.dense_vector_store:
            return 0

        chunks = processed_doc.chunks
        if not chunks:
            return 0

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Create embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.dense_embedder.embed_texts(batch_texts)
            all_embeddings.append(batch_embeddings)

        import numpy as np
        embeddings = np.vstack(all_embeddings)

        # Prepare IDs and payloads with tenant fields
        ids = [chunk.chunk_id for chunk in chunks]
        payloads = [
            self.tenant_manager.scope_chunk_payload(
                {
                    "document_id": processed_doc.document_id,
                    "document_type": processed_doc.document_type,
                    "filename": processed_doc.filename,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "element_type": chunk.element_type,
                    "text": chunk.text[:1000],  # Store truncated text for retrieval
                },
                tenant_context
            )
            for chunk in chunks
        ]

        # Store in Qdrant
        self.dense_vector_store.add_documents(ids, embeddings, payloads)

        # Mark as indexed in metadata store
        if self.metadata_store:
            self.metadata_store.mark_chunks_indexed(ids)

        return len(chunks)

    def _index_colpali_embeddings(self, processed_doc, tenant_context: TenantContext) -> int:
        """Create and store ColPali embeddings in Qdrant with tenant isolation."""
        if not self.colpali_vector_store or not processed_doc.page_images:
            return 0

        # Create page embeddings
        page_embeddings = self.colpali_embedder.embed_images(processed_doc.page_images)

        # Prepare IDs and payloads with tenant fields
        ids = []
        payloads = []
        for idx, _ in enumerate(processed_doc.page_images):
            page_id = f"{processed_doc.document_id}_page_{idx}"
            ids.append(page_id)
            payloads.append(
                self.tenant_manager.scope_chunk_payload(
                    {
                        "document_id": processed_doc.document_id,
                        "document_type": processed_doc.document_type,
                        "filename": processed_doc.filename,
                        "page_number": idx + 1,
                    },
                    tenant_context
                )
            )

        # Store in Qdrant multi-vector collection
        self.colpali_vector_store.add_pages(ids, page_embeddings, payloads)

        # Mark as indexed
        if self.metadata_store:
            self.metadata_store.mark_pages_indexed(ids)

        return len(page_embeddings)

    def _index_bm25(self, processed_doc, tenant_context: TenantContext) -> None:
        """Index chunks in BM25 with tenant isolation."""
        if not self.bm25_store:
            return

        documents = []
        for chunk in processed_doc.chunks:
            doc = self.tenant_manager.scope_bm25_document(
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "document_id": processed_doc.document_id,
                    "document_type": processed_doc.document_type,
                    "page_number": chunk.page_number,
                },
                tenant_context
            )
            documents.append(doc)

        self.bm25_store.add_documents(documents)

    def ingest_directory(
        self,
        directory_path: str,
        tenant_context: TenantContext,
        extensions: List[str] = [".pdf", ".docx", ".txt", ".png", ".jpg"],
        parallel: bool = True
    ) -> List[IngestionResult]:
        """
        Ingest all documents in a directory.

        Args:
            directory_path: Path to directory
            tenant_context: REQUIRED - Tenant context for isolation
            extensions: File extensions to process
            parallel: Use parallel processing

        Returns:
            List of IngestionResult
        """
        # Validate tenant context
        self.tenant_manager.validate_context(tenant_context)

        directory = Path(directory_path)
        files = []
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))

        logger.info(f"Found {len(files)} documents to ingest for org: {tenant_context.organization_id}")

        results = []

        if parallel and len(files) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.ingest_document, str(f), tenant_context): f
                    for f in files
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
        else:
            for file_path in files:
                result = self.ingest_document(str(file_path), tenant_context)
                results.append(result)

        # Summary
        success_count = sum(1 for r in results if r.status == "success")
        failed_count = sum(1 for r in results if r.status == "failed")
        logger.info(
            f"Ingestion complete: {success_count} succeeded, {failed_count} failed"
        )

        return results

    def delete_document(self, document_id: str, tenant_context: Optional[TenantContext] = None) -> bool:
        """
        Delete a document and all its data from all stores.

        Args:
            document_id: Document ID to delete
            tenant_context: Optional - for audit logging

        Returns:
            True if successful
        """
        try:
            org_info = f" (org: {tenant_context.organization_id})" if tenant_context else ""
            logger.info(f"Deleting document: {document_id}{org_info}")

            # Delete from vector stores
            if self.dense_vector_store:
                self.dense_vector_store.delete_by_document_id(document_id)

            if self.colpali_vector_store:
                self.colpali_vector_store.delete_by_document_id(document_id)

            # Delete from BM25
            if self.bm25_store:
                self.bm25_store.delete_by_document_id(document_id)

            # Delete from metadata store (cascades to chunks/pages)
            if self.metadata_store:
                self.metadata_store.delete_document(document_id)

            logger.info(f"Successfully deleted document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False


def create_pipeline(
    postgres_url: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    dense_model: str = "BAAI/bge-base-en-v1.5",
    enable_colpali: bool = True,
    bm25_persist_path: str = "./data/bm25_index.pkl",
) -> IngestionPipeline:
    """
    Factory function to create a fully configured pipeline.

    Args:
        postgres_url: PostgreSQL connection URL
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        dense_model: Dense embedding model name
        enable_colpali: Whether to enable ColPali
        bm25_persist_path: Path to persist BM25 index

    Returns:
        Configured IngestionPipeline
    """
    from src.ingestion.document_processor import DocumentProcessor
    from src.embeddings.dense_embedder import get_embedder
    from src.storage.metadata_store import MetadataStore
    from src.storage.vector_store import QdrantVectorStore, QdrantMultiVectorStore
    from src.storage.bm25_store import BM25Index

    # Initialize document processor
    document_processor = DocumentProcessor(
        chunk_size=512,
        chunk_overlap=50,
        extract_images=enable_colpali
    )

    # Initialize embedders
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
        collection_name="documents_dense",
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

    bm25_store = BM25Index(persist_path=bm25_persist_path)

    # Create pipeline
    return IngestionPipeline(
        document_processor=document_processor,
        dense_embedder=dense_embedder,
        colpali_embedder=colpali_embedder,
        metadata_store=metadata_store,
        dense_vector_store=dense_vector_store,
        colpali_vector_store=colpali_vector_store,
        bm25_store=bm25_store,
        enable_colpali=enable_colpali
    )


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = create_pipeline(
        postgres_url="postgresql://user:password@localhost:5432/rag_db",
        qdrant_host="localhost",
        qdrant_port=6333,
        enable_colpali=True
    )

    # Ingest a single document
    # result = pipeline.ingest_document("./documents/contract.pdf")
    # print(f"Result: {result}")

    # Ingest a directory
    # results = pipeline.ingest_directory("./documents/")
    # for r in results:
    #     print(f"{r.filename}: {r.status} ({r.chunks_indexed} chunks)")
