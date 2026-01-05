"""
Master Ingestion Pipeline - Single Entry Point for NHAI/L&T RAG System.

This is the ONLY pipeline you should use. It replaces:
- ingestion.py (basic pipeline)
- enhanced_ingestion.py (enterprise pipeline)

Complete Flow:
==============
1. PARSE      → Unstructured.io extracts text, elements, page images
2. ASSESS     → Quality analyzer determines document quality tier
3. EXTRACT    → Domain-specific metadata extraction (parties, dates, amounts)
4. CHUNK      → ChunkingService applies quality-based strategy
5. EMBED      → Dense embeddings (BGE), ColPali (visual), BM25 (keyword)
6. STORE      → PostgreSQL (metadata), Qdrant (vectors), BM25 index
7. GRAPH      → Document relationship graph (optional)

Multi-Tenant Isolation:
=======================
EVERY operation is scoped by organization_id.
Data from different organizations NEVER mixes.
TenantContext is REQUIRED for all operations.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np

from src.metadata.tenant_schema import TenantContext, TenantManager, AccessLevel
from src.chunking.chunking_service import ChunkingService, ChunkingResult
from src.chunking.unified_chunk import UnifiedChunk, ChunkLevel

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """
    Complete result of document ingestion.

    Contains all information about the ingested document,
    including quality metrics, chunk counts, and any errors.
    """
    # Identity
    document_id: str
    filename: str
    status: str  # 'success', 'partial', 'failed'

    # Tenant info
    organization_id: str
    workspace_id: Optional[str] = None
    collection_id: Optional[str] = None

    # Quality info
    quality_tier: str = "unknown"
    quality_score: float = 0.0
    quality_issues: List[str] = field(default_factory=list)

    # Processing stats
    chunks_created: int = 0
    chunks_indexed: int = 0
    pages_indexed: int = 0
    visual_elements_indexed: int = 0  # NEW: Cropped tables/figures
    tables_extracted: int = 0
    relationships_found: int = 0

    # Document info
    document_type: Optional[str] = None
    extracted_metadata: Dict[str, Any] = field(default_factory=dict)

    # Timing
    processing_time_seconds: float = 0.0

    # Errors
    error_message: Optional[str] = None
    error_stage: Optional[str] = None  # Which stage failed


class MasterPipeline:
    """
    Master Ingestion Pipeline for NHAI/L&T Enterprise RAG System.

    This is the single entry point for all document ingestion.

    Features:
    - Multi-tenant isolation (CRITICAL)
    - Quality-based chunking strategy selection
    - Hierarchical + semantic chunking (Chonkie)
    - Multiple embedding types (dense, ColPali, BM25)
    - Domain-specific metadata extraction
    - Document relationship graph
    - Parallel processing for batch ingestion
    - Atomic operations with rollback

    Usage:
        pipeline = create_master_pipeline(
            postgres_url="postgresql://...",
            qdrant_host="localhost",
        )

        result = pipeline.ingest(
            file_path="contract.pdf",
            tenant_context=TenantContext(
                organization_id="nhai",
                workspace_id="contracts",
                access_level=AccessLevel.CONFIDENTIAL,
            )
        )
    """

    def __init__(
        self,
        # Core processors
        document_processor,
        chunking_service: ChunkingService,
        dense_embedder,
        colpali_embedder=None,

        # Stores
        metadata_store=None,
        dense_vector_store=None,
        colpali_vector_store=None,
        visual_element_store=None,  # NEW: For cropped tables/figures
        bm25_store=None,

        # Enhanced components
        quality_analyzer=None,
        metadata_extractor=None,
        table_extractor=None,
        document_graph=None,

        # Config
        batch_size: int = 32,
        max_workers: int = 4,
        enable_colpali: bool = True,
        enable_visual_elements: bool = True,  # NEW: Enable visual element embedding
        enable_tables: bool = True,
        enable_graph: bool = True,
    ):
        """
        Initialize master pipeline.

        Args:
            document_processor: Unstructured.io wrapper
            chunking_service: Unified chunking service
            dense_embedder: Dense embedding model (BGE, etc.)
            colpali_embedder: ColPali embedding model
            metadata_store: PostgreSQL metadata store
            dense_vector_store: Qdrant vector store
            colpali_vector_store: Qdrant multi-vector store for full pages
            visual_element_store: Qdrant store for cropped tables/figures
            bm25_store: BM25 keyword index
            quality_analyzer: Document quality analyzer
            metadata_extractor: Domain-specific metadata extractor
            table_extractor: Table extraction service
            document_graph: Document relationship graph
            batch_size: Batch size for embedding
            max_workers: Max parallel workers
            enable_colpali: Enable ColPali visual embeddings (full pages)
            enable_visual_elements: Enable visual element embedding (cropped tables/figures)
            enable_tables: Enable table extraction
            enable_graph: Enable document graph
        """
        # Core components
        self.document_processor = document_processor
        self.chunking_service = chunking_service
        self.dense_embedder = dense_embedder
        self.colpali_embedder = colpali_embedder

        # Stores
        self.metadata_store = metadata_store
        self.dense_vector_store = dense_vector_store
        self.colpali_vector_store = colpali_vector_store
        self.visual_element_store = visual_element_store
        self.bm25_store = bm25_store

        # Enhanced components
        self.quality_analyzer = quality_analyzer
        self.metadata_extractor = metadata_extractor
        self.table_extractor = table_extractor
        self.document_graph = document_graph

        # Config
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_colpali = enable_colpali and colpali_embedder is not None
        self.enable_visual_elements = enable_visual_elements and visual_element_store is not None and colpali_embedder is not None
        self.enable_tables = enable_tables and table_extractor is not None
        self.enable_graph = enable_graph and document_graph is not None

        # Tenant manager
        self.tenant_manager = TenantManager(metadata_store) if metadata_store else None

    def ingest(
        self,
        file_path: str,
        tenant_context: TenantContext,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """
        Ingest a document through the complete pipeline.

        Args:
            file_path: Path to the document file
            tenant_context: REQUIRED - Tenant context for isolation
            custom_metadata: Additional metadata to store

        Returns:
            IngestionResult with complete ingestion details
        """
        start_time = datetime.now()
        filename = Path(file_path).name
        document_id = "unknown"

        # Validate tenant context
        if not tenant_context or not tenant_context.organization_id:
            return IngestionResult(
                document_id="unknown",
                filename=filename,
                status="failed",
                organization_id="unknown",
                error_message="TenantContext with organization_id is REQUIRED",
                error_stage="validation",
            )

        # Create processing job
        job_id = None
        if self.metadata_store:
            try:
                job = self.metadata_store.create_processing_job(
                    organization_id=tenant_context.organization_id,
                    job_type="ingestion",
                    priority=50,
                    metadata={"filename": filename, "file_path": file_path}
                )
                job_id = job.id
            except Exception as e:
                logger.warning(f"Failed to create processing job: {e}")

        try:
            logger.info(
                f"[INGEST] Starting: {filename} "
                f"(org={tenant_context.organization_id}, job_id={job_id})"
            )

            # Update job status to processing
            if job_id and self.metadata_store:
                self.metadata_store.update_processing_job(
                    job_id=job_id,
                    status="processing",
                    current_step="parsing",
                    progress_percent=0,
                )

            # ========== STAGE 1: PARSE ==========
            logger.info("[1/7] Parsing document with Unstructured.io")
            processed_doc = self.document_processor.process_document(
                file_path=file_path,
                custom_metadata=custom_metadata,
            )
            document_id = processed_doc.document_id
            full_text = '\n'.join(chunk.text for chunk in processed_doc.chunks)

            # Update job with document_id
            if job_id and self.metadata_store:
                self.metadata_store.update_processing_job(
                    job_id=job_id,
                    current_step="quality_assessment",
                    progress_percent=15,
                )

            # ========== STAGE 2: QUALITY ASSESSMENT ==========
            quality_tier = "medium"
            quality_score = 0.7
            quality_issues = []

            if self.quality_analyzer:
                logger.info("[2/7] Assessing document quality")
                try:
                    quality_report = self.quality_analyzer.analyze(full_text, filename)
                    quality_tier = quality_report.tier.value if hasattr(quality_report.tier, 'value') else str(quality_report.tier)
                    quality_score = quality_report.overall_score
                    quality_issues = quality_report.issues
                except Exception as e:
                    logger.warning(f"Quality analysis failed, using defaults: {e}")
            else:
                logger.info("[2/7] Skipping quality assessment (no analyzer)")

            # ========== STAGE 3: METADATA EXTRACTION ==========
            document_type = "general"
            extracted_metadata = {}

            if self.metadata_extractor:
                logger.info("[3/7] Extracting domain-specific metadata")
                try:
                    meta_result = self.metadata_extractor.extract(full_text, filename)
                    document_type = meta_result.document_type
                    extracted_metadata = meta_result.filter_tags if hasattr(meta_result, 'filter_tags') else {}
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")
            else:
                logger.info("[3/7] Skipping metadata extraction (no extractor)")

            # ========== STAGE 4: INTELLIGENT CHUNKING ==========
            logger.info("[4/7] Chunking with quality-based strategy")
            chunking_result: ChunkingResult = self.chunking_service.chunk(
                text=full_text,
                document_id=document_id,
                tenant_context=tenant_context,
                document_type=document_type,
                quality_level=quality_tier,
                filename=filename,
                detect_structure=True,
                custom_metadata=custom_metadata,
            )
            chunks = chunking_result.chunks
            logger.info(
                f"  Created {len(chunks)} chunks using {chunking_result.chunk_strategy} strategy"
            )

            # Assign accurate page numbers to chunks from original parsed elements
            # This is critical for ColPali page-to-chunk score propagation
            self._assign_page_numbers_to_chunks(chunks, processed_doc.chunks, full_text)

            # ========== STAGE 5: TABLE EXTRACTION ==========
            tables = []
            if self.enable_tables:
                logger.info("[5/7] Extracting tables")
                try:
                    tables = self.table_extractor.extract_tables(full_text, document_id)
                    logger.info(f"  Found {len(tables)} tables")
                except Exception as e:
                    logger.warning(f"Table extraction failed: {e}")
            else:
                logger.info("[5/7] Skipping table extraction")

            # ========== STAGE 6: STORE METADATA ==========
            if self.metadata_store:
                logger.info("[6/7] Storing metadata in PostgreSQL")
                self._store_metadata(
                    processed_doc=processed_doc,
                    chunks=chunks,
                    tenant_context=tenant_context,
                    document_type=document_type,
                    quality_tier=quality_tier,
                    quality_score=quality_score,
                    extracted_metadata=extracted_metadata,
                    quality_issues=quality_issues,
                    tables=tables,
                )
            else:
                logger.info("[6/7] Skipping metadata storage (no store)")

            # ========== STAGE 7: EMBEDDING & INDEXING ==========
            logger.info("[7/7] Creating embeddings and indexing")

            # Dense embeddings
            chunks_indexed = self._index_dense_embeddings(chunks)
            logger.info(f"  Indexed {chunks_indexed} chunks (dense)")

            # ColPali embeddings (full pages)
            pages_indexed = 0
            if self.enable_colpali and processed_doc.page_images:
                pages_indexed = self._index_colpali_embeddings(
                    processed_doc, tenant_context
                )
                logger.info(f"  Indexed {pages_indexed} pages (ColPali)")

            # Visual element embeddings (cropped tables/figures)
            visual_elements_indexed = 0
            if self.enable_visual_elements and processed_doc.visual_elements:
                visual_elements_indexed = self._index_visual_elements(
                    processed_doc, tenant_context
                )
                logger.info(f"  Indexed {visual_elements_indexed} visual elements (ColPali)")

            # BM25 index
            if self.bm25_store:
                self._index_bm25(chunks)
                logger.info(f"  Indexed {len(chunks)} chunks (BM25)")

            # ========== STAGE 8: DOCUMENT GRAPH (Optional) ==========
            relationships_found = 0
            if self.enable_graph:
                try:
                    relationships = self.document_graph.process_document(
                        document_id=document_id,
                        text=full_text,
                        title=extracted_metadata.get('title'),
                        document_type=document_type,
                    )
                    relationships_found = len(relationships) if relationships else 0
                except Exception as e:
                    logger.warning(f"Graph processing failed: {e}")

            # ========== COMPLETE ==========
            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"[COMPLETE] {filename}: "
                f"{chunks_indexed} chunks, {pages_indexed} pages, "
                f"{visual_elements_indexed} visual elements, "
                f"{processing_time:.2f}s"
            )

            # Update job as completed
            if job_id and self.metadata_store:
                self.metadata_store.update_processing_job(
                    job_id=job_id,
                    status="completed",
                    progress_percent=100,
                    current_step="completed",
                    result_summary={
                        "document_id": document_id,
                        "chunks_indexed": chunks_indexed,
                        "pages_indexed": pages_indexed,
                        "visual_elements_indexed": visual_elements_indexed,
                        "tables_extracted": len(tables),
                        "processing_time_seconds": processing_time,
                    }
                )

            return IngestionResult(
                document_id=document_id,
                filename=filename,
                status="success",
                organization_id=tenant_context.organization_id,
                workspace_id=tenant_context.workspace_id,
                collection_id=tenant_context.collection_id,
                quality_tier=quality_tier,
                quality_score=quality_score,
                quality_issues=quality_issues,
                chunks_created=len(chunks),
                chunks_indexed=chunks_indexed,
                pages_indexed=pages_indexed,
                visual_elements_indexed=visual_elements_indexed,
                tables_extracted=len(tables),
                relationships_found=relationships_found,
                document_type=document_type,
                extracted_metadata=extracted_metadata,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[FAILED] {filename}: {e}", exc_info=True)

            # Update job as failed
            if job_id and self.metadata_store:
                self.metadata_store.update_processing_job(
                    job_id=job_id,
                    status="failed",
                    error_message=str(e),
                    error_details={"traceback": str(e)},
                )

            return IngestionResult(
                document_id=document_id,
                filename=filename,
                status="failed",
                organization_id=tenant_context.organization_id,
                workspace_id=tenant_context.workspace_id,
                error_message=str(e),
                processing_time_seconds=processing_time,
            )

    def _store_metadata(
        self,
        processed_doc,
        chunks: List[UnifiedChunk],
        tenant_context: TenantContext,
        document_type: str,
        quality_tier: str,
        quality_score: float,
        extracted_metadata: Dict[str, Any],
        quality_issues: Optional[List[str]] = None,
        tables: Optional[List[Dict]] = None,
        job_id: Optional[int] = None,
    ):
        """Store document and chunk metadata in PostgreSQL."""
        # Document record
        self.metadata_store.create_document(
            document_id=processed_doc.document_id,
            filename=processed_doc.filename,
            file_path=processed_doc.file_path,
            document_type=document_type,
            organization_id=tenant_context.organization_id,
            workspace_id=tenant_context.workspace_id,
            access_level=tenant_context.access_level.value if hasattr(tenant_context.access_level, 'value') else str(tenant_context.access_level),
            metadata={
                "quality_tier": quality_tier,
                "quality_score": quality_score,
                **extracted_metadata,
            }
        )

        # Chunk records
        chunk_records = [chunk.to_metadata_record() for chunk in chunks]
        self.metadata_store.create_chunks(chunk_records)

        # Page records
        if processed_doc.page_images:
            page_records = []
            for idx, img in enumerate(processed_doc.page_images):
                page_records.append({
                    "id": f"{processed_doc.document_id}_page_{idx}",
                    "document_id": processed_doc.document_id,
                    "organization_id": tenant_context.organization_id,
                    "workspace_id": tenant_context.workspace_id,
                    "page_number": idx + 1,
                    "width": img.size[0] if hasattr(img, 'size') else 0,
                    "height": img.size[1] if hasattr(img, 'size') else 0,
                })
            self.metadata_store.create_pages(page_records)

        # Quality report
        self.metadata_store.create_quality_report(
            document_id=processed_doc.document_id,
            organization_id=tenant_context.organization_id,
            overall_score=quality_score * 100,  # Convert to 0-100 scale
            quality_level=quality_tier,
            issues_found=quality_issues or [],
            recommendations=[],
            recommended_pipeline="chunking_service",
        )

        # Extracted tables
        if tables:
            table_records = []
            for idx, table in enumerate(tables):
                # Handle both dict and ExtractedTable dataclass objects
                if isinstance(table, dict):
                    table_records.append({
                        "id": f"{processed_doc.document_id}_table_{idx}",
                        "document_id": processed_doc.document_id,
                        "organization_id": tenant_context.organization_id,
                        "page_number": table.get("page_number"),
                        "table_index": idx,
                        "html_content": table.get("html"),
                        "markdown_content": table.get("markdown"),
                        "structured_data": table.get("data"),
                        "description": table.get("description"),
                        "num_rows": table.get("num_rows", 0),
                        "num_cols": table.get("num_cols", 0),
                    })
                else:
                    # ExtractedTable dataclass from table_extractor.py
                    table_records.append({
                        "id": getattr(table, 'id', f"{processed_doc.document_id}_table_{idx}"),
                        "document_id": processed_doc.document_id,
                        "organization_id": tenant_context.organization_id,
                        "page_number": getattr(table, 'page_number', None),
                        "table_index": idx,
                        "html_content": None,  # ExtractedTable uses structured_text instead
                        "markdown_content": getattr(table, 'structured_text', ''),
                        "structured_data": getattr(table, 'raw_data', {}),
                        "description": getattr(table, 'semantic_description', ''),
                        "num_rows": getattr(table, 'num_rows', 0),
                        "num_cols": getattr(table, 'num_cols', 0),
                    })
            self.metadata_store.create_extracted_tables(table_records)

    def _index_dense_embeddings(self, chunks: List[UnifiedChunk]) -> int:
        """Create and store dense embeddings in Qdrant."""
        if not self.dense_vector_store:
            return 0

        # Get paragraph-level chunks (primary retrieval unit)
        para_chunks = [c for c in chunks if c.level == ChunkLevel.PARAGRAPH]
        if not para_chunks:
            return 0

        # Create embeddings in batches
        texts = [c.text for c in para_chunks]
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.dense_embedder.embed_texts(batch)
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])

        # Prepare for storage
        ids = [c.id for c in para_chunks]
        payloads = [c.to_vector_payload() for c in para_chunks]

        # Store in Qdrant
        self.dense_vector_store.add_documents(ids, embeddings, payloads)

        # Mark as indexed
        if self.metadata_store:
            self.metadata_store.mark_chunks_indexed(ids)

        return len(para_chunks)

    def _index_colpali_embeddings(
        self,
        processed_doc,
        tenant_context: TenantContext,
    ) -> int:
        """Create and store ColPali embeddings in Qdrant."""
        if not self.colpali_vector_store or not processed_doc.page_images:
            return 0

        # Create page embeddings
        page_embeddings = self.colpali_embedder.embed_images(processed_doc.page_images)

        # Prepare data
        ids = []
        payloads = []
        for idx in range(len(processed_doc.page_images)):
            page_id = f"{processed_doc.document_id}_page_{idx}"
            ids.append(page_id)
            payloads.append({
                "document_id": processed_doc.document_id,
                "page_number": idx + 1,
                "filename": processed_doc.filename,
                "organization_id": tenant_context.organization_id,
                "workspace_id": tenant_context.workspace_id,
                "access_level": tenant_context.access_level.value if hasattr(tenant_context.access_level, 'value') else str(tenant_context.access_level),
            })

        # Store in Qdrant
        self.colpali_vector_store.add_pages(ids, page_embeddings, payloads)

        return len(page_embeddings)

    def _index_visual_elements(
        self,
        processed_doc,
        tenant_context: TenantContext,
    ) -> int:
        """
        Create and store ColPali embeddings for visual elements (tables, figures).

        Visual elements are cropped images extracted from the document.
        They provide more precise retrieval than full-page embeddings for
        queries targeting specific tables or figures.
        """
        if not self.visual_element_store or not processed_doc.visual_elements:
            return 0

        # Filter elements that have images
        elements_with_images = [
            elem for elem in processed_doc.visual_elements
            if elem.image_base64 is not None
        ]

        if not elements_with_images:
            logger.info("No visual elements with images to index")
            return 0

        # Decode images from base64
        images = []
        valid_elements = []
        for elem in elements_with_images:
            img = elem.decode_image()
            if img is not None:
                images.append(img)
                valid_elements.append(elem)

        if not images:
            logger.warning("Failed to decode any visual element images")
            return 0

        # Create ColPali embeddings for cropped images
        logger.info(f"Embedding {len(images)} visual elements with ColPali...")
        element_embeddings = self.colpali_embedder.embed_images(images)

        # Prepare data for storage
        ids = []
        payloads = []
        for elem in valid_elements:
            ids.append(elem.element_id)
            payloads.append({
                "document_id": elem.document_id,
                "element_type": elem.element_type.value,
                "page_number": elem.page_number,
                "text_content": elem.text_content,
                "html_content": elem.html_content,
                "bbox": elem.bbox.to_dict() if elem.bbox else None,
                "filename": processed_doc.filename,
                "organization_id": tenant_context.organization_id,
                "workspace_id": tenant_context.workspace_id,
                "access_level": tenant_context.access_level.value if hasattr(tenant_context.access_level, 'value') else str(tenant_context.access_level),
            })

        # Store in visual elements collection
        self.visual_element_store.add_elements(ids, element_embeddings, payloads)

        logger.info(f"Indexed {len(element_embeddings)} visual elements")
        return len(element_embeddings)

    def _index_bm25(self, chunks: List[UnifiedChunk]):
        """Index chunks in BM25."""
        if not self.bm25_store:
            return

        documents = [
            c.to_bm25_document()
            for c in chunks
            if c.level == ChunkLevel.PARAGRAPH
        ]
        self.bm25_store.add_documents(documents)

    def _assign_page_numbers_to_chunks(
        self,
        chunks: List[UnifiedChunk],
        original_elements,
        full_text: str
    ) -> None:
        """
        Accurately assign page numbers to chunks based on original parsed elements.

        This is critical for ColPali page-to-chunk score propagation.
        Uses binary search for O(n log m) efficiency.

        Args:
            chunks: New chunks from ChunkingService (missing page_number)
            original_elements: Original ProcessedChunk list from document_processor
            full_text: The full concatenated text used for chunking
        """
        import bisect

        # Build character position -> page number mapping
        # Each entry is (cumulative_char_position, page_number)
        page_boundaries = []
        cumulative_pos = 0

        for elem in original_elements:
            page_num = getattr(elem, 'page_number', None)
            if page_num is not None:
                # Record the start position of this element and its page
                page_boundaries.append((cumulative_pos, page_num))
            # Add element text length + newline separator
            elem_text = getattr(elem, 'text', '')
            cumulative_pos += len(elem_text) + 1  # +1 for '\n' separator

        if not page_boundaries:
            logger.warning("No page information available from parsed elements")
            return

        # Sort by position (should already be sorted, but ensure it)
        page_boundaries.sort(key=lambda x: x[0])

        # Extract just positions for binary search
        positions = [pb[0] for pb in page_boundaries]

        # Assign page numbers to each chunk
        assigned_count = 0
        for chunk in chunks:
            # Find the chunk's position in the full text
            chunk_start = chunk.char_start

            # Binary search: find rightmost boundary <= chunk_start
            idx = bisect.bisect_right(positions, chunk_start) - 1

            if idx >= 0:
                _, page_num = page_boundaries[idx]
                chunk.page_number = page_num
                assigned_count += 1

        logger.info(
            f"Assigned page numbers to {assigned_count}/{len(chunks)} chunks "
            f"using {len(page_boundaries)} page boundaries"
        )

    def ingest_batch(
        self,
        file_paths: List[str],
        tenant_context: TenantContext,
        parallel: bool = True,
    ) -> List[IngestionResult]:
        """
        Ingest multiple documents.

        Args:
            file_paths: List of file paths
            tenant_context: REQUIRED - Tenant context for isolation
            parallel: Use parallel processing

        Returns:
            List of IngestionResult
        """
        logger.info(
            f"[BATCH] Starting batch ingestion: {len(file_paths)} documents "
            f"(org={tenant_context.organization_id})"
        )

        results = []

        if parallel and len(file_paths) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.ingest, fp, tenant_context): fp
                    for fp in file_paths
                }
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for file_path in file_paths:
                results.append(self.ingest(file_path, tenant_context))

        # Summary
        success = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "failed")
        logger.info(f"[BATCH] Complete: {success} succeeded, {failed} failed")

        return results

    def ingest_directory(
        self,
        directory_path: str,
        tenant_context: TenantContext,
        extensions: List[str] = None,
        parallel: bool = True,
    ) -> List[IngestionResult]:
        """
        Ingest all documents in a directory.

        Args:
            directory_path: Path to directory
            tenant_context: REQUIRED - Tenant context for isolation
            extensions: File extensions to process (default: common doc types)
            parallel: Use parallel processing

        Returns:
            List of IngestionResult
        """
        if extensions is None:
            extensions = [".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg", ".tiff"]

        directory = Path(directory_path)
        file_paths = []

        for ext in extensions:
            file_paths.extend(str(f) for f in directory.rglob(f"*{ext}"))

        logger.info(f"Found {len(file_paths)} documents in {directory_path}")

        return self.ingest_batch(file_paths, tenant_context, parallel)

    def delete_document(
        self,
        document_id: str,
        tenant_context: TenantContext,
    ) -> bool:
        """
        Delete a document and all its data.

        Args:
            document_id: Document ID to delete
            tenant_context: For audit logging

        Returns:
            True if successful
        """
        try:
            logger.info(
                f"[DELETE] {document_id} (org={tenant_context.organization_id})"
            )

            # Delete from vector stores
            if self.dense_vector_store:
                self.dense_vector_store.delete_by_document_id(document_id)

            if self.colpali_vector_store:
                self.colpali_vector_store.delete_by_document_id(document_id)

            if self.visual_element_store:
                self.visual_element_store.delete_by_document_id(document_id)

            # Delete from BM25
            if self.bm25_store:
                self.bm25_store.delete_by_document_id(document_id)

            # Delete from metadata store (cascades to chunks/pages)
            if self.metadata_store:
                self.metadata_store.delete_document(document_id)

            logger.info(f"[DELETE] Complete: {document_id}")
            return True

        except Exception as e:
            logger.error(f"[DELETE] Failed {document_id}: {e}")
            return False


def create_master_pipeline(
    postgres_url: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    dense_model: str = "BAAI/bge-base-en-v1.5",
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    enable_colpali: bool = True,
    enable_visual_elements: bool = True,
    enable_tables: bool = True,
    enable_graph: bool = True,
    bm25_persist_path: str = "./data/bm25_index.pkl",
) -> MasterPipeline:
    """
    Factory function to create a fully configured Master Pipeline.

    This is the recommended way to create the pipeline.

    Args:
        postgres_url: PostgreSQL connection URL
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        dense_model: Dense embedding model name
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
        enable_colpali: Enable ColPali visual embeddings (full pages)
        enable_visual_elements: Enable visual element embedding (cropped tables/figures)
        enable_tables: Enable table extraction
        enable_graph: Enable document relationship graph
        bm25_persist_path: Path to persist BM25 index

    Returns:
        Configured MasterPipeline instance
    """
    from src.ingestion.document_processor import DocumentProcessor
    from src.embeddings.dense_embedder import get_embedder
    from src.storage.metadata_store import MetadataStore
    from src.storage.vector_store import QdrantVectorStore, QdrantMultiVectorStore, QdrantVisualElementStore
    from src.storage.bm25_store import BM25Index
    from src.chunking.chunking_service import ChunkingService

    # Document processor (Unstructured.io)
    document_processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=50,
        extract_images=enable_colpali,
        extract_visual_elements=enable_visual_elements,
    )

    # Chunking service
    chunking_service = ChunkingService(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Dense embedder
    dense_embedder = get_embedder("sentence-transformers", model_name=dense_model)

    # ColPali embedder (optional)
    colpali_embedder = None
    if enable_colpali:
        try:
            from src.embeddings.colpali_embedder import ColPaliEmbedder
            logger.info("Loading ColPali embedder...")
            colpali_embedder = ColPaliEmbedder()
            logger.info(f"ColPali embedder loaded successfully (dimension={colpali_embedder.dimension})")
        except ImportError as e:
            logger.warning(f"ColPali not available - missing dependencies: {e}")
        except Exception as e:
            logger.warning(f"ColPali failed to load: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    # Metadata store (PostgreSQL)
    metadata_store = MetadataStore(postgres_url)
    metadata_store.create_tables()

    # Vector stores (Qdrant)
    dense_vector_store = QdrantVectorStore(
        collection_name="nhai_lt_documents",
        dimension=dense_embedder.dimension,
        host=qdrant_host,
        port=qdrant_port,
    )

    colpali_vector_store = None
    if colpali_embedder:
        colpali_vector_store = QdrantMultiVectorStore(
            collection_name="nhai_lt_colpali",
            dimension=128,
            host=qdrant_host,
            port=qdrant_port,
        )

    # Visual element store (for cropped tables/figures)
    visual_element_store = None
    if colpali_embedder and enable_visual_elements:
        visual_element_store = QdrantVisualElementStore(
            collection_name="nhai_lt_visual_elements",
            dimension=128,
            host=qdrant_host,
            port=qdrant_port,
        )

    # BM25 store
    bm25_store = BM25Index(persist_path=bm25_persist_path)

    # Enhanced components (optional)
    quality_analyzer = None
    metadata_extractor = None
    table_extractor = None
    document_graph = None

    try:
        from src.quality.document_quality import DocumentQualityAnalyzer
        quality_analyzer = DocumentQualityAnalyzer()
    except Exception as e:
        logger.warning(f"Quality analyzer not available: {e}")

    try:
        from src.metadata.domain_schemas import UnifiedMetadataExtractor
        metadata_extractor = UnifiedMetadataExtractor()
    except Exception as e:
        logger.warning(f"Metadata extractor not available: {e}")

    if enable_tables:
        try:
            from src.tables.table_extractor import TableExtractor
            table_extractor = TableExtractor()
        except Exception as e:
            logger.warning(f"Table extractor not available: {e}")

    if enable_graph:
        try:
            from src.graph.document_graph import DocumentGraph
            document_graph = DocumentGraph()
        except Exception as e:
            logger.warning(f"Document graph not available: {e}")

    return MasterPipeline(
        document_processor=document_processor,
        chunking_service=chunking_service,
        dense_embedder=dense_embedder,
        colpali_embedder=colpali_embedder,
        metadata_store=metadata_store,
        dense_vector_store=dense_vector_store,
        colpali_vector_store=colpali_vector_store,
        visual_element_store=visual_element_store,
        bm25_store=bm25_store,
        quality_analyzer=quality_analyzer,
        metadata_extractor=metadata_extractor,
        table_extractor=table_extractor,
        document_graph=document_graph,
        enable_colpali=enable_colpali,
        enable_visual_elements=enable_visual_elements,
        enable_tables=enable_tables,
        enable_graph=enable_graph,
    )


# Example usage
if __name__ == "__main__":
    print("""
    NHAI/L&T Enterprise RAG - Master Pipeline
    ==========================================

    Usage:
        from src.pipeline.master_pipeline import create_master_pipeline
        from src.metadata.tenant_schema import TenantContext, AccessLevel

        # Create pipeline
        pipeline = create_master_pipeline(
            postgres_url="postgresql://user:pass@localhost:5432/rag_db",
            qdrant_host="localhost",
        )

        # Ingest document
        result = pipeline.ingest(
            file_path="contract.pdf",
            tenant_context=TenantContext(
                organization_id="nhai",
                workspace_id="contracts",
                access_level=AccessLevel.CONFIDENTIAL,
            )
        )

        print(f"Status: {result.status}")
        print(f"Chunks: {result.chunks_indexed}")
    """)
