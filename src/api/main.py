"""
FastAPI application for the RAG system.

Provides REST API endpoints for:
- Document ingestion
- Hybrid search
- Document management
- Health checks
- Multi-tenant isolation

IMPORTANT: All endpoints require X-Organization-ID header for tenant isolation.
"""
import os
# Disable tokenizer parallelism to avoid fork warnings with HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # Load .env file

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.metadata.tenant_schema import TenantContext, AccessLevel

logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
pipeline = None
hybrid_searcher = None


# ============================================================================
# Tenant Context Dependency
# ============================================================================

async def get_tenant_context(
    x_organization_id: str = Header(..., description="Organization ID (required)"),
    x_organization_name: Optional[str] = Header(default=None, description="Organization name"),
    x_workspace_id: Optional[str] = Header(default=None, description="Workspace ID"),
    x_workspace_name: Optional[str] = Header(default=None, description="Workspace name"),
    x_collection_id: Optional[str] = Header(default=None, description="Collection ID"),
    x_user_id: Optional[str] = Header(default=None, description="User ID for audit"),
    x_user_email: Optional[str] = Header(default=None, description="User email for audit"),
    x_access_level: Optional[str] = Header(default="internal", description="Access level"),
) -> TenantContext:
    """
    Extract tenant context from request headers.

    REQUIRED HEADER: X-Organization-ID

    This dependency ensures ALL requests are properly scoped to a tenant.
    """
    if not x_organization_id or len(x_organization_id) < 3:
        raise HTTPException(
            status_code=400,
            detail="X-Organization-ID header is required and must be at least 3 characters"
        )

    try:
        access_level = AccessLevel(x_access_level) if x_access_level else AccessLevel.INTERNAL
    except ValueError:
        access_level = AccessLevel.INTERNAL

    return TenantContext(
        organization_id=x_organization_id,
        organization_name=x_organization_name or x_organization_id,
        workspace_id=x_workspace_id,
        workspace_name=x_workspace_name,
        collection_id=x_collection_id,
        user_id=x_user_id,
        user_email=x_user_email,
        access_level=access_level,
    )


# ============================================================================
# Pydantic Models
# ============================================================================

class SearchRequest(BaseModel):
    """Search request body."""
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    methods: Optional[List[str]] = Field(
        default=None,
        description="Search methods: bm25, dense, colpali"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata filters (tenant filters added automatically)"
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom weights for RRF fusion"
    )


class SearchResult(BaseModel):
    """Individual search result."""
    id: str
    score: float
    text: Optional[str] = None
    metadata: Dict[str, Any] = {}
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    colpali_rank: Optional[int] = None
    colpali_score: Optional[float] = None
    colpali_page_match: bool = False  # True if ColPali score from same page
    colpali_doc_match: bool = False   # True if ColPali score from same document


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    organization_id: str
    total_results: int
    results: List[SearchResult]


class IngestResponse(BaseModel):
    """Ingestion response."""
    document_id: str
    filename: str
    organization_id: str
    workspace_id: Optional[str] = None
    status: str
    chunks_indexed: int
    pages_indexed: int
    message: Optional[str] = None


class DocumentInfo(BaseModel):
    """Document information."""
    id: str
    filename: str
    organization_id: str
    workspace_id: Optional[str] = None
    document_type: Optional[str] = None
    status: Optional[str] = None
    total_chunks: Optional[int] = None
    total_pages: Optional[int] = None
    created_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    services: Dict[str, str]


class StatsResponse(BaseModel):
    """System statistics."""
    database: Dict[str, Any]
    vector_store: Dict[str, Any]
    bm25: Dict[str, Any]


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global pipeline, hybrid_searcher

    logger.info("Initializing RAG services...")

    # Get config from environment
    postgres_url = os.getenv(
        "POSTGRES_URL",
        "postgresql://user:password@localhost:5432/rag_db"
    )
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    enable_colpali = os.getenv("ENABLE_COLPALI", "true").lower() == "true"

    try:
        # Import and create MasterPipeline (unified pipeline)
        from src.pipeline import create_master_pipeline
        pipeline = create_master_pipeline(
            postgres_url=postgres_url,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            enable_colpali=enable_colpali
        )

        # Log component status for debugging
        logger.info(f"Pipeline components status:")
        logger.info(f"  - BM25 store: {'✓' if pipeline.bm25_store else '✗'}")
        logger.info(f"  - Dense store: {'✓' if pipeline.dense_vector_store else '✗'}")
        logger.info(f"  - Dense embedder: {'✓' if pipeline.dense_embedder else '✗'}")
        logger.info(f"  - ColPali store: {'✓' if pipeline.colpali_vector_store else '✗'}")
        logger.info(f"  - ColPali embedder: {'✓' if pipeline.colpali_embedder else '✗'}")

        # If ColPali embedder failed to load in pipeline, try loading it separately
        colpali_embedder = pipeline.colpali_embedder
        colpali_store = pipeline.colpali_vector_store

        if enable_colpali and not colpali_embedder:
            logger.warning("ColPali embedder not loaded by pipeline, attempting separate load...")
            try:
                from src.embeddings.colpali_embedder import ColPaliEmbedder
                colpali_embedder = ColPaliEmbedder()
                logger.info("✓ ColPali embedder loaded separately")

                # Also create the store if embedder loaded successfully
                if not colpali_store:
                    from src.storage.vector_store import QdrantMultiVectorStore
                    colpali_store = QdrantMultiVectorStore(
                        collection_name="nhai_lt_colpali",
                        dimension=128,
                        host=qdrant_host,
                        port=qdrant_port,
                    )
                    logger.info("✓ ColPali store created separately")
            except Exception as e:
                logger.warning(f"ColPali separate load failed: {e}")

        # Create hybrid searcher with all available components
        from src.retrieval.hybrid_search import HybridSearcher
        hybrid_searcher = HybridSearcher(
            bm25_store=pipeline.bm25_store,
            dense_store=pipeline.dense_vector_store,
            colpali_store=colpali_store,
            dense_embedder=pipeline.dense_embedder,
            colpali_embedder=colpali_embedder,
        )

        # Log final search capabilities
        search_methods = []
        if pipeline.bm25_store:
            search_methods.append("BM25")
        if pipeline.dense_vector_store and pipeline.dense_embedder:
            search_methods.append("Dense")
        if colpali_store and colpali_embedder:
            search_methods.append("ColPali")

        logger.info(f"RAG services initialized successfully!")
        logger.info(f"Available search methods: {', '.join(search_methods)}")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        import traceback
        traceback.print_exc()
        # Allow startup but endpoints will return errors

    yield

    # Cleanup
    logger.info("Shutting down RAG services...")
    if pipeline and pipeline.bm25_store:
        try:
            pipeline.bm25_store.save()
        except Exception:
            pass


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Enterprise Document RAG API",
    description="""
    Multi-tenant RAG API for document ingestion and hybrid search.

    **Authentication**: All endpoints (except /health) require:
    - `X-Organization-ID` header (required)
    - `X-Workspace-ID` header (optional)

    **Features**:
    - BM25 + Dense + ColPali hybrid search
    - Multi-tenant isolation
    - Document quality detection
    - Hierarchical chunking
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Endpoints (no auth required)
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check health of all services. No authentication required."""
    services = {}

    # Check database
    try:
        if pipeline and pipeline.metadata_store:
            pipeline.metadata_store.get_stats()
            services["database"] = "healthy"
        else:
            services["database"] = "not_configured"
    except Exception as e:
        services["database"] = f"unhealthy: {str(e)}"

    # Check Qdrant
    try:
        if pipeline and pipeline.dense_vector_store:
            pipeline.dense_vector_store.get_collection_info()
            services["qdrant"] = "healthy"
        else:
            services["qdrant"] = "not_configured"
    except Exception as e:
        services["qdrant"] = f"unhealthy: {str(e)}"

    # Check BM25
    try:
        if pipeline and pipeline.bm25_store:
            pipeline.bm25_store.get_stats()
            services["bm25"] = "healthy"
        else:
            services["bm25"] = "not_configured"
    except Exception as e:
        services["bm25"] = f"unhealthy: {str(e)}"

    overall = "healthy" if all("healthy" in v for v in services.values()) else "degraded"
    return HealthResponse(status=overall, services=services)


@app.get("/stats", response_model=StatsResponse, tags=["Health"])
async def get_stats():
    """Get system statistics. No authentication required."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Services not initialized")

    db_stats = {}
    vector_stats = {}
    bm25_stats = {}

    if pipeline.metadata_store:
        db_stats = pipeline.metadata_store.get_stats()

    if pipeline.dense_vector_store:
        vector_stats = pipeline.dense_vector_store.get_collection_info()

    if pipeline.bm25_store:
        bm25_stats = pipeline.bm25_store.get_stats()

    return StatsResponse(
        database=db_stats,
        vector_store=vector_stats,
        bm25=bm25_stats
    )


# ============================================================================
# Search Endpoints
# ============================================================================

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    request: SearchRequest,
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Perform hybrid search across documents.

    Combines BM25 (keyword), dense (semantic), and ColPali (visual) search
    using Reciprocal Rank Fusion.

    Results are automatically filtered to the organization specified in headers.
    """
    if not hybrid_searcher:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        # Parse methods
        methods = None
        if request.methods:
            from src.retrieval.hybrid_search import RetrievalMethod
            methods = []
            for m in request.methods:
                try:
                    methods.append(RetrievalMethod(m.lower()))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid method: {m}. Use: bm25, dense, colpali"
                    )

        # Merge tenant filters with request filters
        filters = tenant.to_filter_dict()
        if request.filters:
            filters.update(request.filters)

        # Perform search with tenant isolation
        results = hybrid_searcher.search(
            query=request.query,
            limit=request.limit,
            methods=methods,
            weights=request.weights,
            filters=filters
        )

        # Convert to response
        search_results = [
            SearchResult(
                id=r.id,
                score=r.final_score,
                text=r.text,
                metadata=r.metadata,
                bm25_rank=r.bm25_rank,
                bm25_score=r.bm25_score,
                dense_rank=r.dense_rank,
                dense_score=r.dense_score,
                colpali_rank=r.colpali_rank,
                colpali_score=r.colpali_score,
                colpali_page_match=r.colpali_page_match,
                colpali_doc_match=r.colpali_doc_match,
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            organization_id=tenant.organization_id,
            total_results=len(search_results),
            results=search_results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    q: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    limit: int = Query(default=10, ge=1, le=100),
    document_type: Optional[str] = Query(default=None, description="Filter by document type"),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """Simple GET search endpoint with tenant isolation."""
    filters = {"document_type": document_type} if document_type else None
    request = SearchRequest(query=q, limit=limit, filters=filters)
    return await search(request, tenant)


# ============================================================================
# Document Ingestion Endpoints
# ============================================================================

@app.post("/documents/upload", response_model=IngestResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    document_type: Optional[str] = Query(default=None),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Upload and ingest a document.

    Supports PDF, DOCX, TXT, PNG, JPG files.
    Document is automatically associated with the organization in headers.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Ingestion service not initialized")

    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".tiff"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Ingest document with tenant context using MasterPipeline
        custom_metadata = {}
        if document_type:
            custom_metadata["document_type_hint"] = document_type

        # MasterPipeline uses ingest() method
        result = pipeline.ingest(
            file_path=tmp_path,
            tenant_context=tenant,
            custom_metadata=custom_metadata
        )

        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        # Save BM25 index
        if pipeline.bm25_store:
            pipeline.bm25_store.save()

        return IngestResponse(
            document_id=result.document_id,
            filename=result.filename,
            organization_id=tenant.organization_id,
            workspace_id=tenant.workspace_id,
            status=result.status,
            chunks_indexed=result.chunks_indexed,
            pages_indexed=result.pages_indexed,
            message=result.error_message
        )

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/ingest-path", response_model=IngestResponse, tags=["Documents"])
async def ingest_from_path(
    file_path: str = Query(..., description="Path to document on server"),
    document_type: Optional[str] = Query(default=None),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """Ingest a document from a server file path with tenant isolation."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Ingestion service not initialized")

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        custom_metadata = {}
        if document_type:
            custom_metadata["document_type_hint"] = document_type

        # MasterPipeline uses ingest() method
        result = pipeline.ingest(
            file_path=file_path,
            tenant_context=tenant,
            custom_metadata=custom_metadata
        )

        if pipeline.bm25_store:
            pipeline.bm25_store.save()

        return IngestResponse(
            document_id=result.document_id,
            filename=result.filename,
            organization_id=tenant.organization_id,
            workspace_id=tenant.workspace_id,
            status=result.status,
            chunks_indexed=result.chunks_indexed,
            pages_indexed=result.pages_indexed,
            message=result.error_message
        )

    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents(
    document_type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """List documents for the organization. Automatically filtered by tenant."""
    if not pipeline or not pipeline.metadata_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        # TODO: Add tenant filtering to list_documents
        documents = pipeline.metadata_store.list_documents(
            document_type=document_type,
            status=status,
            limit=limit,
            offset=offset
        )

        # Filter by organization (should be done in query for efficiency)
        # For now, filter in memory
        org_docs = [
            doc for doc in documents
            if getattr(doc, 'organization_id', None) == tenant.organization_id
            or not hasattr(doc, 'organization_id')  # Legacy docs without org_id
        ]

        return [
            DocumentInfo(
                id=doc.id,
                filename=doc.filename,
                organization_id=getattr(doc, 'organization_id', tenant.organization_id),
                workspace_id=getattr(doc, 'workspace_id', None),
                document_type=doc.document_type.value if doc.document_type else None,
                status=doc.status.value if doc.status else None,
                total_chunks=doc.total_chunks,
                total_pages=doc.total_pages,
                created_at=doc.created_at.isoformat() if doc.created_at else None,
                metadata=doc._metadata
            )
            for doc in org_docs
        ]

    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}", response_model=DocumentInfo, tags=["Documents"])
async def get_document(
    document_id: str,
    tenant: TenantContext = Depends(get_tenant_context)
):
    """Get a specific document by ID. Must belong to the tenant's organization."""
    if not pipeline or not pipeline.metadata_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    doc = pipeline.metadata_store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Verify document belongs to tenant's organization
    doc_org_id = getattr(doc, 'organization_id', None)
    if doc_org_id and doc_org_id != tenant.organization_id:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentInfo(
        id=doc.id,
        filename=doc.filename,
        organization_id=doc_org_id or tenant.organization_id,
        workspace_id=getattr(doc, 'workspace_id', None),
        document_type=doc.document_type.value if doc.document_type else None,
        status=doc.status.value if doc.status else None,
        total_chunks=doc.total_chunks,
        total_pages=doc.total_pages,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
        metadata=doc._metadata
    )


@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    tenant: TenantContext = Depends(get_tenant_context)
):
    """Delete a document and all its data. Must belong to the tenant's organization."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Services not initialized")

    # Verify document belongs to tenant before deleting
    if pipeline.metadata_store:
        doc = pipeline.metadata_store.get_document(document_id)
        if doc:
            doc_org_id = getattr(doc, 'organization_id', None)
            if doc_org_id and doc_org_id != tenant.organization_id:
                raise HTTPException(status_code=404, detail="Document not found")

    success = pipeline.delete_document(document_id, tenant)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or delete failed")

    if pipeline.bm25_store:
        pipeline.bm25_store.save()

    return {
        "status": "deleted",
        "document_id": document_id,
        "organization_id": tenant.organization_id
    }


# ============================================================================
# Q&A Endpoint (VLM-powered RAG)
# ============================================================================

class AskRequest(BaseModel):
    """RAG question-answering request with multimodal support."""
    query: str = Field(..., min_length=1, max_length=2000, description="Question to answer")
    limit: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    include_images: bool = Field(default=True, description="Include visual context (page images)")
    max_images: int = Field(default=5, ge=0, le=10, description="Maximum images to include")
    model: Optional[str] = Field(default=None, description="VLM model override (e.g., gemini-1.5-pro)")
    temperature: float = Field(default=0.3, ge=0, le=1, description="Generation temperature")
    document_type: Optional[str] = Field(default=None, description="Filter by document type")


class AskResponse(BaseModel):
    """RAG question-answering response."""
    query: str
    answer: str
    sources: List[SearchResult]
    images_used: int
    model: str
    generation_time_ms: int
    retrieval_time_ms: int
    organization_id: str


# Global VLM generator (initialized on first use)
vlm_generator = None


def get_vlm_generator():
    """Get or create VLM generator instance."""
    global vlm_generator
    if vlm_generator is None:
        try:
            from src.generation import create_vlm_generator
            vlm_generator = create_vlm_generator()
            logger.info(f"VLM generator initialized: {vlm_generator.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize VLM generator: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"VLM service not available: {str(e)}"
            )
    return vlm_generator


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(
    request: AskRequest,
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Ask a question and get a generated answer with sources.

    This endpoint implements multimodal RAG:
    1. Retrieves relevant text chunks via hybrid search (BM25 + Dense + ColPali)
    2. Fetches relevant page images for visual context (if include_images=True)
    3. Generates an answer using Google Gemini VLM
    4. Returns the answer with source citations

    **Visual Context**: When include_images=True, the system will include
    page images from documents where ColPali found visual matches. This
    enables answering questions about tables, charts, and figures.

    **Example queries**:
    - "What does the table on page 3 show?"
    - "Summarize the contract terms"
    - "What are the payment milestones?"
    """
    import time
    retrieval_start = time.time()

    if not hybrid_searcher:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        # Build filters with tenant isolation
        filters = tenant.to_filter_dict()
        if request.document_type:
            filters["document_type"] = request.document_type

        # Step 1: Retrieve relevant chunks via hybrid search
        search_results = hybrid_searcher.search(
            query=request.query,
            limit=request.limit,
            filters=filters
        )

        retrieval_time = int((time.time() - retrieval_start) * 1000)

        if not search_results:
            return AskResponse(
                query=request.query,
                answer="I couldn't find any relevant information in the documents for your query.",
                sources=[],
                images_used=0,
                model="none",
                generation_time_ms=0,
                retrieval_time_ms=retrieval_time,
                organization_id=tenant.organization_id,
            )

        # Step 2: Extract text context and metadata
        text_context = []
        metadata_list = []
        for r in search_results:
            text_context.append(r.text or "")
            metadata_list.append({
                "filename": r.metadata.get("filename"),
                "page_number": r.metadata.get("page_number"),
                "section_title": r.metadata.get("section_title"),
                "document_type": r.metadata.get("document_type"),
            })

        # Step 3: Fetch page images if requested and available
        images = []
        if request.include_images and request.max_images > 0:
            images = await _fetch_context_images(
                search_results,
                max_images=request.max_images,
                tenant=tenant
            )

        # Step 4: Generate answer using VLM
        generator = get_vlm_generator()

        # Override model if specified
        if request.model and request.model != generator.model_name:
            from src.generation import GeminiVLMGenerator
            temp_generator = GeminiVLMGenerator(model=request.model)
            gen_result = temp_generator.generate(
                query=request.query,
                text_context=text_context,
                images=images if images else None,
                metadata=metadata_list,
                temperature=request.temperature,
            )
        else:
            gen_result = generator.generate(
                query=request.query,
                text_context=text_context,
                images=images if images else None,
                metadata=metadata_list,
                temperature=request.temperature,
            )

        # Check for generation errors
        if gen_result.error:
            logger.error(f"VLM generation error: {gen_result.error}")
            answer = f"Error generating answer: {gen_result.error}"
        else:
            answer = gen_result.answer

        # Build response with sources
        sources = [
            SearchResult(
                id=r.id,
                score=r.final_score,
                text=r.text,
                metadata=r.metadata,
                bm25_rank=r.bm25_rank,
                bm25_score=r.bm25_score,
                dense_rank=r.dense_rank,
                dense_score=r.dense_score,
                colpali_rank=r.colpali_rank,
                colpali_score=r.colpali_score,
                colpali_page_match=r.colpali_page_match,
                colpali_doc_match=r.colpali_doc_match,
            )
            for r in search_results
        ]

        return AskResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            images_used=gen_result.images_used,
            model=gen_result.model,
            generation_time_ms=gen_result.generation_time_ms,
            retrieval_time_ms=retrieval_time,
            organization_id=tenant.organization_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _fetch_context_images(
    search_results: list,
    max_images: int,
    tenant: TenantContext
) -> List:
    """
    Fetch page images for search results that have ColPali matches.

    This retrieves stored page images to provide visual context to the VLM.
    Images are fetched for pages where ColPali found visual relevance.
    """
    from PIL import Image
    import io

    images = []

    # Collect unique (document_id, page_number) pairs from results with ColPali matches
    pages_to_fetch = []
    seen_pages = set()

    for r in search_results:
        # Prioritize results with direct page matches
        if r.colpali_page_match or r.colpali_doc_match:
            doc_id = r.metadata.get("document_id")
            page_num = r.metadata.get("page_number")

            if doc_id and page_num and (doc_id, page_num) not in seen_pages:
                pages_to_fetch.append((doc_id, page_num, r.colpali_score or 0))
                seen_pages.add((doc_id, page_num))

    # Sort by ColPali score (highest first) and limit
    pages_to_fetch.sort(key=lambda x: x[2], reverse=True)
    pages_to_fetch = pages_to_fetch[:max_images]

    # Try to fetch images from metadata store
    if pipeline and pipeline.metadata_store and pages_to_fetch:
        for doc_id, page_num, _ in pages_to_fetch:
            try:
                # Try to get page record
                page = pipeline.metadata_store.get_page(doc_id, page_num)
                if page and hasattr(page, 'image_data') and page.image_data:
                    # Decode stored image
                    img = Image.open(io.BytesIO(page.image_data))
                    images.append(img)
                    logger.debug(f"Fetched image for {doc_id} page {page_num}")
            except Exception as e:
                logger.warning(f"Could not fetch image for {doc_id} page {page_num}: {e}")

    # If no stored images, try to get from document path (fallback)
    if not images and pages_to_fetch:
        logger.info("No stored images found, skipping visual context")

    return images


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
