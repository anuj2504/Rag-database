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
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
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
    dense_rank: Optional[int] = None
    colpali_rank: Optional[int] = None


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

        # Create hybrid searcher
        from src.retrieval.hybrid_search import HybridSearcher
        hybrid_searcher = HybridSearcher(
            bm25_store=pipeline.bm25_store,
            dense_store=pipeline.dense_vector_store,
            colpali_store=pipeline.colpali_vector_store,
            dense_embedder=pipeline.dense_embedder,
            colpali_embedder=pipeline.colpali_embedder,
        )

        logger.info("RAG services initialized successfully (MasterPipeline with ColPali enabled)")

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
                dense_rank=r.dense_rank,
                colpali_rank=r.colpali_rank
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
