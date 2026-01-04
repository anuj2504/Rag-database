"""
Pipeline Module - Document Ingestion for NHAI/L&T RAG System.

USE MasterPipeline for all new integrations.

Primary Exports:
- MasterPipeline: Unified ingestion pipeline (RECOMMENDED)
- create_master_pipeline: Factory function to create configured pipeline
- IngestionResult: Result of document ingestion

Usage:
    from src.pipeline import create_master_pipeline
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
"""

# Primary exports - USE THESE
from src.pipeline.master_pipeline import (
    MasterPipeline,
    IngestionResult,
    create_master_pipeline,
)

# Legacy exports - for backward compatibility only
# These are DEPRECATED and will be removed
from src.pipeline.ingestion import (
    IngestionPipeline,
    IngestionResult as LegacyIngestionResult,
    create_pipeline,
)

from src.pipeline.enhanced_ingestion import (
    EnhancedIngestionPipeline,
    EnhancedIngestionResult,
    create_enhanced_pipeline,
)

__all__ = [
    # Primary exports (USE THESE)
    "MasterPipeline",
    "IngestionResult",
    "create_master_pipeline",

    # Legacy exports (DEPRECATED)
    "IngestionPipeline",
    "create_pipeline",
    "EnhancedIngestionPipeline",
    "EnhancedIngestionResult",
    "create_enhanced_pipeline",
]
