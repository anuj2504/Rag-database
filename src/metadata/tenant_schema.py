"""
Multi-Tenant Schema for Organization Isolation.

Critical for enterprise RAG:
- Data from Org A must NEVER mix with Org B
- All queries must be scoped to tenant
- Support for departments/projects within orgs
- Audit trail for compliance

Hierarchy:
  Organization (tenant)
    └── Workspace (optional - department/project)
        └── Collection (logical grouping)
            └── Documents
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access levels for documents."""
    PUBLIC = "public"           # Visible to all in org
    INTERNAL = "internal"       # Visible to workspace members
    CONFIDENTIAL = "confidential"  # Restricted access
    SECRET = "secret"           # Highly restricted


@dataclass
class TenantContext:
    """
    Tenant context that MUST be present in all operations.

    This is the primary isolation mechanism.
    Every query, every ingestion, every operation must have this.
    """
    organization_id: str         # Primary tenant identifier
    organization_name: str       # Human-readable name

    # Optional hierarchy
    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    collection_id: Optional[str] = None
    collection_name: Optional[str] = None

    # User context (for audit)
    user_id: Optional[str] = None
    user_email: Optional[str] = None

    # Access control
    access_level: AccessLevel = AccessLevel.INTERNAL
    allowed_access_levels: List[AccessLevel] = field(
        default_factory=lambda: [AccessLevel.PUBLIC, AccessLevel.INTERNAL]
    )

    def to_filter_dict(self) -> Dict[str, Any]:
        """Convert to filter dictionary for queries."""
        filters = {
            "organization_id": self.organization_id,
        }
        if self.workspace_id:
            filters["workspace_id"] = self.workspace_id
        if self.collection_id:
            filters["collection_id"] = self.collection_id

        # Access level filtering
        filters["access_level"] = [al.value for al in self.allowed_access_levels]

        return filters

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to metadata for storage."""
        return {
            "organization_id": self.organization_id,
            "organization_name": self.organization_name,
            "workspace_id": self.workspace_id,
            "workspace_name": self.workspace_name,
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "access_level": self.access_level.value,
        }


@dataclass
class TenantMetadata:
    """
    Metadata fields added to EVERY document for tenant isolation.

    These fields are indexed for efficient filtering.
    """
    # Required - Primary isolation
    organization_id: str

    # Optional hierarchy
    workspace_id: Optional[str] = None
    collection_id: Optional[str] = None

    # Access control
    access_level: str = "internal"
    owner_id: Optional[str] = None
    shared_with: List[str] = field(default_factory=list)  # User/group IDs

    # Audit fields
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None
    updated_at: Optional[datetime] = None

    # Compliance
    retention_policy: Optional[str] = None  # e.g., "7years", "permanent"
    classification: Optional[str] = None     # e.g., "PII", "PHI", "financial"
    legal_hold: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "collection_id": self.collection_id,
            "access_level": self.access_level,
            "owner_id": self.owner_id,
            "shared_with": self.shared_with,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "retention_policy": self.retention_policy,
            "classification": self.classification,
            "legal_hold": self.legal_hold,
        }

    @classmethod
    def from_context(
        cls,
        context: TenantContext,
        **kwargs
    ) -> "TenantMetadata":
        """Create from tenant context."""
        return cls(
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
            collection_id=context.collection_id,
            access_level=context.access_level.value,
            owner_id=context.user_id,
            created_by=context.user_id,
            **kwargs
        )


class TenantManager:
    """
    Manages tenant operations and validation.

    Use this to:
    - Validate tenant context
    - Create tenant-scoped filters
    - Ensure isolation in all operations
    """

    def __init__(self, metadata_store=None):
        self.metadata_store = metadata_store
        self._tenant_cache: Dict[str, Dict] = {}

    def validate_context(self, context: TenantContext) -> bool:
        """
        Validate that tenant context is valid.

        In production, this would check against a tenant registry.
        """
        if not context.organization_id:
            raise ValueError("organization_id is required")

        if len(context.organization_id) < 3:
            raise ValueError("organization_id must be at least 3 characters")

        # Could add database validation here
        return True

    def create_document_metadata(
        self,
        context: TenantContext,
        document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge tenant metadata with document metadata.

        ALWAYS use this when creating documents.
        """
        tenant_meta = TenantMetadata.from_context(context)

        # Merge - tenant metadata takes precedence for isolation fields
        merged = {**document_metadata, **tenant_meta.to_dict()}

        return merged

    def create_query_filter(
        self,
        context: TenantContext,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create filter that ensures tenant isolation.

        ALWAYS use this when querying.
        """
        # Start with mandatory tenant filter
        filters = context.to_filter_dict()

        # Add any additional filters
        if additional_filters:
            filters.update(additional_filters)

        return filters

    def scope_chunk_payload(
        self,
        payload: Dict[str, Any],
        context: TenantContext
    ) -> Dict[str, Any]:
        """
        Add tenant fields to chunk payload for vector store.

        Use this when preparing chunks for embedding/indexing.
        """
        tenant_fields = {
            "organization_id": context.organization_id,
            "workspace_id": context.workspace_id,
            "collection_id": context.collection_id,
            "access_level": context.access_level.value,
        }

        return {**payload, **tenant_fields}

    def scope_bm25_document(
        self,
        document: Dict[str, Any],
        context: TenantContext
    ) -> Dict[str, Any]:
        """
        Add tenant fields to BM25 document.

        Use this when indexing in BM25.
        """
        return self.scope_chunk_payload(document, context)


# SQL Schema additions for PostgreSQL
TENANT_SQL_SCHEMA = """
-- Organizations (tenants) table
CREATE TABLE IF NOT EXISTS organizations (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE
);

-- Workspaces within organizations
CREATE TABLE IF NOT EXISTS workspaces (
    id VARCHAR(255) PRIMARY KEY,
    organization_id VARCHAR(255) NOT NULL REFERENCES organizations(id),
    name VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings JSONB DEFAULT '{}'
);

-- Collections within workspaces
CREATE TABLE IF NOT EXISTS collections (
    id VARCHAR(255) PRIMARY KEY,
    organization_id VARCHAR(255) NOT NULL REFERENCES organizations(id),
    workspace_id VARCHAR(255) REFERENCES workspaces(id),
    name VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings JSONB DEFAULT '{}'
);

-- Add tenant columns to documents table
ALTER TABLE documents ADD COLUMN IF NOT EXISTS organization_id VARCHAR(255);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS workspace_id VARCHAR(255);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS collection_id VARCHAR(255);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS access_level VARCHAR(50) DEFAULT 'internal';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS owner_id VARCHAR(255);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_by VARCHAR(255);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS classification VARCHAR(100);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS legal_hold BOOLEAN DEFAULT FALSE;

-- Create indexes for tenant filtering (CRITICAL for performance)
CREATE INDEX IF NOT EXISTS idx_documents_org ON documents(organization_id);
CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(organization_id, workspace_id);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(organization_id, collection_id);
CREATE INDEX IF NOT EXISTS idx_documents_access ON documents(organization_id, access_level);

-- Add tenant columns to chunks table
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS organization_id VARCHAR(255);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS workspace_id VARCHAR(255);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS access_level VARCHAR(50) DEFAULT 'internal';

-- Create indexes for chunk filtering
CREATE INDEX IF NOT EXISTS idx_chunks_org ON chunks(organization_id);
CREATE INDEX IF NOT EXISTS idx_chunks_workspace ON chunks(organization_id, workspace_id);

-- Row-level security policy (optional but recommended)
-- This ensures queries can ONLY see rows for the current tenant
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY tenant_isolation_policy ON documents
--     USING (organization_id = current_setting('app.current_org_id'));
"""


# Qdrant payload index configuration for tenant filtering
QDRANT_TENANT_INDEXES = [
    ("organization_id", "keyword"),   # CRITICAL - primary isolation
    ("workspace_id", "keyword"),
    ("collection_id", "keyword"),
    ("access_level", "keyword"),
]


def get_tenant_filter_for_qdrant(context: TenantContext) -> Dict[str, Any]:
    """
    Create Qdrant filter conditions for tenant isolation.

    Use this in all Qdrant searches.
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

    conditions = [
        # MANDATORY: Organization filter
        FieldCondition(
            key="organization_id",
            match=MatchValue(value=context.organization_id)
        )
    ]

    # Optional: Workspace filter
    if context.workspace_id:
        conditions.append(
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value=context.workspace_id)
            )
        )

    # Optional: Collection filter
    if context.collection_id:
        conditions.append(
            FieldCondition(
                key="collection_id",
                match=MatchValue(value=context.collection_id)
            )
        )

    # Access level filter
    if context.allowed_access_levels:
        conditions.append(
            FieldCondition(
                key="access_level",
                match=MatchAny(any=[al.value for al in context.allowed_access_levels])
            )
        )

    return Filter(must=conditions)


# Example usage
if __name__ == "__main__":
    # Create tenant context
    context = TenantContext(
        organization_id="org_acme_corp",
        organization_name="Acme Corporation",
        workspace_id="ws_legal",
        workspace_name="Legal Department",
        user_id="user_123",
        user_email="john@acme.com",
        access_level=AccessLevel.INTERNAL,
    )

    print("Tenant Context:")
    print(f"  Org: {context.organization_id}")
    print(f"  Workspace: {context.workspace_id}")

    print("\nFilter for queries:")
    print(f"  {context.to_filter_dict()}")

    print("\nMetadata for documents:")
    tenant_meta = TenantMetadata.from_context(context)
    print(f"  {tenant_meta.to_dict()}")

    # Using TenantManager
    manager = TenantManager()

    # Scope a chunk payload
    chunk_payload = {
        "document_id": "doc_123",
        "text": "Contract terms...",
        "document_type": "contract"
    }
    scoped_payload = manager.scope_chunk_payload(chunk_payload, context)
    print("\nScoped chunk payload:")
    print(f"  {scoped_payload}")
