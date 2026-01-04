"""
PostgreSQL Metadata Store using SQLAlchemy.

Stores document metadata, chunk information, and supports
complex queries for filtering and analytics.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import logging

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    Index,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
    Session,
    synonym,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
import enum

logger = logging.getLogger(__name__)

Base = declarative_base()


# Enums
class DocumentStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentType(enum.Enum):
    # Legal/Contract Documents
    CONTRACT = "contract"
    AGREEMENT = "agreement"
    TENDER = "tender"
    AMENDMENT = "amendment"
    LETTER = "letter"
    LEGAL = "legal"

    # Regulatory/Code Documents
    IRC_CODE = "irc_code"
    BUILDING_CODE = "building_code"
    CODE = "code"
    SPECIFICATION = "specification"
    STANDARD = "standard"

    # Project Documents
    DPR = "dpr"
    FEASIBILITY = "feasibility"
    ESTIMATE = "estimate"
    BOQ = "boq"

    # Financial Documents
    FINANCIAL_REPORT = "financial_report"
    FINANCIAL = "financial"
    INVOICE = "invoice"
    BUDGET = "budget"

    # Technical Documents
    DRAWING = "drawing"
    MANUAL = "manual"
    SOP = "sop"
    TECHNICAL = "technical"
    REPORT = "report"
    FORM = "form"
    DOCUMENT = "document"

    # Correspondence
    MEMO = "memo"
    CIRCULAR = "circular"

    # General
    GENERAL = "general"
    UNKNOWN = "unknown"
    OTHER = "other"


class AccessLevel(enum.Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class QualityLevel(enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    GARBAGE = "garbage"


class ChunkLevelEnum(enum.Enum):
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


class RelationType(enum.Enum):
    AMENDS = "amends"
    SUPERSEDES = "supersedes"
    REFERENCES = "references"
    INCORPORATES = "incorporates"
    EXHIBIT_OF = "exhibit_of"
    SCHEDULE_OF = "schedule_of"
    RELATED_TO = "related_to"
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"


# Models
class Organization(Base):
    """Organization (tenant) table."""
    __tablename__ = "organizations"

    id = Column(String(255), primary_key=True)
    name = Column(String(500), nullable=False)
    settings = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class Workspace(Base):
    """Workspace table (within organizations)."""
    __tablename__ = "workspaces"

    id = Column(String(255), primary_key=True)
    organization_id = Column(String(255), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(500), nullable=False)
    description = Column(Text)
    settings = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class Collection(Base):
    """Collection table (document groups within workspaces)."""
    __tablename__ = "collections"

    id = Column(String(255), primary_key=True)
    organization_id = Column(String(255), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    workspace_id = Column(String(255), ForeignKey("workspaces.id", ondelete="SET NULL"))
    name = Column(String(500), nullable=False)
    description = Column(Text)
    settings = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_collections_org", "organization_id"),
        Index("idx_collections_workspace", "workspace_id"),
    )


class Document(Base):
    """Main document table."""
    __tablename__ = "documents"

    id = Column(String(255), primary_key=True)  # document_id from processor
    filename = Column(String(500), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size_bytes = Column(Integer)
    file_extension = Column(String(20))

    document_type = Column(
        SQLEnum(DocumentType, values_callable=lambda x: [e.value for e in x]),
        default=DocumentType.OTHER
    )
    status = Column(
        SQLEnum(DocumentStatus, values_callable=lambda x: [e.value for e in x]),
        default=DocumentStatus.PENDING
    )

    # MULTI-TENANT ISOLATION - CRITICAL
    organization_id = Column(String(255), nullable=False, index=True)
    workspace_id = Column(String(255), index=True)
    collection_id = Column(String(255), index=True)
    access_level = Column(String(50), default="internal")
    owner_id = Column(String(255))
    created_by = Column(String(255))

    # Processing info
    total_chunks = Column(Integer, default=0)
    total_pages = Column(Integer)
    languages = Column(ARRAY(String), default=["eng"])

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime)

    # Compliance fields
    classification = Column(String(100))  # PII, PHI, financial, etc.
    retention_policy = Column(String(100))
    legal_hold = Column(Boolean, default=False)

    # Custom metadata (flexible JSONB)
    _metadata = Column("metadata", JSONB, default={})
    extra_metadata = synonym("_metadata")

    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")

    # Indexes - TENANT ISOLATION INDEXES ARE CRITICAL FOR PERFORMANCE
    __table_args__ = (
        Index("idx_documents_org", "organization_id"),
        Index("idx_documents_org_workspace", "organization_id", "workspace_id"),
        Index("idx_documents_org_collection", "organization_id", "collection_id"),
        Index("idx_documents_org_access", "organization_id", "access_level"),
        Index("idx_documents_type", "document_type"),
        Index("idx_documents_status", "status"),
        Index("idx_documents_created", "created_at"),
        Index("idx_documents_metadata", "metadata", postgresql_using="gin"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "file_path": self.file_path,
            "document_type": self.document_type.value if self.document_type else None,
            "status": self.status.value if self.status else None,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "collection_id": self.collection_id,
            "access_level": self.access_level,
            "total_chunks": self.total_chunks,
            "total_pages": self.total_pages,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self._metadata,
        }


class Chunk(Base):
    """Document chunks for text-based retrieval."""
    __tablename__ = "chunks"

    id = Column(String(255), primary_key=True)  # chunk_id
    document_id = Column(String(255), ForeignKey("documents.id", ondelete="CASCADE"))

    # MULTI-TENANT ISOLATION - denormalized for query performance
    organization_id = Column(String(255), nullable=False, index=True)
    workspace_id = Column(String(255), index=True)
    access_level = Column(String(50), default="internal")

    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer)
    element_type = Column(String(50))  # Title, NarrativeText, Table, etc.

    # Vector store references (match init.sql schema)
    dense_vector_id = Column(String(255))
    colpali_vector_id = Column(String(255))
    bm25_indexed = Column(Boolean, default=False)
    is_indexed = Column(Boolean, default=False)
    needs_reindex = Column(Boolean, default=False)

    # Chunk metadata
    char_count = Column(Integer)
    word_count = Column(Integer)
    _metadata = Column("metadata", JSONB, default={})
    extra_metadata = synonym("_metadata")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    # Indexes - Include tenant columns for efficient filtering
    __table_args__ = (
        Index("idx_chunks_org", "organization_id"),
        Index("idx_chunks_org_workspace", "organization_id", "workspace_id"),
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_page", "page_number"),
        Index("idx_chunks_indexed", "is_indexed"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "element_type": self.element_type,
            "is_indexed": self.is_indexed,
        }


class Page(Base):
    """Document pages for ColPali visual retrieval."""
    __tablename__ = "pages"

    id = Column(String(255), primary_key=True)  # page_id
    document_id = Column(String(255), ForeignKey("documents.id", ondelete="CASCADE"))

    # MULTI-TENANT ISOLATION - denormalized for query performance
    organization_id = Column(String(255), nullable=False, index=True)
    workspace_id = Column(String(255), index=True)

    page_number = Column(Integer, nullable=False)
    image_path = Column(Text)  # Path to stored page image

    # Page dimensions
    width = Column(Integer)
    height = Column(Integer)
    dpi = Column(Integer)

    # Vector store reference (matches init.sql)
    colpali_vector_id = Column(String(255))
    is_indexed = Column(Boolean, default=False)

    # Quality info
    ocr_confidence = Column(Float)
    has_tables = Column(Boolean, default=False)
    has_figures = Column(Boolean, default=False)

    # Page metadata
    _metadata = Column("metadata", JSONB, default={})
    extra_metadata = synonym("_metadata")

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="pages")

    # Indexes
    __table_args__ = (
        Index("idx_pages_org", "organization_id"),
        Index("idx_pages_document", "document_id"),
        Index("idx_pages_indexed", "is_indexed"),
    )


class ProcessingJob(Base):
    """Track document processing jobs."""
    __tablename__ = "processing_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(String(255), nullable=False, index=True)
    document_id = Column(String(255), ForeignKey("documents.id", ondelete="SET NULL"))

    job_type = Column(String(50), nullable=False)  # 'ingestion', 'reindex', 'delete', 'export'
    status = Column(String(20), default="pending")
    priority = Column(Integer, default=50)

    # Timing
    scheduled_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Progress
    progress_percent = Column(Integer, default=0)
    current_step = Column(String(100))

    # Results
    error_message = Column(Text)
    error_details = Column(JSONB)
    result_summary = Column(JSONB)

    # Retry info
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Worker info
    worker_id = Column(String(100))

    _metadata = Column("metadata", JSONB, default={})
    extra_metadata = synonym("_metadata")

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_jobs_org", "organization_id"),
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_document", "document_id"),
        Index("idx_jobs_type_status", "job_type", "status"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "document_id": self.document_id,
            "job_type": self.job_type,
            "status": self.status,
            "priority": self.priority,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "result_summary": self.result_summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ExtractedTable(Base):
    """Extracted tabular data from documents."""
    __tablename__ = "extracted_tables"

    id = Column(String(255), primary_key=True)
    document_id = Column(String(255), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(String(255), ForeignKey("chunks.id", ondelete="SET NULL"))

    # Multi-tenant isolation
    organization_id = Column(String(255), nullable=False, index=True)

    # Table info
    page_number = Column(Integer)
    table_index = Column(Integer)

    # Content
    html_content = Column(Text)
    markdown_content = Column(Text)
    structured_data = Column(JSONB)  # Parsed table data

    # Semantic description (for retrieval)
    description = Column(Text)

    # Dimensions
    num_rows = Column(Integer)
    num_cols = Column(Integer)

    # Vector references
    structured_vector_id = Column(String(255))
    semantic_vector_id = Column(String(255))

    # Metadata
    _metadata = Column("metadata", JSONB, default={})
    extra_metadata = synonym("_metadata")

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_tables_org", "organization_id"),
        Index("idx_tables_document", "document_id"),
        Index("idx_tables_chunk", "chunk_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "organization_id": self.organization_id,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "html_content": self.html_content,
            "markdown_content": self.markdown_content,
            "structured_data": self.structured_data,
            "description": self.description,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DocumentRelationship(Base):
    """Document relationships (graph edges)."""
    __tablename__ = "document_relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Multi-tenant isolation
    organization_id = Column(String(255), nullable=False, index=True)

    # Source document
    source_document_id = Column(String(255), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    source_chunk_id = Column(String(255), ForeignKey("chunks.id", ondelete="SET NULL"))

    # Target document
    target_document_id = Column(String(255), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    target_chunk_id = Column(String(255), ForeignKey("chunks.id", ondelete="SET NULL"))

    # Relationship info
    relation_type = Column(
        SQLEnum(RelationType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    confidence = Column(Float, default=1.0)

    # Evidence
    evidence_text = Column(Text)
    detected_by = Column(String(50))  # 'rule', 'model', 'manual'

    # Metadata
    _metadata = Column("metadata", JSONB, default={})
    extra_metadata = synonym("_metadata")

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_relationships_org", "organization_id"),
        Index("idx_relationships_source", "source_document_id"),
        Index("idx_relationships_target", "target_document_id"),
        Index("idx_relationships_type", "relation_type"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "source_document_id": self.source_document_id,
            "source_chunk_id": self.source_chunk_id,
            "target_document_id": self.target_document_id,
            "target_chunk_id": self.target_chunk_id,
            "relation_type": self.relation_type.value if self.relation_type else None,
            "confidence": self.confidence,
            "evidence_text": self.evidence_text,
            "detected_by": self.detected_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class QualityReport(Base):
    """Quality analysis reports for documents."""
    __tablename__ = "quality_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(255), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    # Multi-tenant isolation
    organization_id = Column(String(255), nullable=False, index=True)

    # Quality scores (0-100)
    overall_score = Column(Float, nullable=False)
    text_extraction_score = Column(Float)
    ocr_artifact_score = Column(Float)
    formatting_score = Column(Float)
    coherence_score = Column(Float)

    # Quality level
    quality_level = Column(
        SQLEnum(QualityLevel, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )

    # Details
    issues_found = Column(JSONB, default=[])
    recommendations = Column(JSONB, default=[])

    # Processing route
    recommended_pipeline = Column(String(50))  # hierarchical, standard, simple

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_quality_org", "organization_id"),
        Index("idx_quality_document", "document_id"),
        Index("idx_quality_level", "quality_level"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "organization_id": self.organization_id,
            "overall_score": self.overall_score,
            "text_extraction_score": self.text_extraction_score,
            "ocr_artifact_score": self.ocr_artifact_score,
            "formatting_score": self.formatting_score,
            "coherence_score": self.coherence_score,
            "quality_level": self.quality_level.value if self.quality_level else None,
            "issues_found": self.issues_found,
            "recommendations": self.recommendations,
            "recommended_pipeline": self.recommended_pipeline,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MetadataStore:
    """
    PostgreSQL metadata store operations.

    Handles all CRUD operations for documents, chunks, and pages.
    """

    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize metadata store.

        Args:
            database_url: PostgreSQL connection URL
            echo: Log SQL queries (for debugging)
        """
        self.engine = create_engine(
            database_url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Check connection health
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")

    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    # Organization/Workspace operations
    def ensure_organization(self, organization_id: str, name: Optional[str] = None) -> Organization:
        """Ensure organization exists, create if not."""
        with self.get_session() as session:
            org = session.query(Organization).filter(Organization.id == organization_id).first()
            if not org:
                org = Organization(
                    id=organization_id,
                    name=name or organization_id,
                    settings={},
                )
                session.add(org)
                session.flush()
                logger.info(f"Created organization: {organization_id}")
            return org

    def ensure_workspace(self, workspace_id: str, organization_id: str, name: Optional[str] = None) -> Workspace:
        """Ensure workspace exists, create if not."""
        with self.get_session() as session:
            ws = session.query(Workspace).filter(Workspace.id == workspace_id).first()
            if not ws:
                ws = Workspace(
                    id=workspace_id,
                    organization_id=organization_id,
                    name=name or workspace_id,
                )
                session.add(ws)
                session.flush()
                logger.info(f"Created workspace: {workspace_id}")
            return ws

    # Document operations
    def create_document(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        document_type: str = "other",
        organization_id: str = "default",
        workspace_id: Optional[str] = None,
        collection_id: Optional[str] = None,
        access_level: str = "internal",
        owner_id: Optional[str] = None,
        created_by: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Document:
        """Create a new document record with tenant isolation."""
        # Auto-create organization if it doesn't exist
        self.ensure_organization(organization_id)

        # Auto-create workspace if provided and doesn't exist
        if workspace_id:
            self.ensure_workspace(workspace_id, organization_id)

        with self.get_session() as session:
            doc = Document(
                id=document_id,
                filename=filename,
                file_path=file_path,
                document_type=DocumentType(document_type),
                status=DocumentStatus.PENDING,
                organization_id=organization_id,
                workspace_id=workspace_id,
                collection_id=collection_id,
                access_level=access_level,
                owner_id=owner_id,
                created_by=created_by,
                _metadata=metadata or {},
            )
            session.add(doc)
            session.flush()
            return doc

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        with self.get_session() as session:
            return session.query(Document).filter(Document.id == document_id).first()

    def update_document(
        self,
        document_id: str,
        **updates
    ) -> Optional[Document]:
        """Update document fields."""
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == document_id).first()
            if doc:
                for key, value in updates.items():
                    if key == "status" and isinstance(value, str):
                        value = DocumentStatus(value)
                    if key == "document_type" and isinstance(value, str):
                        value = DocumentType(value)
                    setattr(doc, key, value)
                doc.updated_at = datetime.utcnow()
            return doc

    def delete_document(self, document_id: str) -> bool:
        """Delete document and all related records."""
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == document_id).first()
            if doc:
                session.delete(doc)
                return True
            return False

    def list_documents(
        self,
        document_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """List documents with optional filters."""
        with self.get_session() as session:
            query = session.query(Document)

            if document_type:
                query = query.filter(Document.document_type == DocumentType(document_type))
            if status:
                query = query.filter(Document.status == DocumentStatus(status))

            query = query.order_by(Document.created_at.desc())
            return query.offset(offset).limit(limit).all()

    # Chunk operations
    def create_chunks(self, chunks: List[Dict[str, Any]]) -> List[Chunk]:
        """Bulk create chunks with tenant isolation."""
        with self.get_session() as session:
            chunk_records = []
            for chunk_data in chunks:
                chunk = Chunk(
                    id=chunk_data["id"],
                    document_id=chunk_data["document_id"],
                    organization_id=chunk_data.get("organization_id", "default"),
                    workspace_id=chunk_data.get("workspace_id"),
                    access_level=chunk_data.get("access_level", "internal"),
                    text=chunk_data["text"],
                    chunk_index=chunk_data["chunk_index"],
                    page_number=chunk_data.get("page_number"),
                    element_type=chunk_data.get("element_type"),
                    char_count=len(chunk_data["text"]),
                    word_count=len(chunk_data["text"].split()),
                    _metadata=chunk_data.get("metadata", {}),
                )
                session.add(chunk)
                chunk_records.append(chunk)
            session.flush()
            return chunk_records

    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a document."""
        with self.get_session() as session:
            return session.query(Chunk).filter(
                Chunk.document_id == document_id
            ).order_by(Chunk.chunk_index).all()

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Get chunks by their IDs."""
        with self.get_session() as session:
            return session.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()

    def mark_chunks_indexed(self, chunk_ids: List[str]) -> None:
        """Mark chunks as indexed in vector store."""
        with self.get_session() as session:
            session.query(Chunk).filter(Chunk.id.in_(chunk_ids)).update(
                {"is_indexed": True, "dense_vector_id": Chunk.id},
                synchronize_session=False
            )

    # Page operations
    def create_pages(self, pages: List[Dict[str, Any]]) -> List[Page]:
        """Bulk create page records with tenant isolation."""
        with self.get_session() as session:
            page_records = []
            for page_data in pages:
                page = Page(
                    id=page_data["id"],
                    document_id=page_data["document_id"],
                    organization_id=page_data.get("organization_id", "default"),
                    workspace_id=page_data.get("workspace_id"),
                    page_number=page_data["page_number"],
                    image_path=page_data.get("image_path"),
                    width=page_data.get("width"),
                    height=page_data.get("height"),
                    _metadata=page_data.get("metadata", {}),
                )
                session.add(page)
                page_records.append(page)
            session.flush()
            return page_records

    def get_pages_by_document(self, document_id: str) -> List[Page]:
        """Get all pages for a document."""
        with self.get_session() as session:
            return session.query(Page).filter(
                Page.document_id == document_id
            ).order_by(Page.page_number).all()

    def mark_pages_indexed(self, page_ids: List[str]) -> None:
        """Mark pages as indexed in vector store."""
        with self.get_session() as session:
            session.query(Page).filter(Page.id.in_(page_ids)).update(
                {"is_indexed": True, "colpali_vector_id": Page.id},
                synchronize_session=False
            )

    # =========================================================================
    # Processing Job operations
    # =========================================================================
    def create_processing_job(
        self,
        organization_id: str,
        job_type: str,
        document_id: Optional[str] = None,
        priority: int = 50,
        metadata: Optional[Dict] = None
    ) -> ProcessingJob:
        """Create a new processing job."""
        with self.get_session() as session:
            job = ProcessingJob(
                organization_id=organization_id,
                document_id=document_id,
                job_type=job_type,
                status="pending",
                priority=priority,
                _metadata=metadata or {},
            )
            session.add(job)
            session.flush()
            return job

    def update_processing_job(
        self,
        job_id: int,
        status: Optional[str] = None,
        progress_percent: Optional[int] = None,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None,
        error_details: Optional[Dict] = None,
        result_summary: Optional[Dict] = None,
    ) -> Optional[ProcessingJob]:
        """Update processing job status and progress."""
        with self.get_session() as session:
            job = session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                if status:
                    job.status = status
                    if status == "processing" and not job.started_at:
                        job.started_at = datetime.utcnow()
                    elif status in ("completed", "failed"):
                        job.completed_at = datetime.utcnow()
                if progress_percent is not None:
                    job.progress_percent = progress_percent
                if current_step:
                    job.current_step = current_step
                if error_message:
                    job.error_message = error_message
                if error_details:
                    job.error_details = error_details
                if result_summary:
                    job.result_summary = result_summary
            return job

    def get_processing_jobs(
        self,
        organization_id: Optional[str] = None,
        document_id: Optional[str] = None,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ProcessingJob]:
        """Get processing jobs with optional filters."""
        with self.get_session() as session:
            query = session.query(ProcessingJob)
            if organization_id:
                query = query.filter(ProcessingJob.organization_id == organization_id)
            if document_id:
                query = query.filter(ProcessingJob.document_id == document_id)
            if status:
                query = query.filter(ProcessingJob.status == status)
            if job_type:
                query = query.filter(ProcessingJob.job_type == job_type)
            return query.order_by(ProcessingJob.created_at.desc()).limit(limit).all()

    # =========================================================================
    # Quality Report operations
    # =========================================================================
    def create_quality_report(
        self,
        document_id: str,
        organization_id: str,
        overall_score: float,
        quality_level: str,
        text_extraction_score: Optional[float] = None,
        ocr_artifact_score: Optional[float] = None,
        formatting_score: Optional[float] = None,
        coherence_score: Optional[float] = None,
        issues_found: Optional[List] = None,
        recommendations: Optional[List] = None,
        recommended_pipeline: Optional[str] = None,
    ) -> QualityReport:
        """Create a quality report for a document."""
        with self.get_session() as session:
            report = QualityReport(
                document_id=document_id,
                organization_id=organization_id,
                overall_score=overall_score,
                quality_level=QualityLevel(quality_level),
                text_extraction_score=text_extraction_score,
                ocr_artifact_score=ocr_artifact_score,
                formatting_score=formatting_score,
                coherence_score=coherence_score,
                issues_found=issues_found or [],
                recommendations=recommendations or [],
                recommended_pipeline=recommended_pipeline,
            )
            session.add(report)
            session.flush()
            return report

    def get_quality_report(self, document_id: str) -> Optional[QualityReport]:
        """Get quality report for a document."""
        with self.get_session() as session:
            return session.query(QualityReport).filter(
                QualityReport.document_id == document_id
            ).order_by(QualityReport.created_at.desc()).first()

    def get_quality_reports_by_level(
        self,
        organization_id: str,
        quality_level: str,
        limit: int = 100
    ) -> List[QualityReport]:
        """Get quality reports filtered by level."""
        with self.get_session() as session:
            return session.query(QualityReport).filter(
                QualityReport.organization_id == organization_id,
                QualityReport.quality_level == QualityLevel(quality_level)
            ).order_by(QualityReport.created_at.desc()).limit(limit).all()

    # =========================================================================
    # Extracted Table operations
    # =========================================================================
    def create_extracted_tables(self, tables: List[Dict[str, Any]]) -> List[ExtractedTable]:
        """Bulk create extracted table records."""
        with self.get_session() as session:
            table_records = []
            for table_data in tables:
                table = ExtractedTable(
                    id=table_data["id"],
                    document_id=table_data["document_id"],
                    chunk_id=table_data.get("chunk_id"),
                    organization_id=table_data.get("organization_id", "default"),
                    page_number=table_data.get("page_number"),
                    table_index=table_data.get("table_index"),
                    html_content=table_data.get("html_content"),
                    markdown_content=table_data.get("markdown_content"),
                    structured_data=table_data.get("structured_data"),
                    description=table_data.get("description"),
                    num_rows=table_data.get("num_rows"),
                    num_cols=table_data.get("num_cols"),
                    _metadata=table_data.get("metadata", {}),
                )
                session.add(table)
                table_records.append(table)
            session.flush()
            return table_records

    def get_tables_by_document(self, document_id: str) -> List[ExtractedTable]:
        """Get all extracted tables for a document."""
        with self.get_session() as session:
            return session.query(ExtractedTable).filter(
                ExtractedTable.document_id == document_id
            ).order_by(ExtractedTable.page_number, ExtractedTable.table_index).all()

    def get_tables_by_organization(
        self,
        organization_id: str,
        limit: int = 100
    ) -> List[ExtractedTable]:
        """Get extracted tables for an organization."""
        with self.get_session() as session:
            return session.query(ExtractedTable).filter(
                ExtractedTable.organization_id == organization_id
            ).order_by(ExtractedTable.created_at.desc()).limit(limit).all()

    # =========================================================================
    # Document Relationship operations
    # =========================================================================
    def create_document_relationship(
        self,
        organization_id: str,
        source_document_id: str,
        target_document_id: str,
        relation_type: str,
        source_chunk_id: Optional[str] = None,
        target_chunk_id: Optional[str] = None,
        confidence: float = 1.0,
        evidence_text: Optional[str] = None,
        detected_by: str = "rule",
        metadata: Optional[Dict] = None,
    ) -> DocumentRelationship:
        """Create a document relationship."""
        with self.get_session() as session:
            relationship = DocumentRelationship(
                organization_id=organization_id,
                source_document_id=source_document_id,
                source_chunk_id=source_chunk_id,
                target_document_id=target_document_id,
                target_chunk_id=target_chunk_id,
                relation_type=RelationType(relation_type),
                confidence=confidence,
                evidence_text=evidence_text,
                detected_by=detected_by,
                _metadata=metadata or {},
            )
            session.add(relationship)
            session.flush()
            return relationship

    def get_document_relationships(
        self,
        document_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> List[DocumentRelationship]:
        """Get relationships for a document."""
        with self.get_session() as session:
            if direction == "outgoing":
                return session.query(DocumentRelationship).filter(
                    DocumentRelationship.source_document_id == document_id
                ).all()
            elif direction == "incoming":
                return session.query(DocumentRelationship).filter(
                    DocumentRelationship.target_document_id == document_id
                ).all()
            else:
                from sqlalchemy import or_
                return session.query(DocumentRelationship).filter(
                    or_(
                        DocumentRelationship.source_document_id == document_id,
                        DocumentRelationship.target_document_id == document_id
                    )
                ).all()

    def get_related_documents(
        self,
        document_id: str,
        relation_types: Optional[List[str]] = None,
    ) -> List[str]:
        """Get IDs of related documents."""
        with self.get_session() as session:
            from sqlalchemy import or_
            query = session.query(DocumentRelationship).filter(
                or_(
                    DocumentRelationship.source_document_id == document_id,
                    DocumentRelationship.target_document_id == document_id
                )
            )
            if relation_types:
                query = query.filter(
                    DocumentRelationship.relation_type.in_(
                        [RelationType(rt) for rt in relation_types]
                    )
                )

            related_ids = set()
            for rel in query.all():
                if rel.source_document_id == document_id:
                    related_ids.add(rel.target_document_id)
                else:
                    related_ids.add(rel.source_document_id)
            return list(related_ids)

    # Search support
    def get_document_ids_by_metadata(
        self,
        filters: Dict[str, Any]
    ) -> List[str]:
        """Get document IDs matching metadata filters."""
        with self.get_session() as session:
            query = session.query(Document.id)

            for key, value in filters.items():
                # Use JSONB containment operator
                query = query.filter(
                    Document._metadata.op("@>")(f'{{{key!r}: {value!r}}}')
                )

            return [doc_id for (doc_id,) in query.all()]

    # Stats
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_session() as session:
            total_docs = session.query(Document).count()
            total_chunks = session.query(Chunk).count()
            total_pages = session.query(Page).count()
            total_jobs = session.query(ProcessingJob).count()
            total_quality_reports = session.query(QualityReport).count()
            total_tables = session.query(ExtractedTable).count()
            total_relationships = session.query(DocumentRelationship).count()

            docs_by_type = {}
            for doc_type in DocumentType:
                count = session.query(Document).filter(
                    Document.document_type == doc_type
                ).count()
                if count > 0:
                    docs_by_type[doc_type.value] = count

            docs_by_status = {}
            for status in DocumentStatus:
                count = session.query(Document).filter(
                    Document.status == status
                ).count()
                if count > 0:
                    docs_by_status[status.value] = count

            jobs_by_status = {}
            for status in ["pending", "processing", "completed", "failed"]:
                count = session.query(ProcessingJob).filter(
                    ProcessingJob.status == status
                ).count()
                if count > 0:
                    jobs_by_status[status] = count

            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_pages": total_pages,
                "total_processing_jobs": total_jobs,
                "total_quality_reports": total_quality_reports,
                "total_extracted_tables": total_tables,
                "total_relationships": total_relationships,
                "documents_by_type": docs_by_type,
                "documents_by_status": docs_by_status,
                "jobs_by_status": jobs_by_status,
            }


# Example usage
if __name__ == "__main__":
    # Initialize store
    store = MetadataStore(
        database_url="postgresql://user:password@localhost:5432/rag_db"
    )

    # Create tables
    store.create_tables()

    # Create a document
    doc = store.create_document(
        document_id="test_doc_123",
        filename="contract.pdf",
        file_path="/documents/contract.pdf",
        document_type="contract",
        metadata={"client": "Acme Corp", "value": 50000}
    )

    print(f"Created document: {doc.id}")

    # Get stats
    stats = store.get_stats()
    print(f"Stats: {stats}")
