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
    CONTRACT = "contract"
    LETTER = "letter"
    INVOICE = "invoice"
    CODE = "code"
    REPORT = "report"
    FORM = "form"
    DOCUMENT = "document"
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    OTHER = "other"


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
    document_id = Column(String(255), ForeignKey("documents.id", ondelete="SET NULL"))

    job_type = Column(String(50))  # 'ingestion', 'reindex', 'delete'
    status = Column(String(20), default="pending")

    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)

    _metadata = Column("metadata", JSONB, default={})
    extra_metadata = synonym("_metadata")

    __table_args__ = (
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_document", "document_id"),
    )


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
                {"is_indexed": True, "vector_id": Page.id},
                synchronize_session=False
            )

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

            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_pages": total_pages,
                "documents_by_type": docs_by_type,
                "documents_by_status": docs_by_status,
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
