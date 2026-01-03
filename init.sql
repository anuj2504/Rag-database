-- ============================================================================
-- Enterprise RAG Database Schema
-- PostgreSQL 14+
-- ============================================================================
-- This schema supports:
-- - Multi-tenant isolation (organization_id, workspace_id)
-- - Document, chunk, and page management
-- - Quality tracking and processing jobs
-- - Document relationships (graph edges)
-- - Acronym/terminology database
-- - Search analytics
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";      -- Fuzzy text search
CREATE EXTENSION IF NOT EXISTS "btree_gin";    -- Better GIN indexes

-- ============================================================================
-- ENUMS
-- ============================================================================

CREATE TYPE document_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed',
    'archived'
);

CREATE TYPE document_type AS ENUM (
    'contract',
    'letter',
    'invoice',
    'code',
    'report',
    'form',
    'document',
    'financial',
    'legal',
    'technical',
    'other'
);

CREATE TYPE access_level AS ENUM (
    'public',
    'internal',
    'restricted',
    'confidential'
);

CREATE TYPE quality_level AS ENUM (
    'high',
    'medium',
    'low',
    'garbage'
);

CREATE TYPE chunk_level AS ENUM (
    'document',
    'section',
    'paragraph',
    'sentence'
);

CREATE TYPE relation_type AS ENUM (
    'amends',
    'supersedes',
    'references',
    'incorporates',
    'exhibit_of',
    'schedule_of',
    'related_to',
    'parent_of',
    'child_of'
);

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Organizations (Tenants)
CREATE TABLE organizations (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_organizations_name ON organizations(name);
CREATE INDEX idx_organizations_active ON organizations(is_active);

-- Workspaces (within organizations)
CREATE TABLE workspaces (
    id VARCHAR(255) PRIMARY KEY,
    organization_id VARCHAR(255) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,

    UNIQUE(organization_id, name)
);

CREATE INDEX idx_workspaces_org ON workspaces(organization_id);
CREATE INDEX idx_workspaces_active ON workspaces(is_active);

-- Collections (document groups within workspaces)
CREATE TABLE collections (
    id VARCHAR(255) PRIMARY KEY,
    organization_id VARCHAR(255) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id VARCHAR(255) REFERENCES workspaces(id) ON DELETE SET NULL,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_collections_org ON collections(organization_id);
CREATE INDEX idx_collections_workspace ON collections(workspace_id);

-- ============================================================================
-- DOCUMENTS TABLE
-- ============================================================================

CREATE TABLE documents (
    id VARCHAR(255) PRIMARY KEY,

    -- Multi-tenant isolation (CRITICAL)
    organization_id VARCHAR(255) NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id VARCHAR(255) REFERENCES workspaces(id) ON DELETE SET NULL,
    collection_id VARCHAR(255) REFERENCES collections(id) ON DELETE SET NULL,
    access_level access_level DEFAULT 'internal',

    -- File info
    filename VARCHAR(500) NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    file_extension VARCHAR(20),
    mime_type VARCHAR(100),
    file_hash VARCHAR(64),  -- SHA-256 for deduplication

    -- Document classification
    document_type document_type DEFAULT 'other',
    document_subtype VARCHAR(100),  -- e.g., 'irc_code', 'building_code'

    -- Processing status
    status document_status DEFAULT 'pending',
    quality_level quality_level,
    quality_score FLOAT,

    -- Counts
    total_chunks INTEGER DEFAULT 0,
    total_pages INTEGER,
    total_tokens INTEGER,

    -- Content info
    languages VARCHAR(10)[] DEFAULT ARRAY['eng'],
    title TEXT,
    summary TEXT,

    -- Ownership
    owner_id VARCHAR(255),
    created_by VARCHAR(255),
    updated_by VARCHAR(255),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    indexed_at TIMESTAMP,

    -- Compliance
    classification VARCHAR(100),  -- PII, PHI, financial, etc.
    retention_policy VARCHAR(100),
    retention_until DATE,
    legal_hold BOOLEAN DEFAULT FALSE,

    -- Domain-specific metadata (flexible JSONB)
    metadata JSONB DEFAULT '{}',

    -- Source tracking
    source_url TEXT,
    source_system VARCHAR(100),
    external_id VARCHAR(255)
);

-- Critical tenant isolation indexes
CREATE INDEX idx_documents_org ON documents(organization_id);
CREATE INDEX idx_documents_org_workspace ON documents(organization_id, workspace_id);
CREATE INDEX idx_documents_org_collection ON documents(organization_id, collection_id);
CREATE INDEX idx_documents_org_access ON documents(organization_id, access_level);
CREATE INDEX idx_documents_org_status ON documents(organization_id, status);

-- Document lookup indexes
CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_quality ON documents(quality_level);
CREATE INDEX idx_documents_created ON documents(created_at DESC);
CREATE INDEX idx_documents_hash ON documents(file_hash);

-- JSONB metadata index (for filtering)
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);

-- ============================================================================
-- CHUNKS TABLE
-- ============================================================================

CREATE TABLE chunks (
    id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Multi-tenant isolation (denormalized for query performance)
    organization_id VARCHAR(255) NOT NULL,
    workspace_id VARCHAR(255),
    access_level access_level DEFAULT 'internal',

    -- Content
    text TEXT NOT NULL,

    -- Position info
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    char_start INTEGER,
    char_end INTEGER,

    -- Hierarchy
    chunk_level chunk_level DEFAULT 'paragraph',
    parent_chunk_id VARCHAR(255) REFERENCES chunks(id) ON DELETE SET NULL,
    section_id VARCHAR(255),
    section_title TEXT,
    section_number VARCHAR(50),

    -- Token info
    token_count INTEGER,
    char_count INTEGER,
    word_count INTEGER,

    -- Element type (from Unstructured)
    element_type VARCHAR(50),  -- Title, NarrativeText, Table, ListItem, etc.

    -- Vector store references
    dense_vector_id VARCHAR(255),
    colpali_vector_id VARCHAR(255),
    bm25_indexed BOOLEAN DEFAULT FALSE,

    -- Processing flags
    is_indexed BOOLEAN DEFAULT FALSE,
    needs_reindex BOOLEAN DEFAULT FALSE,

    -- Metadata
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Critical tenant isolation indexes for chunks
CREATE INDEX idx_chunks_org ON chunks(organization_id);
CREATE INDEX idx_chunks_org_workspace ON chunks(organization_id, workspace_id);
CREATE INDEX idx_chunks_org_access ON chunks(organization_id, access_level);

-- Chunk lookup indexes
CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_page ON chunks(page_number);
CREATE INDEX idx_chunks_level ON chunks(chunk_level);
CREATE INDEX idx_chunks_indexed ON chunks(is_indexed);
CREATE INDEX idx_chunks_element_type ON chunks(element_type);
CREATE INDEX idx_chunks_section ON chunks(section_id);

-- Composite index for common query pattern
CREATE INDEX idx_chunks_doc_index ON chunks(document_id, chunk_index);

-- JSONB metadata index
CREATE INDEX idx_chunks_metadata ON chunks USING GIN (metadata);

-- ============================================================================
-- PAGES TABLE (for ColPali visual retrieval)
-- ============================================================================

CREATE TABLE pages (
    id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Multi-tenant isolation
    organization_id VARCHAR(255) NOT NULL,
    workspace_id VARCHAR(255),

    -- Page info
    page_number INTEGER NOT NULL,
    image_path TEXT,

    -- Dimensions
    width INTEGER,
    height INTEGER,
    dpi INTEGER,

    -- Vector store reference
    colpali_vector_id VARCHAR(255),
    is_indexed BOOLEAN DEFAULT FALSE,

    -- Quality info
    ocr_confidence FLOAT,
    has_tables BOOLEAN DEFAULT FALSE,
    has_figures BOOLEAN DEFAULT FALSE,

    -- Metadata
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pages_org ON pages(organization_id);
CREATE INDEX idx_pages_document ON pages(document_id);
CREATE INDEX idx_pages_indexed ON pages(is_indexed);
CREATE INDEX idx_pages_doc_page ON pages(document_id, page_number);

-- ============================================================================
-- TABLES (extracted tabular data)
-- ============================================================================

CREATE TABLE extracted_tables (
    id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id VARCHAR(255) REFERENCES chunks(id) ON DELETE SET NULL,

    -- Multi-tenant isolation
    organization_id VARCHAR(255) NOT NULL,

    -- Table info
    page_number INTEGER,
    table_index INTEGER,

    -- Content
    html_content TEXT,
    markdown_content TEXT,
    structured_data JSONB,  -- Parsed table data

    -- Semantic description (for retrieval)
    description TEXT,

    -- Dimensions
    num_rows INTEGER,
    num_cols INTEGER,

    -- Vector references
    structured_vector_id VARCHAR(255),
    semantic_vector_id VARCHAR(255),

    -- Metadata
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tables_org ON extracted_tables(organization_id);
CREATE INDEX idx_tables_document ON extracted_tables(document_id);
CREATE INDEX idx_tables_chunk ON extracted_tables(chunk_id);

-- ============================================================================
-- DOCUMENT RELATIONSHIPS (Graph edges)
-- ============================================================================

CREATE TABLE document_relationships (
    id SERIAL PRIMARY KEY,

    -- Multi-tenant isolation
    organization_id VARCHAR(255) NOT NULL,

    -- Source document
    source_document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_chunk_id VARCHAR(255) REFERENCES chunks(id) ON DELETE SET NULL,

    -- Target document
    target_document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_chunk_id VARCHAR(255) REFERENCES chunks(id) ON DELETE SET NULL,

    -- Relationship info
    relation_type relation_type NOT NULL,
    confidence FLOAT DEFAULT 1.0,

    -- Evidence
    evidence_text TEXT,
    detected_by VARCHAR(50),  -- 'rule', 'model', 'manual'

    -- Metadata
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(source_document_id, target_document_id, relation_type)
);

CREATE INDEX idx_relationships_org ON document_relationships(organization_id);
CREATE INDEX idx_relationships_source ON document_relationships(source_document_id);
CREATE INDEX idx_relationships_target ON document_relationships(target_document_id);
CREATE INDEX idx_relationships_type ON document_relationships(relation_type);

-- ============================================================================
-- ACRONYM/TERMINOLOGY DATABASE
-- ============================================================================

CREATE TABLE acronyms (
    id SERIAL PRIMARY KEY,

    -- Multi-tenant (can be global or org-specific)
    organization_id VARCHAR(255),  -- NULL = global

    -- Acronym info
    acronym VARCHAR(50) NOT NULL,
    expansion TEXT NOT NULL,

    -- Domain context
    domain VARCHAR(50),  -- legal, tax, building, financial, general
    subdomain VARCHAR(100),

    -- Usage info
    priority INTEGER DEFAULT 50,  -- Higher = more likely
    is_preferred BOOLEAN DEFAULT FALSE,

    -- Metadata
    description TEXT,
    source VARCHAR(255),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(organization_id, acronym, domain, expansion)
);

CREATE INDEX idx_acronyms_acronym ON acronyms(UPPER(acronym));
CREATE INDEX idx_acronyms_domain ON acronyms(domain);
CREATE INDEX idx_acronyms_org ON acronyms(organization_id);

-- Insert common acronyms
INSERT INTO acronyms (acronym, expansion, domain, priority, description) VALUES
-- Tax/IRC
('IRC', 'Internal Revenue Code', 'tax', 90, 'US federal tax code'),
('IRS', 'Internal Revenue Service', 'tax', 90, 'US tax authority'),
('FICA', 'Federal Insurance Contributions Act', 'tax', 80, 'Social Security and Medicare taxes'),
('FUTA', 'Federal Unemployment Tax Act', 'tax', 70, 'Federal unemployment tax'),

-- Building Codes
('IRC', 'International Residential Code', 'building', 90, 'Residential building code'),
('IBC', 'International Building Code', 'building', 90, 'Commercial building code'),
('IFC', 'International Fire Code', 'building', 80, 'Fire safety code'),
('IMC', 'International Mechanical Code', 'building', 70, 'HVAC code'),
('IPC', 'International Plumbing Code', 'building', 70, 'Plumbing code'),

-- Legal
('LLC', 'Limited Liability Company', 'legal', 90, 'Business entity type'),
('NDA', 'Non-Disclosure Agreement', 'legal', 85, 'Confidentiality agreement'),
('MSA', 'Master Services Agreement', 'legal', 80, 'Umbrella service contract'),
('SOW', 'Statement of Work', 'legal', 75, 'Project scope document'),
('IP', 'Intellectual Property', 'legal', 85, 'Patents, trademarks, copyrights'),

-- Financial
('GAAP', 'Generally Accepted Accounting Principles', 'financial', 90, 'US accounting standards'),
('EBITDA', 'Earnings Before Interest, Taxes, Depreciation, and Amortization', 'financial', 85, 'Profitability metric'),
('ROI', 'Return on Investment', 'financial', 80, 'Investment return metric'),
('YoY', 'Year over Year', 'financial', 75, 'Annual comparison'),
('QoQ', 'Quarter over Quarter', 'financial', 70, 'Quarterly comparison');

-- ============================================================================
-- PROCESSING JOBS
-- ============================================================================

CREATE TABLE processing_jobs (
    id SERIAL PRIMARY KEY,

    -- Multi-tenant
    organization_id VARCHAR(255) NOT NULL,

    document_id VARCHAR(255) REFERENCES documents(id) ON DELETE SET NULL,

    -- Job info
    job_type VARCHAR(50) NOT NULL,  -- ingestion, reindex, delete, export
    status VARCHAR(20) DEFAULT 'pending',
    priority INTEGER DEFAULT 50,

    -- Timing
    scheduled_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Progress
    progress_percent INTEGER DEFAULT 0,
    current_step VARCHAR(100),

    -- Results
    error_message TEXT,
    error_details JSONB,
    result_summary JSONB,

    -- Retry info
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- Worker info
    worker_id VARCHAR(100),

    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_jobs_org ON processing_jobs(organization_id);
CREATE INDEX idx_jobs_status ON processing_jobs(status);
CREATE INDEX idx_jobs_document ON processing_jobs(document_id);
CREATE INDEX idx_jobs_type_status ON processing_jobs(job_type, status);

-- ============================================================================
-- QUALITY REPORTS
-- ============================================================================

CREATE TABLE quality_reports (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Multi-tenant
    organization_id VARCHAR(255) NOT NULL,

    -- Quality scores (0-100)
    overall_score FLOAT NOT NULL,
    text_extraction_score FLOAT,
    ocr_artifact_score FLOAT,
    formatting_score FLOAT,
    coherence_score FLOAT,

    -- Quality level
    quality_level quality_level NOT NULL,

    -- Details
    issues_found JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',

    -- Processing route
    recommended_pipeline VARCHAR(50),  -- hierarchical, standard, simple

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_quality_org ON quality_reports(organization_id);
CREATE INDEX idx_quality_document ON quality_reports(document_id);
CREATE INDEX idx_quality_level ON quality_reports(quality_level);

-- ============================================================================
-- SEARCH ANALYTICS
-- ============================================================================

CREATE TABLE search_queries (
    id SERIAL PRIMARY KEY,

    -- Multi-tenant
    organization_id VARCHAR(255) NOT NULL,
    workspace_id VARCHAR(255),
    user_id VARCHAR(255),

    -- Query info
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64),  -- For aggregation

    -- Analysis
    query_type VARCHAR(50),  -- broad, precise, reference, comparative
    detected_domain VARCHAR(50),

    -- Results
    total_results INTEGER,
    top_result_score FLOAT,
    retrieval_methods TEXT[],

    -- Expansions
    acronyms_expanded TEXT[],

    -- Performance
    processing_time_ms INTEGER,
    fallback_triggered BOOLEAN DEFAULT FALSE,

    -- Feedback
    user_rating INTEGER,  -- 1-5
    result_clicked BOOLEAN,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_search_org ON search_queries(organization_id);
CREATE INDEX idx_search_hash ON search_queries(query_hash);
CREATE INDEX idx_search_created ON search_queries(created_at DESC);
CREATE INDEX idx_search_type ON search_queries(query_type);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Document overview with counts
CREATE OR REPLACE VIEW document_overview AS
SELECT
    d.id,
    d.organization_id,
    d.workspace_id,
    d.filename,
    d.document_type,
    d.status,
    d.quality_level,
    d.total_chunks,
    d.total_pages,
    d.created_at,
    COUNT(DISTINCT c.id) AS actual_chunks,
    COUNT(DISTINCT p.id) AS actual_pages,
    COUNT(DISTINCT r.id) AS relationship_count
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
LEFT JOIN pages p ON d.id = p.document_id
LEFT JOIN document_relationships r ON d.id = r.source_document_id OR d.id = r.target_document_id
GROUP BY d.id;

-- Organization statistics
CREATE OR REPLACE VIEW organization_stats AS
SELECT
    o.id AS organization_id,
    o.name AS organization_name,
    COUNT(DISTINCT d.id) AS document_count,
    COUNT(DISTINCT c.id) AS chunk_count,
    COUNT(DISTINCT p.id) AS page_count,
    SUM(d.file_size_bytes) AS total_size_bytes,
    COUNT(DISTINCT w.id) AS workspace_count
FROM organizations o
LEFT JOIN documents d ON o.id = d.organization_id
LEFT JOIN chunks c ON d.id = c.document_id
LEFT JOIN pages p ON d.id = p.document_id
LEFT JOIN workspaces w ON o.id = w.organization_id
GROUP BY o.id, o.name;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables
CREATE TRIGGER documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER chunks_updated_at BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER workspaces_updated_at BEFORE UPDATE ON workspaces
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Create default organization (for development)
INSERT INTO organizations (id, name, settings) VALUES
('org_default', 'Default Organization', '{"tier": "free"}')
ON CONFLICT (id) DO NOTHING;

-- Create default workspace
INSERT INTO workspaces (id, organization_id, name) VALUES
('ws_default', 'org_default', 'Default Workspace')
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- PERMISSIONS (for production, customize as needed)
-- ============================================================================

-- Example: Create read-only role for analytics
-- CREATE ROLE rag_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_readonly;

-- Example: Create application role
-- CREATE ROLE rag_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rag_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rag_app;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
