# Enterprise Document RAG System

A production-grade Retrieval-Augmented Generation (RAG) system designed for enterprise documents: contracts, letters, IRC codes, building codes, regulations, financial reports, and scanned PDFs.

**Built to handle real-world enterprise challenges:**
- Documents ranging from perfect PDFs to garbage scanned copies from 1995
- Domain-specific terminology and acronym confusion
- Complex document relationships and cross-references
- Tables with critical data that standard RAG ignores
- **Multi-tenant isolation** - organization data never mixes

## Core Features

- **Multi-Tenant Isolation**: Organization-level data separation with workspace support
- **Document Quality Detection**: Routes documents to appropriate pipelines based on quality scoring
- **Intelligent Chunking**: Chonkie-powered adaptive chunking (Token, Sentence, Semantic, SDPM)
- **Hierarchical Chunking**: Preserves document structure (sections, paragraphs, sentences)
- **Domain-Specific Metadata**: Custom schemas for contracts, IRC codes, building codes, financial reports
- **Table Extraction**: Dedicated pipeline with dual embedding (structured + semantic)
- **Acronym Expansion**: Context-aware term expansion (IRC = Internal Revenue Code OR International Residential Code)
- **Document Relationship Graph**: Tracks references, amendments, exhibits between documents
- **Enhanced Hybrid Search**: BM25 + Dense + ColPali with precision fallbacks and failure detection

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           ENTERPRISE RAG SYSTEM ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│    ┌─────────────────┐                                                                  │
│    │   API Client    │  X-Organization-ID: org_acme (REQUIRED)                         │
│    │   (FastAPI)     │  X-Workspace-ID: ws_legal (optional)                            │
│    └────────┬────────┘                                                                  │
│             │                                                                            │
│             ▼                                                                            │
│    ┌────────────────────────────────────────────────────────────────────────────────┐   │
│    │                         TENANT ISOLATION LAYER                                  │   │
│    │   • Validates organization_id on every request                                 │   │
│    │   • Injects tenant context into all operations                                 │   │
│    │   • Filters results to prevent cross-org data leakage                          │   │
│    └────────────────────────────────────────────────────────────────────────────────┘   │
│             │                                                                            │
│             ├───────────────────────┬───────────────────────┐                           │
│             ▼                       ▼                       ▼                           │
│    ┌────────────────┐     ┌────────────────┐     ┌────────────────┐                    │
│    │   INGESTION    │     │    SEARCH      │     │   MANAGEMENT   │                    │
│    │   /documents   │     │    /search     │     │   /documents   │                    │
│    │    /upload     │     │                │     │    /{id}       │                    │
│    └────────┬───────┘     └────────┬───────┘     └────────────────┘                    │
│             │                       │                                                    │
└─────────────┼───────────────────────┼────────────────────────────────────────────────────┘
              │                       │
              ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION PIPELINE                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  Document (PDF, DOCX, Scanned)                                                          │
│         │                                                                               │
│         ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        1. DOCUMENT PROCESSING (Unstructured.io)                   │  │
│  │   • PDF parsing / OCR for scanned docs                                           │  │
│  │   • Layout detection (titles, paragraphs, tables, lists)                         │  │
│  │   • Element extraction with position information                                  │  │
│  └──────────────────────────────────────┬───────────────────────────────────────────┘  │
│                                         │                                               │
│                                         ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        2. QUALITY DETECTION                                       │  │
│  │   Score components:                                                               │  │
│  │   • Text extraction quality (0-100)                                              │  │
│  │   • OCR artifact detection (0-100)                                               │  │
│  │   • Formatting consistency (0-100)                                               │  │
│  │   • Content coherence (0-100)                                                    │  │
│  │                                                                                   │  │
│  │   Quality Levels → Pipeline Routing:                                             │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────┐    │  │
│  │   │ HIGH (>80)   → Hierarchical chunking + full metadata extraction        │    │  │
│  │   │ MEDIUM (50-80) → Standard chunking + basic metadata                     │    │  │
│  │   │ LOW (25-50)  → Simple chunking + minimal processing                     │    │  │
│  │   │ GARBAGE (<25) → Token-based chunking + visual embedding (ColPali)       │    │  │
│  │   └─────────────────────────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────┬───────────────────────────────────────────┘  │
│                                         │                                               │
│         ┌───────────────────────────────┼───────────────────────────────┐              │
│         ▼                               ▼                               ▼              │
│  ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐        │
│  │  3A. CHUNKING   │          │ 3B. METADATA    │          │ 3C. TABLE       │        │
│  │  (Chonkie)      │          │ EXTRACTION      │          │ EXTRACTION      │        │
│  │                 │          │                 │          │                 │        │
│  │ Strategies:     │          │ Domain schemas: │          │ • HTML/Markdown │        │
│  │ • Token         │          │ • Contracts     │          │ • Structured    │        │
│  │ • Sentence      │          │ • IRC/Tax       │          │   JSON          │        │
│  │ • Semantic      │          │ • Building Code │          │ • Semantic      │        │
│  │ • SDPM (best)   │          │ • Financial     │          │   description   │        │
│  │                 │          │                 │          │                 │        │
│  │ Hierarchy:      │          │ Extracted:      │          │ Dual embedding: │        │
│  │ Doc→Section→    │          │ • Parties       │          │ • Structured    │        │
│  │ Paragraph→      │          │ • Dates         │          │ • Semantic      │        │
│  │ Sentence        │          │ • Clauses       │          │                 │        │
│  └────────┬────────┘          └────────┬────────┘          └────────┬────────┘        │
│           │                            │                            │                  │
│           └────────────────────────────┼────────────────────────────┘                  │
│                                        ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        4. ACRONYM EXPANSION                                       │  │
│  │   Domain-aware term detection:                                                    │  │
│  │   • "IRC" in tax context → "Internal Revenue Code"                               │  │
│  │   • "IRC" in building context → "International Residential Code"                 │  │
│  │   • Custom org-specific acronyms supported                                       │  │
│  └──────────────────────────────────────┬───────────────────────────────────────────┘  │
│                                         │                                               │
│                                         ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        5. DOCUMENT GRAPH                                          │  │
│  │   Relationship detection:                                                         │  │
│  │   • amends, supersedes, references, incorporates                                 │  │
│  │   • exhibit_of, schedule_of, parent_of, child_of                                 │  │
│  │   Pattern matching: "Amendment to Agreement dated..."                            │  │
│  └──────────────────────────────────────┬───────────────────────────────────────────┘  │
│                                         │                                               │
│         ┌───────────────────────────────┼───────────────────────────────┐              │
│         ▼                               ▼                               ▼              │
│  ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐        │
│  │  6A. BM25       │          │ 6B. DENSE       │          │ 6C. ColPali     │        │
│  │  INDEXING       │          │ EMBEDDINGS      │          │ EMBEDDINGS      │        │
│  │                 │          │                 │          │                 │        │
│  │ In-memory or    │          │ Model:          │          │ Vision-language │        │
│  │ Elasticsearch   │          │ BGE-base-en-v1.5│          │ model for       │        │
│  │                 │          │ (768 dim)       │          │ visual docs     │        │
│  │ Payload:        │          │                 │          │                 │        │
│  │ • org_id ✓      │          │ Payload:        │          │ Payload:        │        │
│  │ • workspace_id  │          │ • org_id ✓      │          │ • org_id ✓      │        │
│  │ • access_level  │          │ • workspace_id  │          │ • workspace_id  │        │
│  │ • document_id   │          │ • access_level  │          │ • page_number   │        │
│  │ • text          │          │ • document_id   │          │ • document_id   │        │
│  └────────┬────────┘          └────────┬────────┘          └────────┬────────┘        │
│           │                            │                            │                  │
│           ▼                            ▼                            ▼                  │
│    ┌───────────┐              ┌──────────────┐              ┌──────────────┐           │
│    │  BM25     │              │   Qdrant     │              │   Qdrant     │           │
│    │  Index    │              │   (dense)    │              │  (colpali)   │           │
│    │  (pickle) │              │  collection  │              │  multi-vec   │           │
│    └───────────┘              └──────────────┘              └──────────────┘           │
│                                                                                         │
│    ┌────────────────────────────────────────────────────────────────────────────────┐   │
│    │                         PostgreSQL (Metadata)                                   │   │
│    │   organizations | workspaces | documents | chunks | pages |                    │   │
│    │   document_relationships | quality_reports | search_queries                    │   │
│    └────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              RETRIEVAL PIPELINE                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  Query: "What is the payment in Table 3 per IRC Section 199A?"                         │
│         │                                                                               │
│         ▼                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        1. QUERY ANALYSIS                                          │  │
│  │                                                                                   │  │
│  │   Query Type Detection:                                                           │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────┐    │  │
│  │   │ BROAD      → General overview questions                                 │    │  │
│  │   │ PRECISE    → Specific data points                                       │    │  │
│  │   │ REFERENCE  → Looking for specific section/table                         │    │  │
│  │   │ COMPARATIVE → Comparing entities                                        │    │  │
│  │   │ TEMPORAL   → Time-based queries                                         │    │  │
│  │   └─────────────────────────────────────────────────────────────────────────┘    │  │
│  │                                                                                   │  │
│  │   Precision Detection: "Table 3" → trigger exact BM25 match                      │  │
│  │   Acronym Expansion: "IRC" → "Internal Revenue Code" (tax context detected)      │  │
│  │   Complexity → PRECISE → sentence-level retrieval                                │  │
│  └──────────────────────────────────────┬───────────────────────────────────────────┘  │
│                                         │                                               │
│                                         ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        2. PARALLEL HYBRID SEARCH                                  │  │
│  │                                                                                   │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────┐    │  │
│  │   │  All searches include: organization_id = "org_acme" (from header)       │    │  │
│  │   └─────────────────────────────────────────────────────────────────────────┘    │  │
│  │                                                                                   │  │
│  │         ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │  │
│  │         │    BM25     │     │   Dense     │     │   ColPali   │                  │  │
│  │         │  (keyword)  │     │ (semantic)  │     │  (visual)   │                  │  │
│  │         │  Weight: 0.3│     │ Weight: 0.5 │     │ Weight: 0.2 │                  │  │
│  │         └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                  │  │
│  │                │ Rank List         │ Rank List         │ Rank List              │  │
│  │                └───────────────────┼───────────────────┘                        │  │
│  │                                    ▼                                             │  │
│  │                    ┌───────────────────────────┐                                 │  │
│  │                    │  Reciprocal Rank Fusion   │                                 │  │
│  │                    │  score = Σ(weight / (k + rank))                             │  │
│  │                    │  k = 60 (default)         │                                 │  │
│  │                    └───────────────────────────┘                                 │  │
│  └──────────────────────────────────────┬───────────────────────────────────────────┘  │
│                                         │                                               │
│                                         ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        3. PRECISION FALLBACK                                      │  │
│  │                                                                                   │  │
│  │   If query contains table/section references ("Table 3", "Section 5.2"):         │  │
│  │   • Boost exact BM25 matches for reference terms                                 │  │
│  │   • Add +0.5 score to precision matches                                          │  │
│  │                                                                                   │  │
│  │   If results have low confidence (<0.5):                                         │  │
│  │   • try_keyword_search → Pure BM25 fallback                                      │  │
│  │   • expand_search → Remove filters, broaden scope                                │  │
│  └──────────────────────────────────────┬───────────────────────────────────────────┘  │
│                                         │                                               │
│                                         ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        4. GRAPH AUGMENTATION                                      │  │
│  │                                                                                   │  │
│  │   For each result, find related documents:                                       │  │
│  │   • Amendments to retrieved contract                                             │  │
│  │   • Exhibits referenced in retrieved section                                     │  │
│  │   • Parent/child documents                                                       │  │
│  │                                                                                   │  │
│  │   Add augmented results with relation_path for explainability                    │  │
│  └──────────────────────────────────────┬───────────────────────────────────────────┘  │
│                                         │                                               │
│                                         ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                        5. RESPONSE                                                │  │
│  │                                                                                   │  │
│  │   {                                                                               │  │
│  │     "query": "What is the payment in Table 3 per IRC Section 199A?",            │  │
│  │     "organization_id": "org_acme",                                               │  │
│  │     "total_results": 5,                                                          │  │
│  │     "results": [                                                                 │  │
│  │       {                                                                          │  │
│  │         "id": "doc_123_chunk_45",                                                │  │
│  │         "score": 0.89,                                                           │  │
│  │         "text": "Table 3: Payment Schedule per IRC Section 199A...",            │  │
│  │         "bm25_rank": 1,                                                          │  │
│  │         "dense_rank": 3,                                                         │  │
│  │         "metadata": {"document_type": "contract", "section": "3.2"}             │  │
│  │       }                                                                          │  │
│  │     ]                                                                            │  │
│  │   }                                                                              │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Multi-Tenant Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              MULTI-TENANT ISOLATION                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  Organization: Acme Corp (org_acme)     Organization: Beta Inc (org_beta)              │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐            │
│  │  Workspace: Legal (ws_legal)    │    │  Workspace: Finance (ws_fin)   │            │
│  │  ┌───────────────────────────┐  │    │  ┌───────────────────────────┐  │            │
│  │  │ Collection: Contracts     │  │    │  │ Collection: Reports       │  │            │
│  │  │ • contract_001.pdf        │  │    │  │ • q4_2024.pdf             │  │            │
│  │  │ • contract_002.pdf        │  │    │  │ • annual_2024.pdf         │  │            │
│  │  └───────────────────────────┘  │    │  └───────────────────────────┘  │            │
│  │  ┌───────────────────────────┐  │    └─────────────────────────────────┘            │
│  │  │ Collection: NDAs          │  │                                                    │
│  │  │ • nda_vendor_a.pdf        │  │    These organizations' data NEVER mixes:         │
│  │  └───────────────────────────┘  │    • Separate rows in PostgreSQL                  │
│  └─────────────────────────────────┘    • Separate payloads in Qdrant                  │
│                                         • Separate entries in BM25 index               │
│                                                                                         │
│  Every query, every ingestion, every operation includes:                               │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│  │  filters: {                                                                        │ │
│  │    "organization_id": "org_acme",    ← MANDATORY (from X-Organization-ID header)  │ │
│  │    "workspace_id": "ws_legal",        ← Optional (from X-Workspace-ID header)     │ │
│  │    "access_level": ["public", "internal"]  ← Based on user permissions            │ │
│  │  }                                                                                 │ │
│  └───────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Document Processing | Unstructured.io | OCR, parsing, layout detection |
| Quality Detection | Custom | Score documents, route to appropriate pipeline |
| Intelligent Chunking | Chonkie | Token, Sentence, Semantic, SDPM chunking strategies |
| Hierarchical Chunking | Custom | Preserve document structure (section/paragraph/sentence) |
| Domain Metadata | Custom | Extract contract terms, IRC sections, financial metrics |
| Table Extraction | Custom | Detect, parse, dual-embed tabular data |
| Document Graph | Custom | Track relationships, amendments, references |
| Acronym Database | Custom | Context-aware term expansion |
| Vector Database | Qdrant | Dense & multi-vector storage |
| Metadata Database | PostgreSQL | Documents, chunks, relationships, tenant data |
| Keyword Search | BM25 (rank-bm25) | Exact term matching |
| Dense Embeddings | sentence-transformers (BGE) | Semantic similarity |
| Visual Embeddings | ColPali | Layout-aware retrieval for scanned docs |
| API | FastAPI | REST endpoints with tenant isolation |

## Document Types Supported

| Type | Metadata Extracted | Special Handling |
|------|-------------------|------------------|
| **Contracts** | Parties, dates, governing law, clauses | Legal structure detection, amendment tracking |
| **IRC/Tax Code** | Sections, forms, tax years, categories | Section references, regulation links |
| **Building Codes** | Code type (IBC/IRC/IFC), sections, categories | Code edition, jurisdiction |
| **Financial Reports** | Report type, fiscal periods, metrics | Table extraction, GAAP terms |
| **Letters** | Recipients, dates, subject | Correspondence chains |
| **Scanned PDFs** | OCR quality score | Quality-based routing, visual embeddings |

## Quick Start

### 1. Prerequisites

**Required:**
- Python 3.10+
- PostgreSQL 14+
- Docker & Docker Compose (recommended)

**Optional:**
- NVIDIA GPU for ColPali (8GB+ VRAM recommended)
- Tesseract OCR for scanned documents

### 2. Start Infrastructure

```bash
# Copy environment file
cp .env.example .env

# Start services (PostgreSQL, Qdrant)
docker-compose up -d postgres qdrant
```

### 3. Initialize Database

```bash
# The init.sql file creates all tables, indexes, and seed data
# It runs automatically when PostgreSQL container starts

# To manually run:
psql -h localhost -U raguser -d rag_db -f init.sql
```

### 4. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for sentence tokenization)
python -c "import nltk; nltk.download('punkt')"
```

### 5. Run the API

```bash
# With Docker
docker-compose up api

# Or directly
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Ingest Documents

```bash
# Upload via API (REQUIRED: X-Organization-ID header)
curl -X POST "http://localhost:8000/documents/upload" \
  -H "X-Organization-ID: org_acme" \
  -H "X-Workspace-ID: ws_legal" \
  -F "file=@/path/to/document.pdf"

# From server path
curl -X POST "http://localhost:8000/documents/ingest-path?file_path=/path/to/doc.pdf" \
  -H "X-Organization-ID: org_acme"
```

### 7. Search

```bash
# Hybrid search (REQUIRED: X-Organization-ID header)
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "X-Organization-ID: org_acme" \
  -d '{"query": "payment terms", "limit": 10}'

# With filters
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "X-Organization-ID: org_acme" \
  -H "X-Workspace-ID: ws_legal" \
  -d '{
    "query": "contract termination",
    "limit": 10,
    "filters": {"document_type": "contract"}
  }'
```

## API Endpoints

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/health` | GET | No | Health check |
| `/stats` | GET | No | System statistics |
| `/search` | POST | Yes | Hybrid search |
| `/search?q=...` | GET | Yes | Simple search |
| `/documents/upload` | POST | Yes | Upload document |
| `/documents/ingest-path` | POST | Yes | Ingest from path |
| `/documents` | GET | Yes | List documents |
| `/documents/{id}` | GET | Yes | Get document |
| `/documents/{id}` | DELETE | Yes | Delete document |

**Authentication Headers:**
- `X-Organization-ID` (required): Organization identifier
- `X-Workspace-ID` (optional): Workspace within organization
- `X-User-ID` (optional): User ID for audit logging

## Project Structure

```
rag-database/
├── src/
│   ├── api/                        # FastAPI application
│   │   └── main.py                 # REST endpoints with tenant isolation
│   ├── quality/                    # Document quality detection
│   │   └── document_quality.py     # Quality scoring & routing
│   ├── chunking/                   # Text chunking
│   │   ├── chonkie_chunker.py      # Chonkie integration (NEW)
│   │   └── hierarchical_chunker.py # Structure-preserving chunker
│   ├── metadata/                   # Metadata handling
│   │   ├── domain_schemas.py       # Contract, IRC, Building, Financial
│   │   └── tenant_schema.py        # Multi-tenant isolation (NEW)
│   ├── tables/                     # Table extraction
│   │   └── table_extractor.py
│   ├── terminology/                # Acronym/term handling
│   │   └── acronym_database.py
│   ├── graph/                      # Document relationships
│   │   └── document_graph.py
│   ├── embeddings/                 # Embedding models
│   │   ├── dense_embedder.py       # BGE embeddings
│   │   └── colpali_embedder.py     # Visual embeddings
│   ├── ingestion/                  # Document processing
│   │   └── document_processor.py   # Unstructured.io wrapper
│   ├── pipeline/                   # Ingestion orchestration
│   │   ├── ingestion.py            # Basic pipeline
│   │   └── enhanced_ingestion.py   # Enterprise pipeline
│   ├── retrieval/                  # Search logic
│   │   ├── hybrid_search.py        # Basic hybrid
│   │   └── enhanced_hybrid_search.py # With precision fallbacks
│   └── storage/                    # Data stores
│       ├── vector_store.py         # Qdrant (with tenant indexes)
│       ├── metadata_store.py       # PostgreSQL (with tenant columns)
│       └── bm25_store.py           # Keyword search
├── init.sql                        # Complete database schema (NEW)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Configuration

### Environment Variables

```bash
# Database
POSTGRES_URL=postgresql://user:pass@localhost:5432/rag_db

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embeddings
DENSE_MODEL=BAAI/bge-base-en-v1.5

# ColPali (requires GPU)
ENABLE_COLPALI=false
```

### Search Weights

Default weights for RRF fusion:
```python
{
    "bm25": 0.3,    # Keyword matching
    "dense": 0.5,   # Semantic similarity
    "colpali": 0.2  # Visual understanding
}
```

## Why This Approach?

Based on real-world enterprise RAG challenges (10+ deployments in pharma, banks, law firms):

### Problem: Multi-Tenant Data Must Never Mix
- **Reality**: Legal firm can't risk Client A seeing Client B's contracts
- **Solution**: Organization ID required on every request, indexed in all stores, filtered in all queries

### Problem: Document Quality Varies Wildly
- **Reality**: Enterprise docs range from perfect PDFs to garbage scanned copies from 1995
- **Solution**: Quality detection BEFORE processing → route to appropriate pipeline

### Problem: Fixed-Size Chunking Destroys Structure
- **Reality**: Research papers, contracts, codes all have different structures
- **Solution**: Chonkie + hierarchical chunking preserving document/section/paragraph/sentence levels

### Problem: Metadata is an Afterthought
- **Reality**: "Show me pediatric studies" needs different docs than "adult populations"
- **Solution**: Domain-specific schemas with keyword-based extraction (LLMs are too inconsistent)

### Problem: Semantic Search Fails 15-20% in Specialized Domains
- **Reality**: "IRC" means different things in tax vs. building code contexts
- **Solution**: Context-aware acronym expansion + hybrid search with precision fallbacks

### Problem: Tables Contain Critical Data
- **Reality**: Financial analysts need exact numbers from specific quarters
- **Solution**: Dedicated table extraction with dual embedding (structured + semantic)

### Problem: Documents Reference Each Other
- **Reality**: Amendment to MSA references original agreement
- **Solution**: Document relationship graph for cross-reference discovery

## License

MIT License
