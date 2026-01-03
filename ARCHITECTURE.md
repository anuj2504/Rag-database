# Enterprise RAG System - Architecture Guide

## System Overview

This is an **enterprise-grade RAG (Retrieval-Augmented Generation) system** designed for multi-tenant organizations handling complex documents like contracts, legal codes, financial reports, and technical manuals.

---

## THE COMPLETE FLOW

```
                              DOCUMENT INGESTION FLOW
===================================================================================

    [User/Agent]                    [API Layer]                    [Pipeline]
         |                              |                              |
         |   POST /ingest               |                              |
         |   + X-Organization-ID        |                              |
         |   + file                     |                              |
         |----------------------------->|                              |
         |                              |                              |
         |                              |   validate_tenant()          |
         |                              |----------------------------->|
         |                              |                              |
         |                              |                              |
===================================================================================

                              INSIDE THE PIPELINE
===================================================================================

  STAGE 1: DOCUMENT PARSING (Unstructured.io)
  ============================================

    [Raw Document]                                    [Parsed Output]
    (PDF/DOCX/Image)                                       |
         |                                                 |
         v                                                 v
    +------------------+     +------------------+    +------------------+
    | Unstructured.io  |---->| OCR (if scanned) |--->| Structured       |
    | Parser           |     | (Tesseract/etc)  |    | Elements         |
    +------------------+     +------------------+    +------------------+
                                                           |
    Outputs:                                               |
    - Raw text                                             |
    - Element types (Title, NarrativeText, Table, etc.)    |
    - Page images (for ColPali)                            |
    - Bounding boxes                                       v


  STAGE 2: QUALITY ASSESSMENT
  ============================================

    [Parsed Document]
         |
         v
    +------------------+
    | Quality Analyzer |
    +------------------+
         |
         +---> text_density_score (0-1)
         +---> ocr_confidence (0-1)
         +---> structure_score (0-1)
         +---> language_quality (0-1)
         |
         v
    +------------------+
    | Quality Router   |
    +------------------+
         |
         +---> HIGH    --> Use SDPM chunking (semantic double-pass merge)
         +---> MEDIUM  --> Use Semantic chunking
         +---> LOW     --> Use Sentence chunking
         +---> GARBAGE --> Use Token chunking OR flag for manual review


  STAGE 3: INTELLIGENT CHUNKING (Chonkie + Hierarchical)
  ============================================

    [Parsed Text]
         |
         v
    +-----------------------------------------------------------+
    |                    CHUNKING SERVICE                        |
    +-----------------------------------------------------------+
    |                                                           |
    |   Option A: CHONKIE (Modern, Token-Aware)                 |
    |   =========================================               |
    |   - TokenChunker:    Fast, fixed token count              |
    |   - SentenceChunker: Respects sentence boundaries         |
    |   - SemanticChunker: Groups by semantic similarity        |
    |   - SDPMChunker:     Best quality, double-pass merge      |
    |                                                           |
    |   Option B: HIERARCHICAL (Structure-Aware)                |
    |   =========================================               |
    |   - Document level:  Full document summary                |
    |   - Section level:   Major sections (Article I, etc.)     |
    |   - Paragraph level: Individual paragraphs (PRIMARY)      |
    |   - Sentence level:  For precision queries                |
    |                                                           |
    |   HYBRID APPROACH (Default):                              |
    |   ==========================                              |
    |   1. Detect document structure (sections, articles)       |
    |   2. Apply Chonkie WITHIN each section                    |
    |   3. Maintain parent-child relationships                  |
    |                                                           |
    +-----------------------------------------------------------+
         |
         v
    [EnterpriseChunk]
    - id: "doc_123_sec_1_chunk_0"
    - text: "..."
    - organization_id: "org_acme"      <-- CRITICAL: Tenant isolation
    - workspace_id: "ws_legal"
    - access_level: "confidential"
    - section_title: "Article I: Definitions"
    - parent_id: "doc_123_sec_1"
    - token_count: 256


  STAGE 4: METADATA EXTRACTION
  ============================================

    [Chunks + Full Text]
         |
         v
    +---------------------------+
    | Domain-Specific Extractor |
    +---------------------------+
         |
         +---> Legal:     parties, effective_date, governing_law
         +---> Financial: fiscal_period, amounts, metrics
         +---> Technical: version, specifications, standards
         |
         v
    [ExtractedMetadata]
    - document_type: "contract"
    - parties: ["ABC Corp", "XYZ Inc"]
    - effective_date: "2024-01-01"
    - key_terms: ["indemnification", "limitation of liability"]


  STAGE 5: EMBEDDING GENERATION
  ============================================

    [Chunks]
         |
         +------------------+------------------+
         |                  |                  |
         v                  v                  v
    +-----------+    +-----------+    +-----------+
    | Dense     |    | BM25      |    | ColPali   |
    | Embedder  |    | Indexer   |    | Embedder  |
    +-----------+    +-----------+    +-----------+
         |                  |                  |
         v                  v                  v
    768-dim vector    Token freq       128-dim multi-vector
    (BGE-base)        inverted index   (per page image)


  STAGE 6: STORAGE (With Tenant Isolation)
  ============================================

    +-----------------+     +-----------------+     +-----------------+
    | PostgreSQL      |     | Qdrant          |     | BM25 Index      |
    | (Metadata)      |     | (Vectors)       |     | (Keywords)      |
    +-----------------+     +-----------------+     +-----------------+
          |                       |                       |
          v                       v                       v
    +-------------+         +-------------+         +-------------+
    | documents   |         | Collection: |         | Per-tenant  |
    | chunks      |         | dense_vecs  |         | inverted    |
    | pages       |         |             |         | index       |
    | tables      |         | Payload:    |         |             |
    | ...         |         | org_id -----+---------+-- FILTER    |
    +-------------+         | workspace_id|         |             |
          |                 | access_level|         |             |
          |                 +-------------+         +-------------+
          |                       |
          +--- EVERY QUERY MUST FILTER BY organization_id ---+


===================================================================================
                              RETRIEVAL FLOW
===================================================================================

    [User Query]
    "What is the indemnification limit in the ABC contract?"
         |
         v
    +-----------------------------------------------------------+
    |                    HYBRID SEARCH                           |
    +-----------------------------------------------------------+
    |                                                           |
    |   1. QUERY ANALYSIS                                       |
    |      - Complexity: PRECISE (asks for specific value)      |
    |      - Recommended level: SENTENCE                        |
    |      - Keywords: "indemnification", "limit", "ABC"        |
    |                                                           |
    |   2. PARALLEL RETRIEVAL                                   |
    |      +---> Dense search (semantic similarity)             |
    |      +---> BM25 search (keyword matching)                 |
    |      +---> ColPali search (visual, if applicable)         |
    |                                                           |
    |   3. FUSION (Reciprocal Rank Fusion)                      |
    |      RRF_score = sum(1 / (k + rank_i)) for each method    |
    |                                                           |
    |   4. CONTEXT EXPANSION                                    |
    |      - Get parent chunks for more context                 |
    |      - Get sibling chunks if needed                       |
    |                                                           |
    +-----------------------------------------------------------+
         |
         v
    [Ranked Results with Context]


===================================================================================
                         MULTI-TENANT ISOLATION
===================================================================================

    CRITICAL: Data from different organizations must NEVER mix!

    +------------------+          +------------------+
    | Organization A   |          | Organization B   |
    | (org_acme)       |          | (org_beta)       |
    +------------------+          +------------------+
           |                             |
           v                             v
    +--------------+              +--------------+
    | Workspace 1  |              | Workspace X  |
    | (ws_legal)   |              | (ws_finance) |
    +--------------+              +--------------+
           |                             |
           v                             v
    +--------------+              +--------------+
    | Collection A |              | Collection X |
    | (contracts)  |              | (reports)    |
    +--------------+              +--------------+


    Every API call requires:
    - X-Organization-ID (required)
    - X-Workspace-ID (optional)
    - X-Collection-ID (optional)
    - X-Access-Level (optional, default: internal)

    Every database query includes:
    WHERE organization_id = :org_id
    [AND workspace_id = :ws_id]
    [AND access_level IN (:allowed_levels)]


===================================================================================
                         FILE STRUCTURE
===================================================================================

    src/
    ├── api/
    │   └── main.py                 # FastAPI endpoints
    │
    ├── ingestion/
    │   └── document_processor.py   # Unstructured.io wrapper
    │
    ├── chunking/
    │   ├── chonkie_chunker.py     # Chonkie integration (semantic)
    │   ├── hierarchical_chunker.py # Structure-aware chunking
    │   └── chunking_service.py     # UNIFIED interface (NEW)
    │
    ├── quality/
    │   └── document_quality.py     # Quality assessment & routing
    │
    ├── embeddings/
    │   ├── dense_embedder.py       # Sentence-transformers
    │   └── colpali_embedder.py     # ColPali multi-vector
    │
    ├── storage/
    │   ├── vector_store.py         # Qdrant wrapper
    │   ├── metadata_store.py       # PostgreSQL wrapper
    │   └── bm25_store.py           # BM25 keyword index
    │
    ├── retrieval/
    │   ├── hybrid_search.py        # Basic hybrid search
    │   └── enhanced_hybrid_search.py # Advanced with context
    │
    ├── metadata/
    │   ├── domain_schemas.py       # Document type extractors
    │   └── tenant_schema.py        # Multi-tenant definitions
    │
    ├── pipeline/
    │   ├── ingestion.py            # Basic pipeline
    │   ├── enhanced_ingestion.py   # Full-featured pipeline
    │   └── master_pipeline.py      # UNIFIED entry point (NEW)
    │
    ├── tables/
    │   └── table_extractor.py      # Table detection & embedding
    │
    ├── graph/
    │   └── document_graph.py       # Cross-document relationships
    │
    └── terminology/
        └── acronym_database.py     # Domain acronym expansion


===================================================================================
                         WHICH PIPELINE TO USE?
===================================================================================

    master_pipeline.py (RECOMMENDED)
    ================================
    - Use this for all new integrations
    - Automatically selects best chunking strategy
    - Includes all enterprise features
    - Full multi-tenant support

    Usage:
    ```python
    from src.pipeline.master_pipeline import create_master_pipeline

    pipeline = create_master_pipeline(
        postgres_url="postgresql://...",
        qdrant_host="localhost",
        enable_colpali=True,
    )

    result = pipeline.ingest(
        file_path="contract.pdf",
        tenant_context=TenantContext(
            organization_id="org_acme",
            workspace_id="ws_legal",
            access_level=AccessLevel.CONFIDENTIAL,
        )
    )
    ```


===================================================================================
                         AGENT INTEGRATION
===================================================================================

    For AI agents querying this knowledge base:

    ```python
    from src.retrieval.enhanced_hybrid_search import EnhancedHybridSearch

    search = EnhancedHybridSearch(...)

    # Agent query with tenant context
    results = search.search(
        query="What is the termination clause?",
        tenant_filter={
            "organization_id": "org_acme",
            "workspace_id": "ws_legal",      # optional
            "access_level": ["internal", "public"],  # user's allowed levels
        },
        top_k=10,
        include_context=True,  # Get parent/sibling chunks
    )

    for result in results:
        print(f"Score: {result.score}")
        print(f"Text: {result.text}")
        print(f"Section: {result.section_title}")
        print(f"Document: {result.document_id}")
    ```


===================================================================================
```

## Quick Reference

### Starting the System

```bash
# 1. Start infrastructure
docker-compose up -d

# 2. Initialize database
psql -h localhost -U rag_user -d rag_db -f init.sql

# 3. Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Upload and process document |
| `/search` | POST | Hybrid search query |
| `/documents` | GET | List documents (filtered by tenant) |
| `/documents/{id}` | DELETE | Remove document |
| `/health` | GET | System health check |

### Required Headers

| Header | Required | Description |
|--------|----------|-------------|
| `X-Organization-ID` | Yes | Tenant organization |
| `X-Workspace-ID` | No | Workspace within org |
| `X-Collection-ID` | No | Document collection |
| `X-Access-Level` | No | User's access level |
