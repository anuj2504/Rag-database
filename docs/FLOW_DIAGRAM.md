# NHAI/L&T Enterprise RAG System - Flow Diagrams

## System Overview

This document provides visual flow diagrams for understanding how the RAG system works.

---

## 1. COMPLETE INGESTION FLOW

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DOCUMENT INGESTION FLOW                                │
│                                                                                  │
│  User/Agent uploads document with TenantContext                                  │
│  ↓                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ API LAYER (src/api/main.py)                                                 │ │
│  │                                                                             │ │
│  │  POST /ingest                                                               │ │
│  │  Headers:                                                                   │ │
│  │    - X-Organization-ID: "nhai" (REQUIRED)                                   │ │
│  │    - X-Workspace-ID: "contracts" (optional)                                 │ │
│  │    - X-Access-Level: "confidential" (optional)                              │ │
│  │  Body: multipart/form-data with file                                        │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                        ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ MASTER PIPELINE (src/pipeline/master_pipeline.py)                           │ │
│  │                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ STAGE 1: PARSE (Unstructured.io)                                     │   │ │
│  │  │                                                                       │   │ │
│  │  │  PDF/DOCX/Image → partition_pdf() → Elements                         │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              OCR if scanned (Tesseract)                               │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Extract page images (for ColPali)                        │   │ │
│  │  │                                                                       │   │ │
│  │  │  Output: ProcessedDocument                                            │   │ │
│  │  │    - document_id (MD5 hash)                                           │   │ │
│  │  │    - chunks (raw text elements)                                       │   │ │
│  │  │    - page_images (PIL images)                                         │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                        ↓                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ STAGE 2: QUALITY ASSESSMENT                                          │   │ │
│  │  │                                                                       │   │ │
│  │  │  Full Text → DocumentQualityAnalyzer                                  │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Calculate metrics:                                       │   │ │
│  │  │                - text_density_score (0-1)                             │   │ │
│  │  │                - ocr_confidence (0-1)                                 │   │ │
│  │  │                - structure_score (0-1)                                │   │ │
│  │  │                - language_quality (0-1)                               │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Determine Quality Tier:                                  │   │ │
│  │  │                HIGH   (>0.8) → SDPM chunking                          │   │ │
│  │  │                MEDIUM (>0.5) → Semantic chunking                      │   │ │
│  │  │                LOW    (>0.3) → Sentence chunking                      │   │ │
│  │  │                GARBAGE(<0.3) → Token chunking / Manual review         │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                        ↓                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ STAGE 3: METADATA EXTRACTION                                         │   │ │
│  │  │                                                                       │   │ │
│  │  │  Full Text → UnifiedMetadataExtractor                                 │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Detect document_type:                                    │   │ │
│  │  │                - contract, tender, dpr, specification                 │   │ │
│  │  │                - irc_code, building_code, standard                    │   │ │
│  │  │                - financial_report, boq, estimate                      │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Extract domain-specific fields:                          │   │ │
│  │  │                Legal:     parties, effective_date, governing_law      │   │ │
│  │  │                Financial: fiscal_period, amounts, metrics             │   │ │
│  │  │                Technical: version, standards, specifications          │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                        ↓                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ STAGE 4: INTELLIGENT CHUNKING (ChunkingService)                      │   │ │
│  │  │                                                                       │   │ │
│  │  │  Full Text → StructureDetector                                        │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Detect sections:                                         │   │ │
│  │  │                - Legal: Article I, Section 1.1, Clause (a)            │   │ │
│  │  │                - Technical: Chapter 1, Part A, Drawing No.            │   │ │
│  │  │                - Code: § 123.45, IRC 101, IBC 202                     │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Select Chonkie strategy (based on quality):              │   │ │
│  │  │                - SDPMChunker (semantic double-pass merge)             │   │ │
│  │  │                - SemanticChunker (embedding-based)                    │   │ │
│  │  │                - SentenceChunker (boundary-aware)                     │   │ │
│  │  │                - TokenChunker (fixed size)                            │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Apply chunking within each section                       │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Output: List[UnifiedChunk]                               │   │ │
│  │  │                - id, document_id, text                                │   │ │
│  │  │                - organization_id, workspace_id (TENANT)               │   │ │
│  │  │                - level (DOCUMENT/SECTION/PARAGRAPH)                   │   │ │
│  │  │                - parent_id, section_title                             │   │ │
│  │  │                - token_count                                          │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                        ↓                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ STAGE 5: TABLE EXTRACTION (Optional)                                 │   │ │
│  │  │                                                                       │   │ │
│  │  │  Full Text → TableExtractor                                           │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Detect tables (BOQ, schedules, rate tables)              │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Generate dual representations:                           │   │ │
│  │  │                - Structured: row/column format                        │   │ │
│  │  │                - Semantic: natural language description               │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                        ↓                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ STAGE 6: STORE METADATA (PostgreSQL)                                 │   │ │
│  │  │                                                                       │   │ │
│  │  │  UnifiedChunks → MetadataStore                                        │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Insert into tables:                                      │   │ │
│  │  │                - documents (with organization_id)                     │   │ │
│  │  │                - chunks (with organization_id)                        │   │ │
│  │  │                - pages (with organization_id)                         │   │ │
│  │  │                                                                       │   │ │
│  │  │              ALL records tagged with tenant fields!                   │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                        ↓                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ STAGE 7: EMBEDDING & INDEXING                                        │   │ │
│  │  │                                                                       │   │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │ │
│  │  │  │ DENSE       │  │ BM25        │  │ COLPALI     │                   │   │ │
│  │  │  │ EMBEDDINGS  │  │ INDEX       │  │ EMBEDDINGS  │                   │   │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │ │
│  │  │        ↓                ↓                ↓                            │   │ │
│  │  │  BGE-base-en-v1.5  rank-bm25     ColPali model                        │   │ │
│  │  │  768 dimensions    inverted idx  128-dim multi-vec                    │   │ │
│  │  │        ↓                ↓                ↓                            │   │ │
│  │  │  Qdrant Collection  In-memory    Qdrant Collection                    │   │ │
│  │  │  "nhai_lt_documents" or file    "nhai_lt_colpali"                     │   │ │
│  │  │                                                                       │   │ │
│  │  │  ALL vectors have payload with organization_id!                       │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                        ↓                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ STAGE 8: DOCUMENT GRAPH (Optional)                                   │   │ │
│  │  │                                                                       │   │ │
│  │  │  Document → DocumentGraph                                             │   │ │
│  │  │                   ↓                                                   │   │ │
│  │  │              Find relationships:                                      │   │ │
│  │  │                - References to other documents                        │   │ │
│  │  │                - Amendments to contracts                              │   │ │
│  │  │                - Code sections referencing each other                 │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                        ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ RESULT: IngestionResult                                                     │ │
│  │                                                                             │ │
│  │  {                                                                          │ │
│  │    "document_id": "abc123...",                                              │ │
│  │    "status": "success",                                                     │ │
│  │    "organization_id": "nhai",                                               │ │
│  │    "quality_tier": "high",                                                  │ │
│  │    "chunks_created": 45,                                                    │ │
│  │    "chunks_indexed": 38,                                                    │ │
│  │    "pages_indexed": 12,                                                     │ │
│  │    "document_type": "contract",                                             │ │
│  │    "processing_time_seconds": 4.2                                           │ │
│  │  }                                                                          │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. RETRIEVAL FLOW

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RETRIEVAL FLOW                                      │
│                                                                                  │
│  User/Agent sends search query with TenantContext                                │
│  ↓                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ QUERY ANALYSIS                                                              │ │
│  │                                                                             │ │
│  │  "What is the indemnification limit in the ABC contract?"                   │ │
│  │                   ↓                                                         │ │
│  │  QueryComplexityAnalyzer determines:                                        │ │
│  │    - Query type: PRECISE (asks for specific value)                          │ │
│  │    - Retrieval level: SENTENCE (for precision)                              │ │
│  │    - Keywords: "indemnification", "limit", "ABC"                            │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                        ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ PARALLEL RETRIEVAL (EnhancedHybridSearch)                                   │ │
│  │                                                                             │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │ │
│  │  │ DENSE SEARCH     │  │ BM25 SEARCH      │  │ COLPALI SEARCH   │          │ │
│  │  │                  │  │                  │  │                  │          │ │
│  │  │ embed(query)     │  │ tokenize(query)  │  │ embed(query)     │          │ │
│  │  │      ↓           │  │      ↓           │  │      ↓           │          │ │
│  │  │ Qdrant search    │  │ BM25 ranking     │  │ MaxSim search    │          │ │
│  │  │ with filter:     │  │ with filter:     │  │ with filter:     │          │ │
│  │  │ org_id="nhai"    │  │ org_id="nhai"    │  │ org_id="nhai"    │          │ │
│  │  │      ↓           │  │      ↓           │  │      ↓           │          │ │
│  │  │ Top 20 results   │  │ Top 20 results   │  │ Top 5 pages      │          │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘          │ │
│  │           ↓                    ↓                    ↓                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │ RECIPROCAL RANK FUSION (RRF)                                         │   │ │
│  │  │                                                                       │   │ │
│  │  │  For each document d:                                                 │   │ │
│  │  │    RRF(d) = Σ 1/(k + rank_i(d))                                       │   │ │
│  │  │                                                                       │   │ │
│  │  │  where k=60, rank_i is rank in retrieval method i                     │   │ │
│  │  │                                                                       │   │ │
│  │  │  Merge and re-rank all results                                        │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                        ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ CONTEXT EXPANSION                                                           │ │
│  │                                                                             │ │
│  │  For each top result:                                                       │ │
│  │    1. Get parent chunk (section context)                                    │ │
│  │    2. Get sibling chunks (surrounding context)                              │ │
│  │    3. Get document summary                                                  │ │
│  │                                                                             │ │
│  │  Output: Enriched results with hierarchy                                    │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                        ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ RESULT: SearchResults                                                       │ │
│  │                                                                             │ │
│  │  [                                                                          │ │
│  │    {                                                                        │ │
│  │      "chunk_id": "contract_001_chunk_15",                                   │ │
│  │      "score": 0.92,                                                         │ │
│  │      "text": "The indemnification limit shall not exceed $5,000,000...",    │ │
│  │      "section_title": "Article 8: Indemnification",                         │ │
│  │      "document_id": "contract_001",                                         │ │
│  │      "organization_id": "nhai",                                             │ │
│  │      "parent_context": "...",                                               │ │
│  │    },                                                                       │ │
│  │    ...                                                                      │ │
│  │  ]                                                                          │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. MULTI-TENANT DATA ISOLATION

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MULTI-TENANT ISOLATION                                  │
│                                                                                  │
│  CRITICAL: Data from different organizations NEVER mixes!                        │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ ORGANIZATION HIERARCHY                                                      │ │
│  │                                                                             │ │
│  │  Organization (org_id)                                                      │ │
│  │       │                                                                     │ │
│  │       ├── Workspace (ws_id)                                                 │ │
│  │       │       │                                                             │ │
│  │       │       ├── Collection (collection_id)                                │ │
│  │       │       │       │                                                     │ │
│  │       │       │       └── Documents                                         │ │
│  │       │       │               │                                             │ │
│  │       │       │               └── Chunks                                    │ │
│  │       │       │                                                             │ │
│  │       │       └── Collection                                                │ │
│  │       │               └── Documents                                         │ │
│  │       │                                                                     │ │
│  │       └── Workspace                                                         │ │
│  │               └── Collection                                                │ │
│  │                       └── Documents                                         │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ EXAMPLE: NHAI vs L&T                                                        │ │
│  │                                                                             │ │
│  │  ┌─────────────────────┐         ┌─────────────────────┐                   │ │
│  │  │ NHAI                │         │ L&T                 │                   │ │
│  │  │ org_id: "nhai"      │         │ org_id: "lnt"       │                   │ │
│  │  ├─────────────────────┤         ├─────────────────────┤                   │ │
│  │  │ Workspaces:         │         │ Workspaces:         │                   │ │
│  │  │  - contracts        │         │  - projects         │                   │ │
│  │  │  - specifications   │         │  - tenders          │                   │ │
│  │  │  - dprs             │         │  - finance          │                   │ │
│  │  └─────────────────────┘         └─────────────────────┘                   │ │
│  │           │                               │                                 │ │
│  │           │ NEVER SEES                    │ NEVER SEES                      │ │
│  │           │ L&T DATA                      │ NHAI DATA                       │ │
│  │           ↓                               ↓                                 │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │                        STORAGE LAYER                                 │   │ │
│  │  │                                                                       │   │ │
│  │  │  PostgreSQL                         Qdrant                            │   │ │
│  │  │  ┌─────────────────────┐           ┌─────────────────────┐           │   │ │
│  │  │  │ SELECT * FROM docs  │           │ search(             │           │   │ │
│  │  │  │ WHERE org_id='nhai' │           │   vector,           │           │   │ │
│  │  │  └─────────────────────┘           │   filter: {         │           │   │ │
│  │  │                                    │     org_id: 'nhai'  │           │   │ │
│  │  │                                    │   }                 │           │   │ │
│  │  │                                    │ )                   │           │   │ │
│  │  │                                    └─────────────────────┘           │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ ACCESS LEVELS                                                               │ │
│  │                                                                             │ │
│  │  public       → Anyone in organization can see                              │ │
│  │  internal     → Standard internal access (default)                          │ │
│  │  restricted   → Limited to specific workspaces                              │ │
│  │  confidential → Highest security, audit logged                              │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. CHUNKING STRATEGY SELECTION

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       CHUNKING STRATEGY SELECTION                                │
│                                                                                  │
│  Document Quality → Chunking Strategy → Chunk Quality                            │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │   Quality    Score      Strategy           Description                      │ │
│  │   ───────    ─────      ────────           ───────────                      │ │
│  │                                                                             │ │
│  │   HIGH       >0.8       SDPM               Semantic Double-Pass Merge       │ │
│  │                         (SDPMChunker)      Best semantic coherence          │ │
│  │                                            Two-pass algorithm               │ │
│  │                                            Looks ahead for better splits    │ │
│  │                                                                             │ │
│  │   MEDIUM     >0.5       SEMANTIC           Embedding-based chunking         │ │
│  │                         (SemanticChunker)  Groups by semantic similarity    │ │
│  │                                            Good balance of speed/quality    │ │
│  │                                                                             │ │
│  │   LOW        >0.3       SENTENCE           Sentence boundary aware          │ │
│  │                         (SentenceChunker)  Respects sentence endings        │ │
│  │                                            Robust to OCR noise              │ │
│  │                                                                             │ │
│  │   GARBAGE    <0.3       TOKEN              Fixed token count                │ │
│  │                         (TokenChunker)     Fast, simple splitting           │ │
│  │                                            Last resort for bad OCR          │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ STRUCTURE-AWARE CHUNKING                                                    │ │
│  │                                                                             │ │
│  │  For structured documents (contracts, codes, specs):                        │ │
│  │                                                                             │ │
│  │    1. Detect sections first                                                 │ │
│  │       ├── Article I: Definitions                                            │ │
│  │       ├── Article II: Scope of Work                                         │ │
│  │       └── Article III: Payment Terms                                        │ │
│  │                                                                             │ │
│  │    2. Apply Chonkie WITHIN each section                                     │ │
│  │       Article II: Scope of Work                                             │ │
│  │       ├── chunk_1 (paragraph)                                               │ │
│  │       ├── chunk_2 (paragraph)                                               │ │
│  │       └── chunk_3 (paragraph)                                               │ │
│  │                                                                             │ │
│  │    3. Maintain parent-child relationships                                   │ │
│  │       chunk_1.parent_id = "doc_sec_2"                                       │ │
│  │       chunk_1.section_title = "Article II: Scope of Work"                   │ │
│  │                                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. FILE STRUCTURE

```
src/
├── api/
│   └── main.py                 # FastAPI endpoints
│
├── ingestion/
│   └── document_processor.py   # Unstructured.io wrapper
│
├── chunking/
│   ├── unified_chunk.py        # UnifiedChunk dataclass (USE THIS)
│   ├── chunking_service.py     # ChunkingService (USE THIS)
│   ├── hierarchical_chunker.py # Legacy - structure detection
│   └── chonkie_chunker.py      # Legacy - Chonkie wrapper
│
├── quality/
│   └── document_quality.py     # Quality assessment & routing
│
├── embeddings/
│   ├── dense_embedder.py       # Sentence-transformers (BGE)
│   └── colpali_embedder.py     # ColPali multi-vector
│
├── storage/
│   ├── vector_store.py         # Qdrant wrapper
│   ├── metadata_store.py       # PostgreSQL wrapper
│   └── bm25_store.py           # BM25 keyword index
│
├── retrieval/
│   ├── hybrid_search.py        # Basic hybrid search
│   └── enhanced_hybrid_search.py # Advanced with context expansion
│
├── metadata/
│   ├── domain_schemas.py       # Document type extractors
│   └── tenant_schema.py        # TenantContext, AccessLevel
│
├── pipeline/
│   ├── master_pipeline.py      # MasterPipeline (USE THIS)
│   ├── ingestion.py            # Legacy - basic pipeline
│   └── enhanced_ingestion.py   # Legacy - enterprise pipeline
│
├── tables/
│   └── table_extractor.py      # Table detection & embedding
│
├── graph/
│   └── document_graph.py       # Cross-document relationships
│
└── terminology/
    └── acronym_database.py     # Domain acronym expansion
```

---

## Quick Start

```python
from src.pipeline import create_master_pipeline
from src.metadata.tenant_schema import TenantContext, AccessLevel

# Create pipeline
pipeline = create_master_pipeline(
    postgres_url="postgresql://user:pass@localhost:5432/rag_db",
    qdrant_host="localhost",
)

# Ingest document (NHAI example)
result = pipeline.ingest(
    file_path="contract.pdf",
    tenant_context=TenantContext(
        organization_id="nhai",
        workspace_id="contracts",
        access_level=AccessLevel.CONFIDENTIAL,
    )
)

print(f"Status: {result.status}")
print(f"Chunks indexed: {result.chunks_indexed}")
```
