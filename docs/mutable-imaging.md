# Vision-Aware Multimodal RAG Improvements

Based on the blog "ColPALI Meets DocLayNet: A Vision-Aware Multimodal RAG for Document-QA", here's an analysis of how to improve your RAG system.

## Current State Assessment

Your system **already has solid multimodal foundations**:
- ColPali multi-vector store (`QdrantMultiVectorStore`) with MaxSim retrieval
- Page image extraction during ingestion
- Hybrid search combining BM25 + Dense + ColPali (weights: 0.3/0.5/0.2)
- Quality-based routing for document processing

## Key Gaps Identified (What the Blog Does Better)

### 1. **No Layout Detection (DocLayNet/YOLO)**
**Gap**: Your system extracts pages as images but doesn't segment them into layout components (tables, figures, headers, paragraphs).

**Blog Approach**: Uses YOLO-DocLayNet to detect and extract:
- Tables → separate embedding + structured extraction
- Figures/charts → visual embedding only
- Section headers → structural metadata
- Paragraphs → text embedding

**Benefit**: Enables targeted retrieval of specific visual elements rather than entire pages.

### 2. **Dual Retrieval Paths Not Fully Separated**
**Gap**: ColPali and dense embeddings are fused but don't have separate retrieval paths for text vs. visual queries.

**Blog Approach**:
- Text queries → Text vector collection (mxbai-embed)
- Visual queries → Image vector collection (ColPali)
- Query routed to both, results merged

### 3. **No Visual Element Cropping**
**Gap**: ColPali embeds full pages (~1024 patches), but visually rich elements (tables, charts) are diluted.

**Blog Approach**: YOLO crops individual visual elements → ColPali embeds each separately → More precise retrieval.

### 4. **Missing Vision-Language Model (VLM) for Answer Generation**
**Gap**: Your system retrieves context but doesn't mention using a VLM that can process both retrieved text AND images.

**Blog Approach**: Uses LLaMA-4 (or similar VLM) that can see both text context and relevant figure/table images.

---

## Recommended Improvements

### Priority 1: Add Layout Detection Service (High Impact)

Create a new service that uses YOLO-DocLayNet or similar:

**Files to create:**
- `src/layout/layout_detector.py` - YOLO-DocLayNet integration
- `src/layout/element_types.py` - Enum for detected elements

**Integration points:**
- `src/pipeline/master_pipeline.py` - Add layout detection stage after PARSE
- `src/storage/vector_store.py` - Add separate collection for visual elements

**Models to consider:**
- `yolo-doclaynet` (fast, production-ready)
- `unstructured` already has some layout detection (leverage existing)
- `layoutlm` for structure understanding

### Priority 2: Separate Visual Element Embedding

Instead of embedding full pages:
1. Detect visual elements (tables, figures, charts)
2. Crop each element
3. Embed cropped images with ColPali
4. Store in dedicated `visual_elements` collection

**Files to modify:**
- `src/pipeline/master_pipeline.py` - Add visual element extraction
- `src/storage/vector_store.py` - Add `QdrantVisualElementStore`

### Priority 3: Enhanced Query Routing

**Files to modify:**
- `src/retrieval/enhanced_hybrid_search.py`

Add visual query detection:
```python
class QueryModalityDetector:
    """Detect if query targets visual content."""

    VISUAL_PATTERNS = [
        r'(?:in\s+)?(?:the\s+)?(?:table|figure|chart|graph|diagram)',
        r'(?:show|display|visualize)',
        r'(?:what\s+does\s+the\s+(?:table|figure)\s+show)',
    ]
```

Route visual queries → ColPali with higher weight
Route text queries → Dense with higher weight

### Priority 4: VLM Integration for Generation

When retrieved context includes visual elements:
1. Pass both text chunks AND relevant images to VLM
2. Use vision-capable model (GPT-4V, Claude, Gemini, LLaVA)

**Files to create:**
- `src/generation/vlm_generator.py` - VLM-based answer generation

---

## Implementation Plan

### Phase 1: Layout Detection Integration
1. Add YOLO-DocLayNet dependency
2. Create `LayoutDetector` class
3. Integrate into ingestion pipeline
4. Store detected elements with bounding boxes

### Phase 2: Visual Element Store
1. Create dedicated Qdrant collection for visual elements
2. Crop detected elements from page images
3. Embed with ColPali
4. Add element-level search capability

### Phase 3: Query Routing Enhancement
1. Add visual query detection to `EnhancedHybridSearcher`
2. Implement dynamic weight adjustment based on query type
3. Add element-type filtering (search only tables, only figures, etc.)

### Phase 4: VLM Answer Generation
1. Add VLM integration (OpenAI GPT-4V / Anthropic Claude / local LLaVA)
2. Pass relevant images alongside text context
3. Enable multimodal answer generation

---

## Files Summary

**New files to create:**
- `src/layout/layout_detector.py`
- `src/layout/element_types.py`
- `src/generation/vlm_generator.py`

**Files to modify:**
- `src/pipeline/master_pipeline.py` - Add layout detection stage
- `src/storage/vector_store.py` - Add visual element collection
- `src/retrieval/enhanced_hybrid_search.py` - Add visual query routing
- `src/retrieval/hybrid_search.py` - Support element-level retrieval

---

## Dependencies to Add

```
# Layout detection
ultralytics>=8.0.0  # YOLO
# or
doclaynet-base  # Hugging Face model

# VLM (choose one)
openai>=1.0.0  # GPT-4V
anthropic>=0.18.0  # Claude
transformers>=4.36.0  # Local LLaVA
```

---

## Quick Wins (Can Implement Now)

1. **Boost ColPali weight for visual queries**: Detect "table", "figure", "chart" in query → increase ColPali weight from 0.2 to 0.5
2. **Add element-type metadata**: During table extraction, tag with `element_type: table` for filtering
3. **Leverage existing Unstructured layout**: Unstructured.io already detects some layout elements - extract and use that metadata

---

## Questions for You

1. **Which layout detection model do you prefer?**
   - YOLO-DocLayNet (fast, 60 FPS)
   - LayoutLMv3 (more accurate, slower)
   - Leverage Unstructured.io's existing detection

2. **VLM for generation?**
   - Cloud API (GPT-4V, Claude)
   - Local model (LLaVA, Qwen-VL)
   - Both with fallback

3. **Priority focus?**
   - Tables (most common visual element)
   - Figures/charts (complex visual reasoning)
   - Both equally
