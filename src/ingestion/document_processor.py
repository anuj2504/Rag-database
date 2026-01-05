"""Document processing using Unstructured.io with full feature utilization.

This module leverages Unstructured.io's advanced features:
- YOLOX layout detection for better element classification
- Image/Table extraction with base64 encoding
- Bounding box coordinates for visual elements
- Multi-language support with per-element detection
"""
import os
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.xlsx import partition_xlsx
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from pdf2image import convert_from_path
from PIL import Image
import io

logger = logging.getLogger(__name__)


class VisualElementType(Enum):
    """Types of visual elements that can be extracted."""
    TABLE = "Table"
    IMAGE = "Image"
    FIGURE = "Figure"
    CHART = "Chart"
    FORMULA = "Formula"


@dataclass
class BoundingBox:
    """Bounding box coordinates for a visual element."""
    x1: float  # Left
    y1: float  # Top
    x2: float  # Right
    y2: float  # Bottom
    page_number: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "page_number": self.page_number
        }

    @classmethod
    def from_coordinates(cls, coords: Any, page_number: int) -> Optional["BoundingBox"]:
        """Create BoundingBox from Unstructured coordinates metadata."""
        if coords is None:
            return None
        try:
            # Unstructured returns points as list of (x, y) tuples
            if hasattr(coords, 'points'):
                points = coords.points
                if points and len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    return cls(
                        x1=min(xs), y1=min(ys),
                        x2=max(xs), y2=max(ys),
                        page_number=page_number
                    )
        except Exception as e:
            logger.warning(f"Failed to parse coordinates: {e}")
        return None


@dataclass
class VisualElement:
    """Represents an extracted visual element (table, figure, chart)."""
    element_id: str
    document_id: str
    element_type: VisualElementType
    page_number: int
    image_base64: Optional[str]  # Base64-encoded cropped image
    bbox: Optional[BoundingBox]
    text_content: Optional[str]  # For tables: HTML representation
    html_content: Optional[str]  # Full HTML if available
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decode_image(self) -> Optional[Image.Image]:
        """Decode base64 image to PIL Image."""
        if not self.image_base64:
            return None
        try:
            image_data = base64.b64decode(self.image_base64)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
            return None


@dataclass
class ProcessedChunk:
    """Represents a processed document chunk."""
    chunk_id: str
    document_id: str
    text: str
    page_number: Optional[int]
    chunk_index: int
    element_type: str  # 'Title', 'NarrativeText', 'Table', etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Represents a fully processed document with visual elements."""
    document_id: str
    filename: str
    file_path: str
    document_type: str  # 'contract', 'letter', 'code', etc.
    chunks: List[ProcessedChunk]
    page_images: List[Image.Image]  # Full page images for ColPali
    visual_elements: List[VisualElement]  # Extracted tables, figures, charts
    metadata: Dict[str, Any]
    processed_at: datetime


class DocumentProcessor:
    """
    Process documents using Unstructured.io with full feature utilization.

    Features enabled:
    - YOLOX layout detection model for accurate element classification
    - Visual element extraction (tables, images) as base64
    - Bounding box coordinates for all elements
    - Per-element language detection
    - Form field extraction (optional)
    """

    # Visual element types to extract
    VISUAL_ELEMENT_TYPES = ["Image", "Table"]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        extract_images: bool = True,
        extract_visual_elements: bool = True,
        ocr_languages: List[str] = ["eng"],
        hi_res_model: str = "yolox",  # Better layout detection than default
        detect_language_per_element: bool = False,
        extract_forms: bool = False,
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            extract_images: Whether to extract full page images for ColPali
            extract_visual_elements: Whether to extract tables/figures as separate elements
            ocr_languages: Languages for OCR
            hi_res_model: Layout detection model ('yolox', 'detectron2_onnx', 'chipper')
            detect_language_per_element: Enable per-element language detection
            extract_forms: Enable form field extraction
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images
        self.extract_visual_elements = extract_visual_elements
        self.ocr_languages = ocr_languages
        self.hi_res_model = hi_res_model
        self.detect_language_per_element = detect_language_per_element
        self.extract_forms = extract_forms

        logger.info(
            f"DocumentProcessor initialized with hi_res_model={hi_res_model}, "
            f"extract_visual_elements={extract_visual_elements}"
        )

    def _generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID from file path and content hash."""
        with open(file_path, 'rb') as f:
            content_hash = hashlib.md5(f.read()).hexdigest()[:12]
        filename = Path(file_path).stem
        return f"{filename}_{content_hash}"

    def _detect_document_type(self, filename: str, elements: List[Element]) -> str:
        """Detect document type based on content and filename."""
        filename_lower = filename.lower()

        # Filename-based detection
        if any(term in filename_lower for term in ['contract', 'agreement', 'terms']):
            return 'contract'
        if any(term in filename_lower for term in ['letter', 'correspondence']):
            return 'letter'
        if any(term in filename_lower for term in ['invoice', 'bill', 'receipt']):
            return 'invoice'
        if any(term in filename_lower for term in ['boq', 'bill of quantities', 'quantity']):
            return 'boq'
        if any(term in filename_lower for term in ['schedule', 'timeline', 'gantt']):
            return 'schedule'
        if filename_lower.endswith(('.xlsx', '.xls')):
            return 'spreadsheet'
        if filename_lower.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c')):
            return 'code'

        # Content-based detection
        text_content = ' '.join([str(el) for el in elements[:10]]).lower()
        if any(term in text_content for term in ['hereby agree', 'party', 'whereas', 'contract']):
            return 'contract'
        if any(term in text_content for term in ['dear', 'sincerely', 'regards']):
            return 'letter'

        return 'document'

    def _extract_page_images(self, file_path: str, dpi: int = 150) -> List[Image.Image]:
        """Extract page images from PDF for ColPali processing."""
        images = []
        try:
            if file_path.lower().endswith('.pdf'):
                # Convert PDF pages to images
                pages = convert_from_path(file_path, dpi=dpi)
                for page in pages:
                    # Resize if too large
                    max_size = 1024
                    if max(page.size) > max_size:
                        page.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    images.append(page)
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                img = Image.open(file_path)
                images.append(img)
        except Exception as e:
            print(f"Warning: Could not extract images from {file_path}: {e}")
        return images

    def _extract_visual_elements(
        self,
        elements: List[Element],
        document_id: str
    ) -> List[VisualElement]:
        """
        Extract visual elements (tables, images, figures) from parsed elements.

        Args:
            elements: Raw elements from Unstructured
            document_id: Parent document ID

        Returns:
            List of VisualElement objects
        """
        visual_elements = []
        visual_idx = 0

        for element in elements:
            category = getattr(element, 'category', None)
            if category not in self.VISUAL_ELEMENT_TYPES:
                continue

            # Get metadata
            elem_metadata = element.metadata.to_dict() if hasattr(element.metadata, 'to_dict') else {}
            page_number = elem_metadata.get('page_number', 1)

            # Get bounding box
            bbox = None
            coords = getattr(element.metadata, 'coordinates', None)
            if coords:
                bbox = BoundingBox.from_coordinates(coords, page_number)

            # Get base64 image if available
            image_base64 = elem_metadata.get('image_base64')

            # Get text/HTML content
            text_content = str(element) if element else None
            html_content = elem_metadata.get('text_as_html')

            # Determine element type
            if category == "Table":
                elem_type = VisualElementType.TABLE
            elif category == "Image":
                elem_type = VisualElementType.IMAGE
            else:
                elem_type = VisualElementType.FIGURE

            element_id = f"{document_id}_visual_{visual_idx}"
            visual_elements.append(VisualElement(
                element_id=element_id,
                document_id=document_id,
                element_type=elem_type,
                page_number=page_number,
                image_base64=image_base64,
                bbox=bbox,
                text_content=text_content,
                html_content=html_content,
                metadata={
                    'category': category,
                    'has_image': image_base64 is not None,
                    'has_bbox': bbox is not None,
                    **{k: v for k, v in elem_metadata.items()
                       if k not in ['image_base64', 'text_as_html', 'coordinates']}
                }
            ))
            visual_idx += 1

        logger.info(f"Extracted {len(visual_elements)} visual elements from {document_id}")
        return visual_elements

    def process_document(
        self,
        file_path: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """
        Process a document through Unstructured.io pipeline with full feature utilization.

        Args:
            file_path: Path to the document
            custom_metadata: Additional metadata to attach

        Returns:
            ProcessedDocument with chunks, page images, and visual elements
        """
        file_path = str(Path(file_path).resolve())
        filename = Path(file_path).name
        document_id = self._generate_document_id(file_path)

        logger.info(f"Processing document: {filename} (ID: {document_id})")

        # Use Unstructured.io to partition the document with full features
        if file_path.lower().endswith('.pdf'):
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                hi_res_model_name=self.hi_res_model,  # Use YOLOX for better layout detection
                infer_table_structure=True,
                languages=self.ocr_languages,
                include_page_breaks=True,  # Better page tracking
                # Extract visual elements as images
                extract_image_block_types=self.VISUAL_ELEMENT_TYPES if self.extract_visual_elements else None,
                extract_image_block_to_payload=self.extract_visual_elements,  # Base64 in metadata
                # Optional features
                detect_language_per_element=self.detect_language_per_element,
                extract_forms=self.extract_forms,
            )
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            # Excel files - partition each sheet as tables
            elements = partition_xlsx(
                filename=file_path,
                infer_table_structure=True,
            )
        else:
            elements = partition(
                filename=file_path,
                strategy="hi_res",
                languages=self.ocr_languages,
                include_page_breaks=True,
            )

        logger.info(f"Partitioned into {len(elements)} elements")

        # Extract visual elements BEFORE chunking (from raw elements)
        visual_elements = []
        if self.extract_visual_elements:
            visual_elements = self._extract_visual_elements(elements, document_id)

        # Detect document type
        doc_type = self._detect_document_type(filename, elements)

        # Chunk the document
        chunked_elements = chunk_by_title(
            elements,
            max_characters=self.chunk_size,
            overlap=self.chunk_overlap,
            combine_text_under_n_chars=100,
        )

        # Create ProcessedChunks
        chunks = []
        for idx, element in enumerate(chunked_elements):
            chunk_id = f"{document_id}_chunk_{idx}"

            # Extract element metadata
            elem_metadata = element.metadata.to_dict() if hasattr(element.metadata, 'to_dict') else {}

            # Parse coordinates to structured format
            coords = getattr(element.metadata, 'coordinates', None)
            bbox = None
            if coords:
                bbox = BoundingBox.from_coordinates(coords, elem_metadata.get('page_number', 1))

            chunk = ProcessedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=str(element),
                page_number=elem_metadata.get('page_number'),
                chunk_index=idx,
                element_type=element.category if hasattr(element, 'category') else 'Text',
                metadata={
                    'bbox': bbox.to_dict() if bbox else None,
                    'parent_id': elem_metadata.get('parent_id'),
                    'languages': elem_metadata.get('languages'),
                    'text_as_html': elem_metadata.get('text_as_html'),
                    # Include other metadata but exclude large fields
                    **{k: v for k, v in elem_metadata.items()
                       if k not in ['coordinates', 'image_base64', 'text_as_html', 'languages']}
                }
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks")

        # Extract page images for ColPali (full page embeddings)
        page_images = []
        if self.extract_images:
            page_images = self._extract_page_images(file_path)

        # Compile document metadata
        metadata = {
            'filename': filename,
            'file_extension': Path(file_path).suffix,
            'file_size_bytes': os.path.getsize(file_path),
            'total_chunks': len(chunks),
            'total_pages': len(page_images) if page_images else None,
            'total_visual_elements': len(visual_elements),
            'visual_element_types': {
                'tables': sum(1 for v in visual_elements if v.element_type == VisualElementType.TABLE),
                'images': sum(1 for v in visual_elements if v.element_type == VisualElementType.IMAGE),
            },
            'languages': self.ocr_languages,
            'hi_res_model': self.hi_res_model,
            **(custom_metadata or {})
        }

        return ProcessedDocument(
            document_id=document_id,
            filename=filename,
            file_path=file_path,
            document_type=doc_type,
            chunks=chunks,
            page_images=page_images,
            visual_elements=visual_elements,
            metadata=metadata,
            processed_at=datetime.utcnow()
        )

    def process_directory(
        self,
        directory_path: str,
        extensions: List[str] = ['.pdf', '.docx', '.txt', '.png', '.jpg', '.xlsx', '.xls']
    ) -> List[ProcessedDocument]:
        """Process all documents in a directory."""
        documents = []
        directory = Path(directory_path)

        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in extensions:
                try:
                    doc = self.process_document(str(file_path))
                    documents.append(doc)
                    print(f"Processed: {file_path.name} ({len(doc.chunks)} chunks)")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        return documents


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor(
        chunk_size=512,
        chunk_overlap=50,
        extract_images=True
    )

    # Process a single document
    # doc = processor.process_document("path/to/contract.pdf")
    # print(f"Document ID: {doc.document_id}")
    # print(f"Type: {doc.document_type}")
    # print(f"Chunks: {len(doc.chunks)}")
    # print(f"Pages: {len(doc.page_images)}")
