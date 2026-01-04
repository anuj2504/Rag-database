"""Document processing using Unstructured.io."""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.xlsx import partition_xlsx
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from pdf2image import convert_from_path
from PIL import Image
import io


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
    """Represents a fully processed document."""
    document_id: str
    filename: str
    file_path: str
    document_type: str  # 'contract', 'letter', 'code', etc.
    chunks: List[ProcessedChunk]
    page_images: List[Image.Image]  # For ColPali
    metadata: Dict[str, Any]
    processed_at: datetime


class DocumentProcessor:
    """Process documents using Unstructured.io."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        extract_images: bool = True,
        ocr_languages: List[str] = ["eng"]
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images
        self.ocr_languages = ocr_languages

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

    def process_document(
        self,
        file_path: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """
        Process a document through Unstructured.io pipeline.

        Args:
            file_path: Path to the document
            custom_metadata: Additional metadata to attach

        Returns:
            ProcessedDocument with chunks and images
        """
        file_path = str(Path(file_path).resolve())
        filename = Path(file_path).name
        document_id = self._generate_document_id(file_path)

        # Use Unstructured.io to partition the document
        # For scanned PDFs, this handles OCR automatically
        if file_path.lower().endswith('.pdf'):
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",  # Best for scanned docs
                infer_table_structure=True,
                languages=self.ocr_languages,
                extract_images_in_pdf=False,  # We handle images separately for ColPali
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
            )

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

            chunk = ProcessedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=str(element),
                page_number=elem_metadata.get('page_number'),
                chunk_index=idx,
                element_type=element.category if hasattr(element, 'category') else 'Text',
                metadata={
                    'coordinates': elem_metadata.get('coordinates'),
                    'parent_id': elem_metadata.get('parent_id'),
                    **elem_metadata
                }
            )
            chunks.append(chunk)

        # Extract page images for ColPali
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
            'languages': self.ocr_languages,
            **(custom_metadata or {})
        }

        return ProcessedDocument(
            document_id=document_id,
            filename=filename,
            file_path=file_path,
            document_type=doc_type,
            chunks=chunks,
            page_images=page_images,
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
