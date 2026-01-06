"""
ColPali embeddings for visual document understanding.

ColPali processes documents as images and generates multi-vector embeddings
that capture both visual and textual information. This is especially useful for:
- Scanned PDFs with poor OCR
- Documents with complex layouts (tables, forms, charts)
- Handwritten documents
- Documents where visual structure matters
"""
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch


class ColPaliEmbedder:
    """
    ColPali visual document embedder.

    Generates multi-vector embeddings per page/image.
    Each image produces multiple patch embeddings (typically 1030 patches).
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: Optional[str] = None,
        max_image_size: int = 1024
    ):
        from colpali_engine.models import ColPali
        from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor

        self.model_name = model_name
        self.max_image_size = max_image_size

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading ColPali model on {self.device}...")

        # Load model and processor
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        ).to(self.device).eval()

        self.processor = ColPaliProcessor.from_pretrained(model_name)

        # ColPali embedding dimension (per patch)
        self._dimension = 128

    @property
    def dimension(self) -> int:
        """Dimension of each patch embedding."""
        return self._dimension

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Resize image if too large."""
        if max(image.size) > self.max_image_size:
            image.thumbnail(
                (self.max_image_size, self.max_image_size),
                Image.Resampling.LANCZOS
            )
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def embed_images(
        self,
        images: List[Image.Image],
        batch_size: int = 8
    ) -> List[np.ndarray]:
        """
        Embed document page images.

        Args:
            images: List of PIL Images (document pages)
            batch_size: Batch size for processing

        Returns:
            List of embedding arrays, each shape (num_patches, 128)
            where num_patches is typically ~1030 per image
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # Preprocess images
            processed_images = [self._preprocess_image(img) for img in batch_images]

            # Process through ColPali
            with torch.no_grad():
                batch_inputs = self.processor.process_images(processed_images)
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

                embeddings = self.model(**batch_inputs)

                # Move to CPU and convert to numpy
                for emb in embeddings:
                    # emb shape: (num_patches, 128)
                    all_embeddings.append(emb.cpu().float().numpy())

        return all_embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a text query for retrieval.

        Args:
            query: Search query text

        Returns:
            Query embedding array, shape (num_tokens, 128)
        """
        with torch.no_grad():
            query_inputs = self.processor.process_queries([query])
            query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}

            query_embedding = self.model(**query_inputs)

            # Return first (only) query embedding
            return query_embedding[0].cpu().float().numpy()

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: List[np.ndarray]
    ) -> List[float]:
        """
        Compute MaxSim scores between query and documents.

        ColPali uses late interaction (like ColBERT):
        - For each query token, find max similarity with any document patch
        - Sum these max similarities

        Args:
            query_embedding: Shape (num_query_tokens, 128)
            document_embeddings: List of arrays, each (num_patches, 128)

        Returns:
            List of similarity scores
        """
        scores = []

        for doc_emb in document_embeddings:
            # Compute token-to-patch similarities
            # Shape: (num_query_tokens, num_patches)
            similarities = np.dot(query_embedding, doc_emb.T)

            # MaxSim: for each query token, take max over all patches
            max_sims = similarities.max(axis=1)

            # Sum of max similarities
            score = max_sims.sum()
            scores.append(float(score))

        return scores


class ColPaliLiteEmbedder:
    """
    Lighter ColPali alternative using average pooling.

    Stores single vector per page instead of multi-vector.
    Trade-off: Less accurate but much more storage efficient.
    """

    def __init__(self, base_embedder: Optional[ColPaliEmbedder] = None, **kwargs):
        self.base_embedder = base_embedder or ColPaliEmbedder(**kwargs)

    @property
    def dimension(self) -> int:
        return self.base_embedder.dimension

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Embed images with average pooling to single vector.

        Returns:
            Array of shape (num_images, 128)
        """
        multi_embeddings = self.base_embedder.embed_images(images)

        # Average pool each image's patches to single vector
        single_embeddings = []
        for emb in multi_embeddings:
            # emb shape: (num_patches, 128)
            avg_emb = emb.mean(axis=0)
            # Normalize
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            single_embeddings.append(avg_emb)

        return np.array(single_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with average pooling."""
        query_emb = self.base_embedder.embed_query(query)
        avg_emb = query_emb.mean(axis=0)
        return avg_emb / np.linalg.norm(avg_emb)


# Example usage
if __name__ == "__main__":
    from PIL import Image

    # Initialize embedder
    embedder = ColPaliEmbedder()

    # Example: embed a document image
    # img = Image.open("document_page.png")
    # page_embeddings = embedder.embed_images([img])
    # print(f"Page embeddings shape: {page_embeddings[0].shape}")

    # Query
    # query_emb = embedder.embed_query("What is the contract value?")
    # print(f"Query embedding shape: {query_emb.shape}")

    # Compute similarity
    # scores = embedder.compute_similarity(query_emb, page_embeddings)
    # print(f"Similarity scores: {scores}")
