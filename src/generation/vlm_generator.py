"""
VLM Generator for multimodal RAG answer generation.

Uses Google Gemini to generate answers from text context and images.
This enables visual question answering for documents with tables, charts,
and figures.

Based on the ColPALI Meets DocLayNet approach for multimodal document QA.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import logging
import time
import os

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from VLM generation."""
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    generation_time_ms: int
    images_used: int
    sources_used: int
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class GeminiVLMGenerator:
    """
    Generate answers using Google Gemini with multimodal support.

    Supports both text-only and text+image context for RAG.

    Usage:
        generator = GeminiVLMGenerator()
        result = generator.generate(
            query="What does the table show?",
            text_context=["Context chunk 1...", "Context chunk 2..."],
            images=[page_image1, page_image2],
        )
        print(result.answer)
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful document assistant for an enterprise document management system. Answer questions based on the provided context from documents.

Rules:
- Only use information from the provided context (text and images)
- If the context doesn't contain the answer, say "I cannot find this information in the provided documents"
- Cite specific sources, sections, or page numbers when possible
- Be concise but complete
- CRITICAL: Extract and include ALL numerical values, measurements, dimensions, and specifications from the context
  - Include exact numbers (e.g., "11.00m", "7 metres", "300 metres")
  - Include ratios and slopes (e.g., "1V:3H", "1:4")
  - Include ranges and tolerances
  - Never omit specific values that answer the question
- If images are provided, analyze them carefully for tables, charts, figures, and visual information
- When referencing visual content, describe what you see and how it answers the question
- Format your response clearly with bullet points or numbered lists when appropriate"""

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize Gemini VLM generator.

        Args:
            model: Gemini model name (gemini-2.0-flash-exp, gemini-1.5-pro, etc.)
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            system_prompt: Custom system prompt for generation
        """
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for VLM generation. "
                "Install with: pip install google-generativeai"
            )

        # Get API key
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Configure API
        genai.configure(api_key=api_key)

        self.model_name = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Create model with system instruction
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=self.system_prompt,
        )

        logger.info(f"Initialized Gemini VLM generator with model: {model}")

    def generate(
        self,
        query: str,
        text_context: List[str],
        images: Optional[List[Image.Image]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> GenerationResult:
        """
        Generate answer from multimodal context.

        Args:
            query: User question
            text_context: List of retrieved text chunks
            images: Optional list of page/table images (PIL Images)
            metadata: Optional metadata for each chunk (for citations)
            max_tokens: Maximum response tokens
            temperature: Generation temperature (lower = more focused)

        Returns:
            GenerationResult with answer and metadata
        """
        start_time = time.time()

        # Build context string with source info
        context_parts = []
        for i, text in enumerate(text_context):
            source_info = ""
            if metadata and i < len(metadata):
                meta = metadata[i]
                if meta.get("filename"):
                    source_info = f" [Source: {meta['filename']}"
                    if meta.get("page_number"):
                        source_info += f", Page {meta['page_number']}"
                    if meta.get("section_title"):
                        source_info += f", Section: {meta['section_title']}"
                    source_info += "]"
            context_parts.append(f"--- Context {i+1}{source_info} ---\n{text}")

        context_str = "\n\n".join(context_parts)

        # Build prompt
        if images:
            prompt = f"""Based on the following text context AND the provided document images, answer the question.

TEXT CONTEXT:
{context_str}

DOCUMENT IMAGES: {len(images)} image(s) from relevant document pages are provided below.
Analyze both the text context and images to provide a comprehensive answer.

IMPORTANT: Include ALL specific numerical values, measurements, dimensions, ratios, and specifications found in the context. Do not omit any numbers.

Question: {query}

Answer:"""
        else:
            prompt = f"""Based on the following context, answer the question.

{context_str}

IMPORTANT: Include ALL specific numerical values, measurements, dimensions, ratios, and specifications found in the context. Do not omit any numbers.

Question: {query}

Answer:"""

        # Build content list for Gemini (text + images)
        content = [prompt]

        # Add images if provided
        images_used = 0
        if images:
            for i, img in enumerate(images[:10]):  # Limit to 10 images
                try:
                    # Ensure image is in RGB mode
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    content.append(img)
                    images_used += 1
                except Exception as e:
                    logger.warning(f"Failed to add image {i}: {e}")

        # Generate response
        try:
            response = self.model.generate_content(
                content,
                generation_config=self._genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            answer = response.text

            # Get token counts from usage metadata
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

            generation_time = int((time.time() - start_time) * 1000)

            return GenerationResult(
                answer=answer,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time_ms=generation_time,
                images_used=images_used,
                sources_used=len(text_context),
            )

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            generation_time = int((time.time() - start_time) * 1000)

            return GenerationResult(
                answer="",
                model=self.model_name,
                input_tokens=0,
                output_tokens=0,
                generation_time_ms=generation_time,
                images_used=images_used,
                sources_used=len(text_context),
                error=str(e),
            )

    def generate_with_visual_elements(
        self,
        query: str,
        text_context: List[str],
        page_images: Optional[List[Image.Image]] = None,
        visual_elements: Optional[List[Image.Image]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> GenerationResult:
        """
        Generate answer with separate page images and visual elements.

        This method allows passing both full page images and cropped
        visual elements (tables, figures) for more precise visual context.

        Args:
            query: User question
            text_context: List of retrieved text chunks
            page_images: Full page images from ColPali retrieval
            visual_elements: Cropped tables/figures from visual element retrieval
            metadata: Optional metadata for each chunk
            max_tokens: Maximum response tokens
            temperature: Generation temperature

        Returns:
            GenerationResult with answer and metadata
        """
        # Combine images: visual elements first (more specific), then page images
        combined_images = []

        if visual_elements:
            combined_images.extend(visual_elements[:5])  # Up to 5 visual elements

        if page_images:
            remaining_slots = 10 - len(combined_images)
            combined_images.extend(page_images[:remaining_slots])

        return self.generate(
            query=query,
            text_context=text_context,
            images=combined_images if combined_images else None,
            metadata=metadata,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def create_vlm_generator(
    provider: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> GeminiVLMGenerator:
    """
    Factory function to create VLM generator.

    Args:
        provider: VLM provider (currently only "gemini" supported)
        model: Model name override
        api_key: API key override

    Returns:
        Configured VLM generator instance
    """
    if provider == "gemini":
        return GeminiVLMGenerator(
            model=model or os.getenv("VLM_MODEL", "gemini-2.0-flash-exp"),
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown VLM provider: {provider}. Currently only 'gemini' is supported.")


# Example usage
if __name__ == "__main__":
    # Test generation
    generator = GeminiVLMGenerator()

    # Text-only test
    result = generator.generate(
        query="What is the contract value?",
        text_context=[
            "The total contract value is $1,500,000 for a period of 3 years.",
            "Payment shall be made in quarterly installments of $125,000.",
        ],
    )

    print(f"Answer: {result.answer}")
    print(f"Model: {result.model}")
    print(f"Generation time: {result.generation_time_ms}ms")
    print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
