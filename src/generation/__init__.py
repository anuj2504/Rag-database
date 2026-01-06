"""
Generation module for VLM-based answer generation.

This module provides multimodal answer generation using Vision Language Models
(VLMs) like Google Gemini. It takes retrieved text chunks and images to
generate comprehensive answers with source citations.
"""

from src.generation.vlm_generator import (
    GeminiVLMGenerator,
    GenerationResult,
    create_vlm_generator,
)

__all__ = [
    "GeminiVLMGenerator",
    "GenerationResult",
    "create_vlm_generator",
]
