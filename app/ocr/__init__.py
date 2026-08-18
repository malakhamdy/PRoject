"""OCI OCR Module - Arabic-aware OCR with PaddleOCR."""

from app.ocr.ocr_engine import (
    OCREngine,
    PaddleOCREngine,
    FallbackOCREngine,
    OCRResult,
    OCRCandidate,
    get_paddle_engine,
    get_fallback_engine,
    reset_engines,
)

__all__ = [
    "OCREngine",
    "PaddleOCREngine",
    "FallbackOCREngine",
    "OCRResult",
    "OCRCandidate",
    "get_paddle_engine",
    "get_fallback_engine",
    "reset_engines",
]
