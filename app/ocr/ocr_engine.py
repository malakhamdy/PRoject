"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
OCR Engine Module

Arabic-aware OCR engine with lazy-loaded PaddleOCR.
Field-specific OCR strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR result for a text region."""
    text: str
    confidence: float
    bbox: Optional[List[int]] = None
    language: str = "ara"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "language": self.language,
        }


@dataclass
class OCRCandidate:
    """
    OCR candidate with raw and normalized values.
    
    Preserves raw OCR output while providing normalized version.
    """
    value: str
    normalized_value: Optional[str] = None
    ocr_confidence: float = 0.0
    preprocessing_variant: str = "default"
    engine: str = "paddleocr"
    raw_output: Optional[str] = None
    validation_score: float = 0.0
    final_score: float = 0.0
    
    # Additional metadata
    char_confidences: List[float] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "normalized_value": self.normalized_value,
            "ocr_confidence": self.ocr_confidence,
            "preprocessing_variant": self.preprocessing_variant,
            "engine": self.engine,
            "raw_output": self.raw_output,
            "validation_score": self.validation_score,
            "final_score": self.final_score,
            "alternatives": self.alternatives,
        }


class OCREngine:
    """
    Base OCR engine interface.
    
    All OCR engines must implement this interface.
    """
    
    def initialize(self) -> bool:
        """Initialize the OCR engine. Returns True if successful."""
        raise NotImplementedError
    
    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        raise NotImplementedError
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """Recognize text in an image region."""
        raise NotImplementedError
    
    def recognize_field(self, image: np.ndarray, field_type: str) -> List[OCRCandidate]:
        """
        Recognize text optimized for a specific field type.
        
        Args:
            image: Field crop image
            field_type: One of 'nid', 'name', 'dob', 'gender', 'governorate', 'address'
            
        Returns:
            List of OCR candidates
        """
        raise NotImplementedError


class PaddleOCREngine(OCREngine):
    """
    PaddleOCR engine implementation with Arabic support.
    
    Lazy-loads models to avoid unnecessary initialization.
    Thread-safe model caching.
    """
    
    def __init__(self, lang: str = "arabic", use_angle_cls: bool = True):
        """
        Initialize PaddleOCR engine configuration.
        
        Models are NOT loaded until initialize() is called.
        
        Args:
            lang: Language code ('arabic', 'en', etc.)
            use_angle_cls: Use angle classification for rotation correction
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._ocr = None
        self._initialized = False
        self._init_error: Optional[str] = None
    
    def initialize(self) -> bool:
        """
        Lazy-load PaddleOCR models.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
        
        if self._init_error:
            logger.warning(f"PaddleOCR previously failed to initialize: {self._init_error}")
            return False
        
        try:
            logger.info(f"Initializing PaddleOCR with language={self.lang}")
            from paddleocr import PaddleOCR
            
            # Configure PaddleOCR
            # Note: lang='arabic' for Arabic script support
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False,  # Suppress verbose logging
            )
            
            self._initialized = True
            logger.info("PaddleOCR initialized successfully")
            return True
            
        except Exception as e:
            self._init_error = str(e)
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if PaddleOCR is initialized."""
        return self._initialized and self._ocr is not None
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """
        Recognize text in an image region.
        
        Args:
            image: Image region (numpy array, BGR or grayscale)
            
        Returns:
            List of OCRResult objects
        """
        if not self.is_initialized():
            if not self.initialize():
                return []
        
        try:
            # PaddleOCR expects BGR or grayscale
            result = self._ocr.ocr(image, cls=self.use_angle_cls)
            
            ocr_results = []
            
            if result and len(result) > 0:
                # Result structure: [[[bbox], (text, confidence)], ...]
                for line in result[0]:  # First page only
                    if line:
                        bbox_points = line[0]
                        text, confidence = line[1]
                        
                        # Convert bbox to [x1, y1, x2, y2]
                        x_coords = [p[0] for p in bbox_points]
                        y_coords = [p[1] for p in bbox_points]
                        bbox = [
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords)),
                            int(max(y_coords)),
                        ]
                        
                        ocr_results.append(OCRResult(
                            text=text,
                            confidence=float(confidence),
                            bbox=bbox,
                            language=self.lang,
                        ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"PaddleOCR recognition error: {e}")
            return []
    
    def recognize_field(self, image: np.ndarray, field_type: str) -> List[OCRCandidate]:
        """
        Recognize text optimized for a specific field type.
        
        Args:
            image: Field crop image
            field_type: One of 'nid', 'name', 'dob', 'gender', 'governorate', 'address'
            
        Returns:
            List of OCRCandidate objects
        """
        ocr_results = self.recognize(image)
        
        if not ocr_results:
            return []
        
        candidates = []
        
        for result in ocr_results:
            candidate = OCRCandidate(
                value=result.text,
                normalized_value=None,  # Will be set by normalization layer
                ocr_confidence=result.confidence,
                preprocessing_variant="default",
                engine="paddleocr",
                raw_output=result.text,
            )
            candidates.append(candidate)
        
        return candidates
    
    def get_full_text(self, image: np.ndarray) -> str:
        """
        Get full recognized text as a single string.
        
        Args:
            image: Image to recognize
            
        Returns:
            Concatenated text from all recognized regions
        """
        results = self.recognize(image)
        texts = [r.text for r in results if r.text.strip()]
        return "\n".join(texts)


class FallbackOCREngine(OCREngine):
    """
    Optional fallback OCR engine.
    
    Only used when PaddleOCR fails or produces poor results.
    Currently a placeholder - can be extended with EasyOCR or other engines.
    """
    
    def __init__(self):
        self._initialized = False
        logger.warning("FallbackOCR is a placeholder - no actual fallback engine configured")
    
    def initialize(self) -> bool:
        """Fallback initialization (placeholder)."""
        return False
    
    def is_initialized(self) -> bool:
        return self._initialized
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """Not implemented."""
        return []
    
    def recognize_field(self, image: np.ndarray, field_type: str) -> List[OCRCandidate]:
        """Not implemented."""
        return []


# Global engine instances with lazy loading
_paddle_engine: Optional[PaddleOCREngine] = None
_fallback_engine: Optional[FallbackOCREngine] = None


def get_paddle_engine(lang: str = "arabic") -> PaddleOCREngine:
    """
    Get or create the global PaddleOCR engine instance.
    
    Args:
        lang: Language for OCR ('arabic', 'en', etc.)
        
    Returns:
        PaddleOCREngine instance
    """
    global _paddle_engine
    
    if _paddle_engine is None:
        _paddle_engine = PaddleOCREngine(lang=lang)
    
    return _paddle_engine


def get_fallback_engine() -> FallbackOCREngine:
    """Get or create the global fallback OCR engine instance."""
    global _fallback_engine
    
    if _fallback_engine is None:
        _fallback_engine = FallbackOCREngine()
    
    return _fallback_engine


def reset_engines() -> None:
    """Reset all OCR engine instances (useful for testing)."""
    global _paddle_engine, _fallback_engine
    _paddle_engine = None
    _fallback_engine = None
