"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
API Routes

Provides REST API endpoints for the OCI system.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import logging
from typing import Optional

from app.config import get_config, set_config, AppConfig
from app.schemas.models import PipelineResult
from app.pipeline import OCIPipeline


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="OCI - Egyptian National ID OCR System",
        description="Egyptian National ID Intelligent OCR, Validation and Document Analysis System",
        version="0.3.0",  # Phase 3 complete
    )
    
    # Initialize pipeline
    pipeline = OCIPipeline()
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "phase": "3-localization"}
    
    @app.post("/api/v1/ocr")
    async def process_ocr(
        file: UploadFile = File(...),
        debug: bool = Query(default=False, description="Enable debug mode")
    ):
        """
        Process an Egyptian National ID image.
        
        Phases 1-3: Returns localization results only (no OCR extraction yet).
        
        Returns:
            Pipeline result with:
            - Card detection status and bounding box
            - Rectification confidence
            - Field localization results (bbox + confidence for each field)
            - Performance metrics
            
        Note: OCR values will be null in Phase 3.
        """
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Expected image, got {file.content_type}"
            )
        
        try:
            # Read image
            contents = await file.read()
            image_array = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to decode image"
                )
            
            # Configure debug mode if requested
            if debug:
                config = get_config()
                config.debug.enabled = True
                set_config(config)
            
            # Process through pipeline
            result = pipeline.process(image)
            
            # Convert to dict for JSON response
            response_data = result.to_dict()
            
            # Don't include binary image data in JSON response
            # Debug images would need to be saved to disk and referenced by URL
            if "debug_artifacts" in response_data:
                # Remove binary image data from response
                response_data["debug_artifacts"] = {
                    "enabled": debug,
                    "note": "Debug images available in debug directory when running locally"
                }
            
            status_code = 200 if result.success else 206  # 206 Partial Content for partial success
            
            return JSONResponse(
                status_code=status_code,
                content=response_data
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"OCR processing error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Processing error: {str(e)}"
            )
    
    @app.get("/api/v1/status")
    async def get_status():
        """Get current system status and configuration."""
        config = get_config()
        
        return {
            "phase": "3-localization",
            "phases_completed": ["1-foundation", "2-card-detection-rectification", "3-field-localization"],
            "phases_pending": ["4-ocr", "5-extraction", "6-validation", "7-consistency"],
            "configuration": {
                "debug_enabled": config.debug.enabled,
                "canonical_dimensions": {
                    "width": config.card.canonical_width,
                    "height": config.card.canonical_height,
                },
                "localization_fields": [
                    "nid",
                    "name", 
                    "dob",
                    "gender",
                    "governorate",
                    "address",
                ],
            },
        }
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
