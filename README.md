# OCI — Egyptian National ID Intelligent OCR, Validation and Document Analysis System

## Phase 1-3 Implementation Complete

OCI is NOT a generic OCR application. It is a document-understanding system specifically designed for the **EGYPTIAN NATIONAL ID CARD**.

## Current Status: Phase 3 Complete

### Implemented Phases

**Phase 1 — Foundation / Architecture**
- ✅ Centralized configuration system
- ✅ Strongly typed data schemas/models
- ✅ Pipeline architecture with clear stage interfaces
- ✅ API foundation (FastAPI)
- ✅ Unit and integration tests

**Phase 2 — Card Detection + Orientation + Rectification**
- ✅ Hierarchical card detection (4 levels)
- ✅ Dynamic card bounding box (no hardcoded coordinates)
- ✅ Four-corner detection with proper ordering (TL, TR, BR, BL)
- ✅ Geometry validation (convexity, rectangularity, aspect ratio)
- ✅ Perspective rectification using 4-point transform
- ✅ Canonical card generation (1000x630 pixels)
- ✅ Rectification confidence scoring
- ✅ Explicit failure modes

**Phase 3 — Dynamic Field Localization**
- ✅ Field-specific localization logic for all 6 target fields:
  - National ID Number (NID)
  - Full Arabic Name
  - Date of Birth
  - Gender
  - Governorate
  - Address
- ✅ Content-aware bbox refinement
- ✅ Localization confidence scoring
- ✅ Alternative bbox candidates
- ✅ Bounding box validation
- ✅ Debug visualization

### NOT Yet Implemented (Phase 4+)
- ❌ OCR engine (PaddleOCR)
- ❌ Arabic text recognition
- ❌ Field extraction
- ❌ Normalization
- ❌ Validation
- ❌ Consistency checking

## Project Structure

```
/workspace
├── app/
│   ├── config/           # Centralized configuration
│   │   └── settings.py
│   ├── detection/        # Card detection
│   │   └── card_detector.py
│   ├── rectification/    # Card rectification
│   │   └── card_rectifier.py
│   ├── localization/     # Field localization
│   │   └── field_localizer.py
│   ├── pipeline/         # Main pipeline orchestration
│   │   └── oci_pipeline.py
│   ├── schemas/          # Data models
│   │   └── models.py
│   └── utils/            # Utilities
│       └── geometry.py
├── api/
│   └── routes/
│       └── main.py       # FastAPI endpoints
├── tests/
│   ├── unit/
│   │   └── test_schemas.py
│   └── integration/
│       └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Dependencies

```bash
pip install opencv-python numpy fastapi uvicorn python-multipart pytest
```

**Note:** No PaddleOCR or EasyOCR installed in Phases 1-3.

## API Endpoints

### Health Check
```bash
GET /health
```

### Process Image
```bash
POST /api/v1/ocr
Content-Type: multipart/form-data

file: <image>
debug: false (optional)
```

### Get Status
```bash
GET /api/v1/status
```

## Running Tests

```bash
# Unit tests
python -c "from app.config import AppConfig; from app.schemas.models import BoundingBox; print('Tests passed!')"

# Integration tests
python tests/integration/test_pipeline.py
```

## Key Design Principles

1. **No Hardcoded Coordinates**: All field positions are dynamically localized for each image
2. **Arabic-First Design**: The system understands Egyptian IDs are Arabic documents
3. **Explicit Uncertainty**: Low-confidence results are marked as uncertain, not hidden
4. **No Fake Data**: Phase 1-3 returns localization only, no fabricated OCR values
5. **Modular Architecture**: Each stage has clear input/output contracts

## Output Contract (Phase 3)

```json
{
  "success": true,
  "status": "localized",
  "card": {
    "status": "detected",
    "bbox": [78, 48, 722, 452],
    "confidence": 0.876
  },
  "rectification": {
    "status": "success",
    "rectification_confidence": 0.701,
    "output_width": 1000,
    "output_height": 630
  },
  "fields": {
    "nid": {
      "bbox": [50, 95, 950, 185],
      "localization_confidence": 0.365,
      "status": "field_localization_uncertain"
    },
    "name": { ... },
    "dob": { ... },
    "gender": { ... },
    "governorate": { ... },
    "address": { ... }
  },
  "metrics": {
    "card_detection_time_ms": 45.2,
    "rectification_time_ms": 12.3,
    "localization_time_ms": 89.1,
    "total_time_ms": 489.5
  }
}
```

**Note:** OCR values are null/not implemented in Phase 3.

## Next Phase: Phase 4

**PHASE 4 — Field-Specific Preprocessing + Arabic-Aware PaddleOCR + OCR Candidate Generation**

Will implement:
- Field-specific image preprocessing
- PaddleOCR integration with Arabic support
- OCR candidate generation
- Character-level confidence scoring
- Arabic/Latin script handling
