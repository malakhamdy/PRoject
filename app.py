"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Streamlit Application

Interactive web interface for processing Egyptian National ID cards with PaddleOCR.
Full pipeline: Detection → Rectification → Localization → OCR → Normalization → Validation → Consistency
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, Any, Optional, List, Tuple
import time
import json

# Import OCI components
from app.pipeline.oci_pipeline import OCIPipeline
from app.ocr.ocr_engine import PaddleOCREngine, get_paddle_engine, OCRCandidate
from app.config.settings import get_config
from app.schemas.models import FieldResult, FieldStatus
from app.normalization.arabic_normalizer import (
    normalize_arabic_text,
    normalize_numeric_candidate,
    normalize_gender_text,
    is_mostly_arabic,
    is_mostly_numeric,
)
from app.validation.nid_validator import NIDValidator, NIDValidationResult
from app.validation.governorate_mapping import get_governorate_by_code
from app.consistency.consistency_engine import ConsistencyEngine, ConsistencyStatus


def initialize_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'ocr_engine' not in st.session_state:
        st.session_state.ocr_engine = None
    if 'nid_validator' not in st.session_state:
        st.session_state.nid_validator = None
    if 'consistency_engine' not in st.session_state:
        st.session_state.consistency_engine = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'result' not in st.session_state:
        st.session_state.result = None


def load_ocr_engine():
    """Load PaddleOCR engine with Arabic support."""
    if st.session_state.ocr_engine is None:
        with st.spinner("🔄 Loading PaddleOCR engine (Arabic language model)..."):
            try:
                st.session_state.ocr_engine = get_paddle_engine(lang="arabic")
                # Initialize the engine
                init_success = st.session_state.ocr_engine.initialize()
                if init_success:
                    st.success("✅ PaddleOCR engine loaded successfully!")
                    st.info("📚 Arabic text recognition ready")
                else:
                    st.error("❌ Failed to initialize PaddleOCR")
                    return None
            except Exception as e:
                st.error(f"❌ Failed to load PaddleOCR: {str(e)}")
                return None
    return st.session_state.ocr_engine


def load_pipeline():
    """Load OCI pipeline."""
    if st.session_state.pipeline is None:
        st.session_state.pipeline = OCIPipeline()
    return st.session_state.pipeline


def load_nid_validator():
    """Load NID validator."""
    if st.session_state.nid_validator is None:
        st.session_state.nid_validator = NIDValidator()
    return st.session_state.nid_validator


def load_consistency_engine():
    """Load consistency engine."""
    if st.session_state.consistency_engine is None:
        validator = load_nid_validator()
        st.session_state.consistency_engine = ConsistencyEngine(nid_validator=validator)
    return st.session_state.consistency_engine


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def draw_field_boxes(image: np.ndarray, fields: Dict[str, FieldResult]) -> np.ndarray:
    """Draw bounding boxes on field regions."""
    viz = image.copy()
    
    colors = {
        'nid': (0, 255, 0),      # Green
        'name': (255, 0, 0),     # Blue
        'dob': (0, 0, 255),      # Red
        'gender': (255, 255, 0), # Cyan
        'governorate': (255, 0, 255),  # Magenta
        'address': (0, 255, 255) # Yellow
    }
    
    field_labels = {
        'nid': 'National ID',
        'name': 'Name',
        'dob': 'Date of Birth',
        'gender': 'Gender',
        'governorate': 'Governorate',
        'address': 'Address'
    }
    
    for field_name, field_result in fields.items():
        if field_result.bbox:
            bbox = field_result.bbox
            color = colors.get(field_name, (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(
                viz,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                color,
                2
            )
            
            # Draw label
            label = f"{field_labels.get(field_name, field_name)}"
            if field_result.value:
                label += f": {field_result.value[:20]}"
            
            cv2.putText(
                viz,
                label,
                (bbox.x1, bbox.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
    
    return viz


def preprocess_field_crop(crop: np.ndarray, field_type: str) -> np.ndarray:
    """
    Apply field-specific preprocessing to optimize OCR accuracy.
    
    Args:
        crop: Field crop image
        field_type: Type of field ('nid', 'name', 'dob', etc.)
    
    Returns:
        Preprocessed image
    """
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    if field_type == 'nid':
        # NID is purely numeric - use binary thresholding
        _, enhanced = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Add slight dilation to connect broken characters
        kernel = np.ones((1, 1), np.uint8)
        enhanced = cv2.dilate(enhanced, kernel, iterations=1)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    elif field_type == 'dob':
        # DOB contains numbers and separators - adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    elif field_type == 'gender':
        # Gender is short Arabic text - mild enhancement
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, enhanced = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    elif field_type in ['governorate', 'address', 'name']:
        # Arabic text fields - preserve grayscale with contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return crop


def extract_and_normalize_field(
    ocr_engine: PaddleOCREngine,
    field_crop: np.ndarray,
    field_type: str
) -> Tuple[Optional[str], float, str]:
    """
    Extract value from field crop using PaddleOCR and apply normalization.
    
    Args:
        ocr_engine: PaddleOCR engine instance
        field_crop: Field crop image
        field_type: Type of field
    
    Returns:
        Tuple of (normalized_value, confidence, raw_value)
    """
    if ocr_engine is None or not ocr_engine.is_initialized():
        return None, 0.0, ""
    
    try:
        # Preprocess based on field type
        enhanced = preprocess_field_crop(field_crop, field_type)
        
        # Run OCR
        candidates = ocr_engine.recognize_field(enhanced, field_type)
        
        if not candidates:
            return None, 0.0, ""
        
        # Get best candidate by confidence
        best = max(candidates, key=lambda c: c.ocr_confidence)
        raw_value = best.value
        confidence = best.ocr_confidence
        
        # Apply field-specific normalization
        normalized_value = normalize_field_value(raw_value, field_type)
        
        return normalized_value, confidence, raw_value
    
    except Exception as e:
        st.warning(f"OCR extraction error for {field_type}: {str(e)}")
        return None, 0.0, ""


def normalize_field_value(raw_value: str, field_type: str) -> Optional[str]:
    """
    Apply field-specific normalization to OCR output.
    
    Args:
        raw_value: Raw OCR output
        field_type: Type of field
    
    Returns:
        Normalized value
    """
    if not raw_value or not isinstance(raw_value, str):
        return None
    
    if field_type == 'nid':
        # NID: Pure numeric normalization
        normalized = normalize_numeric_candidate(raw_value)
        return normalized if normalized else None
    
    elif field_type == 'dob':
        # DOB: Normalize digits and common date formats
        normalized = normalize_arabic_text(raw_value, normalize_digits=True, remove_diacritics_flag=True)
        # Remove non-date characters except digits, /, -, .
        cleaned = ''.join(c for c in normalized if c.isdigit() or c in '/-.')
        return cleaned if cleaned else None
    
    elif field_type == 'gender':
        # Gender: Normalize to 'male' or 'female'
        normalized = normalize_gender_text(raw_value)
        return normalized
    
    elif field_type == 'governorate':
        # Governorate: Normalize Arabic text
        normalized = normalize_arabic_text(raw_value, normalize_digits=False, remove_diacritics_flag=True)
        return normalized if normalized.strip() else None
    
    elif field_type == 'name':
        # Name: Normalize Arabic text but preserve structure
        normalized = normalize_arabic_text(raw_value, normalize_digits=False, remove_diacritics_flag=True, normalize_ws=True)
        return normalized if normalized.strip() else None
    
    elif field_type == 'address':
        # Address: Normalize Arabic text
        normalized = normalize_arabic_text(raw_value, normalize_digits=True, remove_diacritics_flag=True, normalize_ws=True)
        return normalized if normalized.strip() else None
    
    return raw_value


def process_image(image: np.ndarray, progress_bar):
    """
    Process image through full OCI pipeline:
    1. Card Detection
    2. Perspective Rectification
    3. Field Localization
    4. OCR Extraction (PaddleOCR with Arabic support)
    5. Text Normalization
    6. NID Validation
    7. Cross-Field Consistency Check
    """
    results = {
        'success': False,
        'status': '',
        'card_detection': None,
        'rectification': None,
        'fields': {},
        'normalized_fields': {},
        'validation_results': {},
        'consistency_result': None,
        'metrics': None,
        'canonical_image': None,
        'field_crops': {}
    }
    
    try:
        # Step 1: Load pipeline
        pipeline = load_pipeline()
        progress_bar.progress(5, text="✅ Pipeline loaded")
        
        # Step 2: Run detection, rectification, and localization
        pipeline_result = pipeline.process(image)
        progress_bar.progress(25, text="✅ Card detection & localization complete")
        
        results['card_detection'] = pipeline_result.card_detection
        results['rectification'] = pipeline_result.rectification
        results['metrics'] = pipeline_result.metrics
        
        if pipeline_result.rectification and pipeline_result.rectification.canonical_image is not None:
            results['canonical_image'] = pipeline_result.rectification.canonical_image
            canonical = pipeline_result.rectification.canonical_image
            
            # Step 3: Load OCR engine
            ocr_engine = load_ocr_engine()
            if not ocr_engine or not ocr_engine.is_initialized():
                results['status'] = "OCR engine not available"
                return results
            
            progress_bar.progress(40, text="✅ PaddleOCR loaded with Arabic support")
            
            # Step 4: Extract and normalize each field
            field_names = ['nid', 'name', 'dob', 'gender', 'governorate', 'address']
            
            for i, field_name in enumerate(field_names):
                field_result = pipeline_result.fields.get(field_name)
                
                if field_result and field_result.bbox:
                    bbox = field_result.bbox
                    
                    # Extract field crop from canonical image
                    field_crop = canonical[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
                    results['field_crops'][field_name] = field_crop
                    
                    # Run OCR + Normalization
                    normalized_value, confidence, raw_value = extract_and_normalize_field(
                        ocr_engine, field_crop, field_name
                    )
                    
                    # Update field result
                    field_result.value = normalized_value
                    field_result.raw_value = raw_value  # Store raw OCR output
                    field_result.ocr_confidence = confidence
                    
                    if normalized_value:
                        field_result.validation_status = FieldStatus.EXTRACTED
                        results['normalized_fields'][field_name] = normalized_value
                    else:
                        field_result.validation_status = FieldStatus.OCR_FAILED
                        field_result.failure_reason = "OCR extraction failed or normalization produced empty result"
                    
                    results['fields'][field_name] = field_result
                
                progress_bar.progress(40 + int((i+1) * 8), text=f"✅ Extracted & normalized {field_name}")
            
            # Step 5: Validate NID if extracted
            nid_value = results['normalized_fields'].get('nid')
            if nid_value:
                validator = load_nid_validator()
                nid_validation = validator.validate(nid_value)
                results['validation_results']['nid'] = {
                    'is_valid': nid_validation.is_valid,
                    'status': nid_validation.status,
                    'derived_dob': nid_validation.date_of_birth,
                    'derived_gender': nid_validation.gender,
                    'governorate_code': nid_validation.governorate_code,
                    'governorate_name': nid_validation.governorate_name_arabic,
                    'errors': nid_validation.errors,
                    'warnings': nid_validation.warnings,
                }
                progress_bar.progress(90, text="✅ NID validation complete")
            
            # Step 6: Run consistency checks
            consistency_engine = load_consistency_engine()
            consistency_result = consistency_engine.check_all(
                nid_value=nid_value,
                dob_value=results['normalized_fields'].get('dob'),
                gender_value=results['normalized_fields'].get('gender'),
                governorate_value=results['normalized_fields'].get('governorate'),
            )
            results['consistency_result'] = {
                'overall_status': consistency_result.overall_status.value,
                'has_conflicts': consistency_result.has_conflicts,
                'conflict_count': consistency_result.conflict_count,
                'consistent_count': consistency_result.consistent_count,
                'nid_dob_status': consistency_result.nid_dob_status.value,
                'nid_gender_status': consistency_result.nid_gender_status.value,
                'nid_governorate_status': consistency_result.nid_governorate_status.value,
                'summary': consistency_result.summary,
                'recommendations': consistency_result.recommendations,
                'confidence_score': consistency_engine.get_confidence_score(consistency_result),
            }
            progress_bar.progress(95, text="✅ Consistency check complete")
            
            results['success'] = True
            results['status'] = "processing_complete"
            progress_bar.progress(100, text="🎉 Processing complete!")
        
        return results
    
    except Exception as e:
        import traceback
        results['status'] = f"error: {str(e)}"
        results['error_traceback'] = traceback.format_exc()
        return results


def main():
    st.set_page_config(
        page_title="OCI - Egyptian National ID OCR",
        page_icon="🇪🇬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .field-label {
        font-weight: bold;
        color: #333;
        font-size: 1.1rem;
    }
    .field-value {
        font-size: 1.3rem;
        color: #1f77b4;
        margin: 5px 0;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🇪🇬 OCI - Egyptian National ID OCR</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Document Analysis with PaddleOCR</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Engine status
        st.subheader("Engine Status")
        if st.session_state.ocr_engine and st.session_state.ocr_engine.is_initialized():
            st.success("✅ PaddleOCR Ready")
        else:
            st.warning("⏳ PaddleOCR Not Loaded")
            if st.button("Load PaddleOCR Engine"):
                load_ocr_engine()
                st.rerun()
        
        st.divider()
        
        # Info
        st.info("""
        **Supported Fields:**
        - National ID Number (NID)
        - Full Name (Arabic)
        - Date of Birth
        - Gender
        - Governorate
        - Address
        
        **Full Pipeline:**
        1. ✅ Card Detection
        2. ✅ Perspective Rectification
        3. ✅ Field Localization
        4. ✅ PaddleOCR (Arabic)
        5. ✅ Text Normalization
        6. ✅ NID Validation
        7. ✅ Consistency Checks
        
        **Features:**
        - Arabic-Indic digit conversion
        - Diacritics removal
        - Cross-field validation
        - Confidence scoring
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an Egyptian National ID image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of an Egyptian National ID card"
        )
        
        if uploaded_file:
            # Convert to OpenCV format
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is not None:
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Uploaded Image",
                    use_container_width=True
                )
                
                st.write(f"**Image size:** {image.shape[1]} x {image.shape[0]} pixels")
                
                # Process button
                if st.button("🚀 Process ID Card", type="primary", use_container_width=True):
                    progress_bar = st.progress(0, text="Starting processing...")
                    
                    with st.spinner("Processing your image..."):
                        results = process_image(image, progress_bar)
                        st.session_state.result = results
                        st.session_state.processing_complete = True
                        st.rerun()
    
    with col2:
        st.header("📊 Results")
        
        if st.session_state.processing_complete and st.session_state.result:
            results = st.session_state.result
            
            if results['success']:
                st.success("✅ Processing completed successfully!")
                
                # Show metrics
                if results['metrics']:
                    with st.expander("📈 Performance Metrics"):
                        m = results['metrics']
                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Detection Time", f"{m.card_detection_time_ms:.1f}ms")
                        col_m2.metric("Rectification Time", f"{m.rectification_time_ms:.1f}ms")
                        col_m3.metric("Total Time", f"{m.total_time_ms:.1f}ms")
                
                # Show extracted fields
                st.subheader("🔍 Extracted Information")
                
                field_display_names = {
                    'nid': '🆔 National ID Number',
                    'name': '👤 Full Name',
                    'dob': '📅 Date of Birth',
                    'gender': '⚧ Gender',
                    'governorate': '📍 Governorate',
                    'address': '🏠 Address'
                }
                
                for field_name, display_name in field_display_names.items():
                    field_result = results['fields'].get(field_name)
                    
                    with st.container():
                        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="field-label">{display_name}</div>', unsafe_allow_html=True)
                        
                        if field_result and field_result.value:
                            st.markdown(f'<div class="field-value">{field_result.value}</div>', unsafe_allow_html=True)
                            
                            # Confidence indicator
                            conf = field_result.ocr_confidence
                            if conf >= 0.8:
                                conf_class = "confidence-high"
                                conf_text = f"High ({conf:.1%})"
                            elif conf >= 0.5:
                                conf_class = "confidence-medium"
                                conf_text = f"Medium ({conf:.1%})"
                            else:
                                conf_class = "confidence-low"
                                conf_text = f"Low ({conf:.1%})"
                            
                            st.markdown(f'<div class="{conf_class}">Confidence: {conf_text}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="field-value" style="color: #999;">Not extracted</div>', unsafe_allow_html=True)
                            
                            if field_result and field_result.failure_reason:
                                st.warning(f"Issue: {field_result.failure_reason}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Show visualizations
                st.subheader("🖼️ Visualizations")
                
                tab1, tab2, tab3 = st.tabs(["Canonical Card", "Field Localization", "Field Crops"])
                
                with tab1:
                    if results['canonical_image'] is not None:
                        st.image(
                            cv2.cvtColor(results['canonical_image'], cv2.COLOR_BGR2RGB),
                            caption="Rectified Canonical Card",
                            use_container_width=True
                        )
                
                with tab2:
                    if results['canonical_image'] is not None and results['fields']:
                        viz_image = draw_field_boxes(results['canonical_image'], results['fields'])
                        st.image(
                            cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB),
                            caption="Detected Field Regions",
                            use_container_width=True
                        )
                
                with tab3:
                    if results['field_crops']:
                        cols = st.columns(2)
                        for idx, (field_name, crop) in enumerate(results['field_crops'].items()):
                            with cols[idx % 2]:
                                st.image(
                                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                    caption=f"{field_name.upper()} Crop",
                                    use_container_width=True
                                )
                
                # Show validation results
                if results.get('validation_results', {}).get('nid'):
                    st.subheader("✅ NID Validation")
                    nid_val = results['validation_results']['nid']
                    
                    val_col1, val_col2, val_col3 = st.columns(3)
                    
                    with val_col1:
                        status_icon = "✅" if nid_val['is_valid'] else "⚠️"
                        st.metric(f"{status_icon} Validity", nid_val['status'])
                    
                    with val_col2:
                        if nid_val['derived_dob']:
                            st.metric("📅 Derived DOB", nid_val['derived_dob'])
                    
                    with val_col3:
                        if nid_val['derived_gender']:
                            gender_display = "👨 Male" if nid_val['derived_gender'] == 'male' else "👩 Female"
                            st.metric("⚧ Derived Gender", gender_display)
                    
                    # Show governorate info
                    if nid_val.get('governorate_name'):
                        st.info(f"📍 Governorate from NID: {nid_val['governorate_name']} (Code: {nid_val['governorate_code']})")
                    
                    # Show warnings/errors
                    if nid_val.get('warnings'):
                        for warning in nid_val['warnings']:
                            st.warning(f"⚠️ {warning}")
                    
                    if nid_val.get('errors'):
                        for error in nid_val['errors']:
                            st.error(f"❌ {error}")
                
                # Show consistency check results
                if results.get('consistency_result'):
                    st.subheader("🔗 Cross-Field Consistency Check")
                    cons = results['consistency_result']
                    
                    # Overall status
                    overall_icon = {"consistent": "✅", "conflict": "⚠️", "unknown": "❓"}.get(cons['overall_status'], "❓")
                    st.markdown(f"**{overall_icon} Overall Status:** {cons['overall_status'].upper()}")
                    
                    # Confidence score
                    conf_score = cons.get('confidence_score', 0)
                    st.progress(conf_score, text=f"Confidence Score: {conf_score:.1%}")
                    
                    # Summary stats
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    stat_col1.metric("✅ Consistent Checks", cons['consistent_count'])
                    stat_col2.metric("⚠️ Conflicts", cons['conflict_count'])
                    stat_col3.metric("❓ Unknown", cons.get('unknown_count', 0))
                    
                    # Individual check statuses
                    st.markdown("**Detailed Checks:**")
                    check_cols = st.columns(3)
                    
                    dob_icon = {"consistent": "✅", "conflict": "⚠️", "unknown": "❓", "missing": "➖"}.get(cons['nid_dob_status'], "❓")
                    check_cols[0].markdown(f"{dob_icon} **NID ↔ DOB:** {cons['nid_dob_status']}")
                    
                    gender_icon = {"consistent": "✅", "conflict": "⚠️", "unknown": "❓", "missing": "➖"}.get(cons['nid_gender_status'], "❓")
                    check_cols[1].markdown(f"{gender_icon} **NID ↔ Gender:** {cons['nid_gender_status']}")
                    
                    gov_icon = {"consistent": "✅", "conflict": "⚠️", "unknown": "❓", "missing": "➖"}.get(cons['nid_governorate_status'], "❓")
                    check_cols[2].markdown(f"{gov_icon} **NID ↔ Governorate:** {cons['nid_governorate_status']}")
                    
                    # Recommendations
                    if cons.get('recommendations'):
                        st.markdown("**💡 Recommendations:**")
                        for rec in cons['recommendations']:
                            st.info(rec)
                    
                    if cons.get('summary'):
                        st.caption(f"Summary: {cons['summary']}")
                
                # Download results
                st.subheader("💾 Export Results")
                
                # Create comprehensive JSON export
                import json
                export_data = {
                    'status': results['status'],
                    'normalized_fields': results.get('normalized_fields', {}),
                    'raw_fields': {},
                    'validation_results': results.get('validation_results', {}),
                    'consistency_result': results.get('consistency_result'),
                    'metrics': {
                        'card_detection_time_ms': results['metrics'].card_detection_time_ms if results.get('metrics') else None,
                        'rectification_time_ms': results['metrics'].rectification_time_ms if results.get('metrics') else None,
                        'total_time_ms': results['metrics'].total_time_ms if results.get('metrics') else None,
                    } if results.get('metrics') else None,
                }
                
                # Add raw and processed field data
                for field_name, field_result in results['fields'].items():
                    export_data['raw_fields'][field_name] = getattr(field_result, 'raw_value', None)
                    if 'fields' not in export_data:
                        export_data['fields'] = {}
                    export_data['fields'][field_name] = {
                        'normalized_value': field_result.value,
                        'raw_value': getattr(field_result, 'raw_value', None),
                        'confidence': field_result.ocr_confidence if hasattr(field_result, 'ocr_confidence') else None,
                        'status': field_result.validation_status.value if field_result.validation_status else None,
                        'failure_reason': getattr(field_result, 'failure_reason', None),
                    }
                
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📥 Download Full Results (JSON)",
                    data=json_str,
                    file_name="id_extraction_full_results.json",
                    mime="application/json",
                    use_container_width=True
                )
                
            else:
                st.error(f"❌ Processing failed: {results.get('status', 'Unknown error')}")
                
                if results.get('card_detection'):
                    cd = results['card_detection']
                    st.warning(f"Card Detection Status: {cd.status.value if hasattr(cd.status, 'value') else cd.status}")
                    if hasattr(cd, 'failure_reason') and cd.failure_reason:
                        st.write(f"Reason: {cd.failure_reason}")
        
        else:
            st.info("👆 Upload an image and click 'Process ID Card' to see results here.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>OCI - Egyptian National ID Intelligent OCR System | Powered by PaddleOCR</p>
        <p>For demonstration purposes only. Handle personal data responsibly.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
