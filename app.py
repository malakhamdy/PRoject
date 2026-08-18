"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Streamlit Application

Interactive web interface for processing Egyptian National ID cards with PaddleOCR.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, Any, Optional
import time

# Import OCI components
from app.pipeline.oci_pipeline import OCIPipeline
from app.ocr.ocr_engine import PaddleOCREngine, get_paddle_engine
from app.config.settings import get_config
from app.schemas.models import FieldResult, FieldStatus


def initialize_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'ocr_engine' not in st.session_state:
        st.session_state.ocr_engine = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'result' not in st.session_state:
        st.session_state.result = None


def load_ocr_engine():
    """Load PaddleOCR engine."""
    if st.session_state.ocr_engine is None:
        with st.spinner("Loading PaddleOCR engine (this may take a moment)..."):
            try:
                st.session_state.ocr_engine = get_paddle_engine(lang="arabic")
                # Initialize the engine
                st.session_state.ocr_engine.initialize()
                st.success("✅ PaddleOCR engine loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load PaddleOCR: {str(e)}")
                return None
    return st.session_state.ocr_engine


def load_pipeline():
    """Load OCI pipeline."""
    if st.session_state.pipeline is None:
        st.session_state.pipeline = OCIPipeline()
    return st.session_state.pipeline


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


def extract_field_value(ocr_engine: PaddleOCREngine, field_crop: np.ndarray, field_type: str) -> tuple:
    """
    Extract value from a field crop using PaddleOCR.
    Returns (value, confidence)
    """
    if ocr_engine is None or not ocr_engine.is_initialized():
        return None, 0.0
    
    try:
        # Apply preprocessing based on field type
        if field_type == 'nid':
            # NID is numeric - enhance contrast
            if len(field_crop.shape) == 3:
                gray = cv2.cvtColor(field_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = field_crop
            _, enhanced = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = field_crop
        
        # Run OCR
        candidates = ocr_engine.recognize_field(enhanced, field_type)
        
        if candidates:
            # Get best candidate
            best = max(candidates, key=lambda c: c.ocr_confidence)
            return best.value, best.ocr_confidence
        
        return None, 0.0
    
    except Exception as e:
        st.warning(f"OCR extraction error for {field_type}: {str(e)}")
        return None, 0.0


def process_image(image: np.ndarray, progress_bar):
    """Process image through full pipeline."""
    results = {
        'success': False,
        'status': '',
        'card_detection': None,
        'rectification': None,
        'fields': {},
        'metrics': None,
        'canonical_image': None,
        'field_crops': {}
    }
    
    try:
        # Load pipeline
        pipeline = load_pipeline()
        progress_bar.progress(10, text="Pipeline loaded")
        
        # Run detection, rectification, and localization
        pipeline_result = pipeline.process(image)
        progress_bar.progress(40, text="Card detection & localization complete")
        
        results['card_detection'] = pipeline_result.card_detection
        results['rectification'] = pipeline_result.rectification
        results['metrics'] = pipeline_result.metrics
        
        if pipeline_result.rectification and pipeline_result.rectification.canonical_image is not None:
            results['canonical_image'] = pipeline_result.rectification.canonical_image
            canonical = pipeline_result.rectification.canonical_image
            
            # Load OCR engine
            ocr_engine = load_ocr_engine()
            progress_bar.progress(60, text="PaddleOCR loaded")
            
            if ocr_engine and ocr_engine.is_initialized():
                # Extract each field
                field_names = ['nid', 'name', 'dob', 'gender', 'governorate', 'address']
                
                for i, field_name in enumerate(field_names):
                    field_result = pipeline_result.fields.get(field_name)
                    
                    if field_result and field_result.bbox:
                        bbox = field_result.bbox
                        
                        # Extract field crop from canonical image
                        field_crop = canonical[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
                        results['field_crops'][field_name] = field_crop
                        
                        # Run OCR on field
                        value, confidence = extract_field_value(ocr_engine, field_crop, field_name)
                        
                        # Update field result
                        field_result.value = value
                        field_result.ocr_confidence = confidence
                        
                        if value:
                            field_result.validation_status = FieldStatus.EXTRACTED
                        else:
                            field_result.validation_status = FieldStatus.OCR_FAILED
                        
                        results['fields'][field_name] = field_result
                    
                    progress_bar.progress(60 + int((i+1) * 6), text=f"Extracted {field_name}")
                
                results['success'] = True
                results['status'] = "extraction_complete"
                progress_bar.progress(100, text="Processing complete!")
        
        return results
    
    except Exception as e:
        results['status'] = f"error: {str(e)}"
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
        
        **Features:**
        - Automatic card detection
        - Perspective correction
        - Arabic text recognition
        - Field-specific extraction
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
                
                # Download results
                st.subheader("💾 Export Results")
                
                # Create JSON export
                import json
                export_data = {
                    'status': results['status'],
                    'fields': {}
                }
                
                for field_name, field_result in results['fields'].items():
                    export_data['fields'][field_name] = {
                        'value': field_result.value,
                        'confidence': field_result.ocr_confidence if hasattr(field_result, 'ocr_confidence') else None,
                        'status': field_result.validation_status.value if field_result.validation_status else None
                    }
                
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json_str,
                    file_name="id_extraction_results.json",
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
