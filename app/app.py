import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.spatial_engine import SpatialVisionEngine

st.set_page_config(
    page_title="Edge Vision Sentinel",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Edge Vision Sentinel: Real-Time Spatial Safety Engine")
st.caption("Hardware-optimized ONNX Runtime inference with polygon-defined exclusion zone analytics.")

@st.cache_resource
def load_engine():
    return SpatialVisionEngine(model_path="models/yolov8n.onnx")

engine = load_engine()

with st.sidebar:
    st.header("⚙️ Inference & Zone Controls")
    conf_thresh = st.slider("Detection Confidence", 0.2, 0.9, 0.45, 0.05)
    engine.conf_threshold = conf_thresh
    
    st.subheader("🚧 Exclusion Zone Bounds")
    zone_x1 = st.slider("Zone Top-Left X %", 0, 100, 20)
    zone_y1 = st.slider("Zone Top-Left Y %", 0, 100, 30)
    zone_x2 = st.slider("Zone Bottom-Right X %", 0, 100, 80)
    zone_y2 = st.slider("Zone Bottom-Right Y %", 0, 100, 90)
    
    st.markdown("---")
    input_mode = st.radio("Telemetry Source:", ["Upload Test Image", "Live Webcam Feed"])

def get_relative_polygon(w, h):
    return [
        (int(w * (zone_x1 / 100)), int(h * (zone_y1 / 100))),
        (int(w * (zone_x2 / 100)), int(h * (zone_y1 / 100))),
        (int(w * (zone_x2 / 100)), int(h * (zone_y2 / 100))),
        (int(w * (zone_x1 / 100)), int(h * (zone_y2 / 100)))
    ]

if input_mode == "Upload Test Image":
    uploaded = st.file_uploader("Select an image to analyze:", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        zone = get_relative_polygon(w, h)
        
        result = engine.process_frame(frame_bgr, zone_polygon=zone)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Subjects Count", result["detections_count"])
        m2.metric("Inference Latency", f"{result['latency_ms']} ms")
        m3.metric("Effective FPS", f"{result['fps']}")
        m4.metric("Safety Zone State", "BREACH DETECTED" if result["intrusion_detected"] else "SECURE")
        
        if result["intrusion_detected"]:
            st.error("🚨 PERIMETER BREACH: Subject detected inside restricted hazard polygon!")
        else:
            st.success("✅ PERIMETER SECURE: No unauthorized entities detected inside polygon.")

        annotated_rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

elif input_mode == "Live Webcam Feed":
    st.info("Ensure browser camera access is granted. Toggle the checkbox below to stream inference.")
    run_cam = st.checkbox("Initialize Camera Stream")
    FRAME_WINDOW = st.image([])
    
    if run_cam:
        cap = cv2.VideoCapture(0)
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access camera stream.")
                break
            h, w = frame.shape[:2]
            zone = get_relative_polygon(w, h)
            result = engine.process_frame(frame, zone_polygon=zone)
            
            disp = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(disp)
            time.sleep(0.01)
        cap.release()
