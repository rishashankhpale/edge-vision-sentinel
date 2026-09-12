# 👁️ Edge Vision Sentinel: Real-Time Spatial Safety & Intrusion Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20CPU-purple.svg)](https://onnxruntime.ai/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Model-YOLOv8%20Nano-yellow.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

A low-latency, edge-optimized Computer Vision pipeline designed for industrial hazard safety and spatial analytics. It leverages CPU-accelerated ONNX Runtime inference, dynamic polygon exclusion zones, and ray-casting intrusion analytics to provide real-time perimeter monitoring.

---

## 🎯 System Architecture & Engineering Highlights

1. **Edge-Optimized CPU Graph Execution:**
   * PyTorch YOLOv8 weights are exported into a standalone **ONNX graph**, decoupling deployment from large deep learning runtime dependencies.
   * Leverages **multi-threaded ONNX Runtime** (`intra_op_num_threads=4`, sequential execution mode, and Level-All graph optimization) to deliver real-time sub-50ms inference on commodity multi-core CPUs.

2. **Spatial Geometry & Intrusion Detection:**
   * Custom pre-processing with aspect-ratio preserved letterboxing and float32 normalization.
   * Post-processing integrates confidence filtering and Non-Maximum Suppression (NMS, IoU = 0.50).
   * Calculates intrusion using OpenCV ray-casting (`cv2.pointPolygonTest`) mapped strictly to the **bottom-center ground contact point** of each subject bounding box, eliminating false alarms caused by upper-body overhang or camera perspective angles.

3. **Dual Telemetry Interface:**
   * Built with Streamlit to support both static image auditing and client-side browser camera live streaming (`st.camera_input`).
   * Displays live metrics: subject counts, per-frame inference latency (ms), effective throughput (FPS), and automated breach alerts.

---

## 📂 Project Structure

```text
edge-vision-sentinel/
├── app/
│   └── app.py                 # Streamlit operational dashboard & UI
├── models/
│   └── yolov8n.onnx           # Compiled ONNX execution graph
├── src/
│   ├── export_model.py        # Model export & ONNX compilation script
│   └── spatial_engine.py      # Core inference, NMS, & polygon ray-casting engine
├── .gitignore
├── README.md
└── requirements.txt
