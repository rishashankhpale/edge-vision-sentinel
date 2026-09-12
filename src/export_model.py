from ultralytics import YOLO
import os

def export_to_onnx():
    print("Loading baseline YOLOv8 nano model...")
    model = YOLO("yolov8n.pt")
    
    os.makedirs("models", exist_ok=True)
    print("Exporting model to standard ONNX format...")
    onnx_path = model.export(format="onnx", imgsz=640, dynamic=False)
    
    # Place artifact in models directory
    dest_path = os.path.join("models", "yolov8n.onnx")
    if os.path.exists(onnx_path) and os.path.abspath(onnx_path) != os.path.abspath(dest_path):
        os.replace(onnx_path, dest_path)
    
    print(f"ONNX Model ready at: {dest_path}")

if __name__ == "__main__":
    export_to_onnx()
