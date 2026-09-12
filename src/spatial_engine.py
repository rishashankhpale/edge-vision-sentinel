import cv2
import numpy as np
import onnxruntime as ort
import time

class SpatialVisionEngine:
    def __init__(self, model_path="models/yolov8n.onnx", conf_threshold=0.45, iou_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.TARGET_CLASS_ID = 0

    def preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = min(640 / h, 640 / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[(640 - nh) // 2 : (640 - nh) // 2 + nh, (640 - nw) // 2 : (640 - nw) // 2 + nw] = resized
        
        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        
        return blob, scale, (640 - nw) // 2, (640 - nh) // 2

    def postprocess(self, outputs, orig_shape, scale, pad_x, pad_y):
        preds = np.squeeze(outputs[0]).T
        
        boxes, scores = [], []
        for pred in preds:
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            score = class_scores[class_id]
            
            if class_id == self.TARGET_CLASS_ID and score > self.conf_threshold:
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                
                x1 = (cx - w / 2 - pad_x) / scale
                y1 = (cy - h / 2 - pad_y) / scale
                x2 = (cx + w / 2 - pad_x) / scale
                y2 = (cy + h / 2 - pad_y) / scale
                
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                scores.append(float(score))

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        final_boxes = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, w, h = boxes[idx]
                final_boxes.append((x, y, x + w, y + h, scores[idx]))
        return final_boxes

    def process_frame(self, frame_bgr, zone_polygon=None):
        t0 = time.perf_counter()
        orig_h, orig_w = frame_bgr.shape[:2]
        
        blob, scale, pad_x, pad_y = self.preprocess(frame_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        detections = self.postprocess(outputs, (orig_h, orig_w), scale, pad_x, pad_y)
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        
        intrusion_detected = False
        annotated = frame_bgr.copy()
        
        poly_np = None
        if zone_polygon is not None and len(zone_polygon) >= 3:
            poly_np = np.array(zone_polygon, np.int32)
            cv2.polylines(annotated, [poly_np], isClosed=True, color=(0, 255, 255), thickness=2)

        for x1, y1, x2, y2, score in detections:
            ground_pt = ((x1 + x2) // 2, y2)
            
            is_inside_zone = False
            if poly_np is not None:
                is_inside_zone = cv2.pointPolygonTest(poly_np, (float(ground_pt[0]), float(ground_pt[1])), False) >= 0
                if is_inside_zone:
                    intrusion_detected = True

            box_color = (0, 0, 255) if is_inside_zone else (0, 255, 0)
            label = f"VIOLATION: {score:.2f}" if is_inside_zone else f"Person: {score:.2f}"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(annotated, ground_pt, radius=5, color=box_color, thickness=-1)
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        return {
            "annotated_frame": annotated,
            "detections_count": len(detections),
            "intrusion_detected": intrusion_detected,
            "latency_ms": round(latency_ms, 2),
            "fps": round(fps, 1)
        }
