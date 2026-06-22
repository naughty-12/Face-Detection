"""Real-time face detection — webcam / video file inference"""
import os
import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
from collections import deque

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(DEPLOY_DIR, "best_model_v2.onnx")


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw, dh = dw//2, dh//2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img, (r, dw, dh)


def scale_boxes(boxes, original_shape, ratio_pad):
    r, dw, dh = ratio_pad
    boxes[:, [0,2]] -= dw
    boxes[:, [1,3]] -= dh
    boxes[:, :4] /= r
    boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, original_shape[1])
    boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, original_shape[0])
    return boxes


def parse_onnx_output(output, original_shape, ratio_pad, conf_threshold=0.25):
    output = output[0].transpose(1, 0)
    boxes_xywh = output[:, :4]
    scores = output[:, 4:].max(axis=1)
    mask = scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    if len(boxes_xywh) == 0:
        return np.zeros((0,4)), np.zeros(0)

    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2]/2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3]/2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]/2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]/2
    boxes_xyxy = scale_boxes(boxes_xyxy, original_shape, ratio_pad)

    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), 0.25, 0.45)
    if len(indices) > 0:
        keep = indices.flatten()
        return boxes_xyxy[keep], scores[keep]
    return np.zeros((0,4)), np.zeros(0)


def preprocess(frame, imgsz=640):
    img, ratio_pad = letterbox(frame, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, ratio_pad


def draw_results(frame, boxes, scores, fps):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Faces: {len(boxes)}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame


def run_realtime(model_path, input_source=0, imgsz=640):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {input_source}")

    print(f"Real-time face detection started. Press 'q' to quit.")
    print(f"Model: {model_path}")

    fps_deque = deque(maxlen=30)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_tensor, ratio_pad = preprocess(frame, imgsz)
        t0 = time.perf_counter()
        ort_output = session.run(None, {input_name: img_tensor})
        t1 = time.perf_counter()
        latency = (t1-t0)*1000
        fps_deque.append(1000.0/latency)
        boxes, scores = parse_onnx_output(ort_output, frame.shape[:2], ratio_pad)
        avg_fps = np.mean(fps_deque)
        frame = draw_results(frame, boxes, scores, avg_fps)
        cv2.imshow("Real-time Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time Face Detection")
    parser.add_argument("--input", type=str, default="0")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    input_source = int(args.input) if args.input.isdigit() else args.input
    run_realtime(args.model, input_source, args.imgsz)


if __name__ == "__main__":
    main()
