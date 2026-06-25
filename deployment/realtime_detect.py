"""Real-time face detection — webcam / video file inference (PyTorch)"""
import os
import argparse
import time
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "training", "checkpoints", "best_model_v2.pt")


def draw_results(frame, results, fps):
    """Draw boxes + FPS overlay on frame"""
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        faces = len(boxes)
    else:
        faces = 0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Faces: {faces}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame


def run_realtime(model_path, input_source=0, imgsz=640, conf=0.25):
    """Main loop: capture → inference → draw → display"""
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {input_source}")

    print(f"Model: {model_path}")
    print(f"Input: {'webcam' if isinstance(input_source, int) else input_source}")
    print(f"Press 'q' to quit.\n")

    fps_deque = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
        t1 = time.perf_counter()

        latency = (t1 - t0) * 1000
        fps_deque.append(1000.0 / latency)
        avg_fps = np.mean(fps_deque)

        frame = draw_results(frame, results, avg_fps)
        cv2.imshow("Real-time Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time Face Detection")
    parser.add_argument("--input", type=str, default="0",
                        help="Camera index (0) or video file path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to PyTorch model (.pt)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    args = parser.parse_args()

    input_source = int(args.input) if args.input.isdigit() else args.input
    run_realtime(args.model, input_source, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
