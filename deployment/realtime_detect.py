"""Real-time face detection — webcam / video file / image / folder (PyTorch)

Usage:
    python deployment/realtime_detect.py --input 0             # webcam
    python deployment/realtime_detect.py --input video.mp4     # video file
    python deployment/realtime_detect.py --input photo.jpg     # single image
    python deployment/realtime_detect.py --input my_photos/    # image folder (batch)
    python deployment/realtime_detect.py --input photo.jpg --save output/   # save results
    python deployment/realtime_detect.py --input video.mp4 --save output.mp4 # save video
"""
import os
import sys
import argparse
import time
import cv2
import numpy as np
from collections import deque
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "training", "checkpoints", "best_model_v2.pt")

# ── image extensions ────────────────────────────────────────────
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}


def draw_results(frame, results, fps=None):
    """Draw boxes + optional FPS overlay on frame. Returns (annotated_frame, face_count)."""
    face_count = 0
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        face_count = len(boxes)
    else:
        face_count = 0

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Faces: {face_count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame, face_count


def detect_image(model, img_path, imgsz, conf, save_dir=None):
    """Analyze a single image and optionally save result."""
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        return None, 0

    t0 = time.perf_counter()
    results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
    t1 = time.perf_counter()

    latency = (t1 - t0) * 1000
    frame, faces = draw_results(frame, results, fps=None)

    print(f"  {os.path.basename(img_path)}  →  {faces} face(s)  |  {latency:.1f} ms")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_name = os.path.splitext(os.path.basename(img_path))[0] + "_detected.jpg"
        out_path = os.path.join(save_dir, out_name)
        cv2.imwrite(out_path, frame)
        print(f"    saved → {out_path}")

    return frame, faces


def detect_folder(model, folder_path, imgsz, conf, save_dir=None):
    """Batch analyze all images in a folder."""
    folder_path = os.path.abspath(folder_path)
    img_paths = []
    for ext in IMAGE_EXTS:
        img_paths.extend(Path(folder_path).rglob(f"*{ext}"))
        img_paths.extend(Path(folder_path).rglob(f"*{ext.upper()}"))

    if not img_paths:
        print(f"[ERROR] No images found in: {folder_path}")
        return

    img_paths = sorted(set(str(p) for p in img_paths))
    print(f"\n{'='*60}")
    print(f"Batch image analysis — {len(img_paths)} images found")
    print(f"{'='*60}")

    total_faces = 0
    total_time = 0.0

    for i, img_path in enumerate(img_paths):
        frame, faces = detect_image(model, img_path, imgsz, conf, save_dir=save_dir)
        if frame is not None:
            total_faces += faces
            total_time += 1

        # show progress every 10 images
        if (i + 1) % 10 == 0:
            print(f"  --- progress: {i+1}/{len(img_paths)} ---")

    if total_time > 0:
        avg_face_count = total_faces / total_time
    else:
        avg_face_count = 0.0

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(img_paths)} images, {total_faces} total faces")
    print(f"Avg faces/image: {avg_face_count:.1f}")
    if save_dir:
        print(f"Results saved to: {save_dir}")
    print(f"{'='*60}\n")


def detect_stream(model, input_source, imgsz, conf, save_path=None):
    """Real-time detection on webcam or video file with display."""
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {input_source}")

    is_webcam = isinstance(input_source, int)
    source_name = "Webcam" if is_webcam else os.path.basename(str(input_source))

    # video writer (if saving)
    writer = None
    if save_path and not is_webcam:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps_src, (w, h))
        print(f"Recording output → {save_path}")
    elif save_path and is_webcam:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, 30.0, (w, h))
        print(f"Recording output → {save_path}")

    print(f"\nSource: {source_name}")
    print(f"Model: {model.model_name if hasattr(model, 'model_name') else 'YOLOv8n-face'}")
    print(f"Press 'q' to quit, or wait for video to finish.\n")

    fps_deque = deque(maxlen=30)
    frame_count = 0
    total_faces = 0

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

        frame, faces = draw_results(frame, results, fps=avg_fps)
        frame_count += 1
        total_faces += faces

        if writer:
            writer.write(frame)

        cv2.imshow("Real-time Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and is_webcam:
            # screenshot
            ss_path = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(ss_path, frame)
            print(f"Screenshot saved → {ss_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # summary
    if frame_count > 0:
        avg_face_count = total_faces / frame_count
        print(f"\n{'='*60}")
        print(f"Stream complete: {frame_count} frames processed")
        print(f"Total faces detected: {total_faces}")
        print(f"Avg faces/frame: {avg_face_count:.1f}")
        if save_path:
            print(f"Output saved to: {save_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Face Detection — supports webcam / video / image / folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deployment/realtime_detect.py --input 0               # webcam (default)
  python deployment/realtime_detect.py --input video.mp4        # video file
  python deployment/realtime_detect.py --input photo.jpg        # single image
  python deployment/realtime_detect.py --input my_photos/       # batch folder
  python deployment/realtime_detect.py --input photo.jpg --save results/
  python deployment/realtime_detect.py --input video.mp4 --save output.mp4
        """
    )
    parser.add_argument("--input", type=str, default="0",
                        help="Camera index (0,1,…), video path, image path, or folder path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to PyTorch model (.pt)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default 0.25)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save path: directory for images, .mp4 file for video/webcam")
    args = parser.parse_args()

    # ── load model ───────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print(f"  Expected at: {args.model}")
        print(f"  Make sure you've cloned the repo with model weights, or specify --model.")
        sys.exit(1)

    print(f"Loading model: {args.model} ...")
    model = YOLO(args.model)
    print("Model loaded.\n")

    # ── determine input type ──────────────────────────────────
    if args.input.isdigit():
        # webcam
        input_source = int(args.input)
        detect_stream(model, input_source, args.imgsz, args.conf, save_path=args.save)

    elif os.path.isdir(args.input):
        # folder → batch image analysis
        detect_folder(model, args.input, args.imgsz, args.conf, save_dir=args.save)

    elif os.path.isfile(args.input):
        ext = os.path.splitext(args.input)[1].lower()
        if ext in IMAGE_EXTS:
            # single image
            save_dir = args.save
            frame = None
            if save_dir and os.path.splitext(save_dir)[1]:  # save is a file path
                os.makedirs(os.path.dirname(save_dir) or ".", exist_ok=True)
                frame, faces = detect_image(model, args.input, args.imgsz, args.conf, save_dir=None)
                if frame is not None:
                    cv2.imwrite(save_dir, frame)
                    print(f"Saved → {save_dir}")
            else:
                frame, faces = detect_image(model, args.input, args.imgsz, args.conf, save_dir=save_dir)
            if frame is not None:
                cv2.imshow("Detection Result", frame)
                print("\nPress any key to close the image window.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif ext in VIDEO_EXTS:
            # video file
            detect_stream(model, args.input, args.imgsz, args.conf, save_path=args.save)
        else:
            print(f"[ERROR] Unknown file type: {ext}")
            print(f"  Supported image: {', '.join(IMAGE_EXTS)}")
            print(f"  Supported video: {', '.join(VIDEO_EXTS)}")
            sys.exit(1)

    else:
        # assume it's a video file (could be URL or pipe)
        print(f"[WARN] Input not found as file/directory, treating as video source: {args.input}")
        detect_stream(model, args.input, args.imgsz, args.conf, save_path=args.save)


if __name__ == "__main__":
    main()
