"""Real-time object detection with YOLO and OpenCV.

Examples:
    python detector.py
    python detector.py --source sample.mp4 --save
    python detector.py --confidence 0.50 --mirror
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import time
from typing import Any


def parse_source(value: str) -> int | str:
    """Turn a numeric camera source such as '0' into an integer.

    OpenCV expects camera indexes as integers and file paths as strings.
    """

    stripped = value.strip()
    return int(stripped) if stripped.isdigit() else stripped


def class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    """Look up a readable class name from a YOLO class ID."""

    if isinstance(names, dict):
        return names.get(class_id, f"class_{class_id}")
    if 0 <= class_id < len(names):
        return names[class_id]
    return f"class_{class_id}"


def format_counts(counts: Counter[str]) -> str:
    """Format class counts in a stable, readable order."""

    return " | ".join(f"{name}: {counts[name]}" for name in sorted(counts))


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Return a repeatable BGR color for each class."""

    palette = (
        (255, 178, 50),
        (50, 205, 50),
        (60, 140, 255),
        (220, 80, 180),
        (255, 220, 70),
        (180, 100, 255),
    )
    return palette[class_id % len(palette)]


def draw_label(
    cv2: Any,
    frame: Any,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw readable text with a solid background."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    label_top = max(0, y - text_height - baseline - 8)
    label_bottom = max(text_height + baseline + 8, y)
    cv2.rectangle(
        frame,
        (x, label_top),
        (x + text_width + 8, label_bottom),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 4, label_bottom - baseline - 4),
        font,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def draw_predictions(cv2: Any, frame: Any, result: Any) -> Counter[str]:
    """Draw YOLO boxes manually and return counts by class."""

    counts: Counter[str] = Counter()
    boxes = result.boxes
    if boxes is None:
        return counts

    for box in boxes:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].cpu().tolist())
        confidence = float(box.conf[0].cpu())
        class_id = int(box.cls[0].cpu())
        name = class_name(result.names, class_id)
        counts[name] += 1

        color = color_for_class(class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(cv2, frame, f"{name} {confidence:.2f}", x1, y1, color)

    return counts


def draw_dashboard(cv2: Any, frame: Any, fps: float, counts: Counter[str]) -> None:
    """Draw performance and object-count information on the frame."""

    total = sum(counts.values())
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  Objects: {total}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if counts:
        cv2.putText(
            frame,
            format_counts(counts),
            (12, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def make_output_path(requested_path: str | None) -> Path:
    """Create an output path, using a timestamp when no name was provided."""

    if requested_path:
        path = Path(requested_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("outputs") / f"detection_{timestamp}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect common objects in webcam or video frames using YOLO."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (usually 0) or a video-file path. Default: 0",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Ultralytics model weights. Default: yolo26n.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Keep detections at or above this score (0 to 1). Default: 0.35",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="YOLO input size. Smaller is faster but may miss small objects. Default: 640",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 'cpu' or '0' for the first GPU.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Flip frames horizontally, like a mirror.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Display the detector in a full-screen window.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="Display-window width when not full-screen. Default: 1280",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        default=None,
        metavar="OUTPUT.mp4",
        help="Save annotated video. A path is optional.",
    )
    parser.add_argument(
        "--hide-counts",
        action="store_true",
        help="Hide the FPS and object-count dashboard.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.image_size <= 0:
        parser.error("--image-size must be greater than 0")
    if args.window_width <= 0:
        parser.error("--window-width must be greater than 0")


def run(args: argparse.Namespace) -> int:
    # Imports live here so `python detector.py --help` works before installation.
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as error:
        print(
            "Missing dependency. Activate your virtual environment, then run:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Details: {error}"
        )
        return 1

    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Could not open source {source!r}.")
        if isinstance(source, int):
            print("Try another camera index, such as: --source 1")
        else:
            print("Check that the video path exists and is readable.")
        return 1

    print(f"Loading {args.model}. The first run may download model weights...")
    try:
        model = YOLO(args.model)
    except Exception as error:
        capture.release()
        print(f"Could not load model {args.model!r}: {error}")
        return 1

    writer = None
    output_path = None
    smoothed_fps = 0.0
    previous_time = time.perf_counter()
    window_name = "YOLO Object Detector - press Q or Esc to quit"
    window_created = False

    try:
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                # End of a video file is normal; a camera read failure is not fatal.
                print("No more frames were available.")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            prediction_options: dict[str, Any] = {
                "conf": args.confidence,
                "imgsz": args.image_size,
                "verbose": False,
            }
            if args.device is not None:
                prediction_options["device"] = args.device

            result = model.predict(frame, **prediction_options)[0]
            counts = draw_predictions(cv2, frame, result)

            current_time = time.perf_counter()
            instantaneous_fps = 1.0 / max(current_time - previous_time, 1e-9)
            previous_time = current_time
            smoothed_fps = (
                instantaneous_fps
                if smoothed_fps == 0.0
                else 0.90 * smoothed_fps + 0.10 * instantaneous_fps
            )

            if not args.hide_counts:
                draw_dashboard(cv2, frame, smoothed_fps, counts)

            if args.save is not None and writer is None:
                output_path = make_output_path(args.save or None)
                height, width = frame.shape[:2]
                source_fps = capture.get(cv2.CAP_PROP_FPS)
                if source_fps <= 1.0 or source_fps > 240.0:
                    source_fps = 20.0
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {output_path}")

            if writer is not None:
                writer.write(frame)

            if not window_created:
                # WINDOW_NORMAL lets OpenCV enlarge the display beyond the
                # camera's native resolution. This affects only the window,
                # not detection accuracy or the saved video.
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if args.fullscreen:
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                else:
                    frame_height, frame_width = frame.shape[:2]
                    display_height = round(
                        args.window_width * frame_height / frame_width
                    )
                    cv2.resizeWindow(
                        window_name, args.window_width, display_height
                    )
                window_created = True

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if output_path is not None:
        print(f"Saved annotated video to: {output_path.resolve()}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
