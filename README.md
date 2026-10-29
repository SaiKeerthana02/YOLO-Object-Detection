# Real-Time Object Detection with YOLO and OpenCV

A beginner-friendly computer vision project that detects common objects in a webcam feed or video file using Python, OpenCV, and a pretrained YOLO model.

The application draws bounding boxes and confidence scores, counts detected objects by class, estimates processing speed (FPS), and supports saving annotated videos.

**Note:** This project performs inference using a pretrained YOLO model. It does not train a new model.

## Features

- Real-time object detection using a webcam or video file
- Bounding boxes and confidence score display
- Per-frame object counting by class
- Frames-per-second (FPS) measurement
- Configurable confidence threshold and image size
- Full-screen and mirrored webcam modes
- Annotated video export
- CPU inference support

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- Computer Vision

## Prerequisites

- Python 3.11 or newer
- Webcam or sample video file
- Required Python packages from `requirements.txt`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SaiKeerthana02/YOLO-Object-Detection.git
cd YOLO-Object-Detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Project

### Start webcam detection

```bash
python detector.py
```

Press `Q` or `Esc` to close the detection window.

### Adjust the confidence threshold

```bash
python detector.py --confidence 0.50
```

### Change image size

```bash
python detector.py --image-size 480
```

### Run on CPU

```bash
python detector.py --device cpu
```

### Detect objects in a video

```bash
python detector.py --source path\to\sample.mp4
```

### Save the annotated video

```bash
python detector.py --save
```

*These commands assume the project files support the listed options.*

## How It Works

1. OpenCV captures frames from a webcam or video.
2. The pretrained YOLO model processes each frame.
3. The model returns detected object classes, bounding-box coordinates, and confidence scores.
4. OpenCV draws bounding boxes and labels on the frames.
5. The application displays the processed frames and estimates FPS.
6. Optionally, the annotated frames are saved as a video.

## Testing and Evaluation

The project can be tested under different conditions:

- One object versus multiple objects
- Bright versus dim lighting
- Fully visible versus partially obstructed objects
- Different confidence thresholds
- Different image sizes and processing speeds

These checks provide qualitative observations rather than a formal accuracy benchmark.

## Limitations

- The pretrained model recognizes only the categories it was trained on.
- Poor lighting, motion blur, and occlusion may reduce detection quality.
- Confidence scores do not guarantee correct predictions.
- Performance depends on hardware and model settings.
- Object counts are calculated per frame; object identity tracking is not implemented.

## Future Improvements

- Add multi-object tracking with persistent IDs.
- Evaluate performance on a labeled test dataset.
- Compare different YOLO model sizes.
- Optimize inference for improved speed and accuracy.

## Author

**Sai Keerthana Yarraguntla**

GitHub: [SaiKeerthana02](https://github.com/SaiKeerthana02)
