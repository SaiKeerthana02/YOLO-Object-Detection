Real-Time Object Detection with YOLO and OpenCV
A beginner-friendly computer vision project that detects common objects in a webcam feed or video file. It draws bounding boxes and confidence scores, counts objects by class, estimates processing speed, and can save an annotated video.

This project uses a pretrained YOLO model. It performs inference (using a trained model on new frames); it does not train a new model.

What you will learn
How OpenCV reads a webcam or video one frame at a time
How a YOLO detector turns an image into boxes, class IDs, and confidence scores
How to draw annotations with OpenCV
How confidence and image-size settings affect detection quality and speed
How to measure FPS and write processed frames to a video
1. Prerequisites
Install Python 3.11 or newer. This project was verified with Python 3.14, but Python 3.11 or 3.12 can still be a conservative choice on older systems where machine-learning packages have not caught up. Check your version:

python --version
You also need a webcam, or an existing video file for --source.

2. Download and open the project
Clone the repository, then enter its folder:

git clone https://github.com/Athilesten/yolo-opencv-object-detector.git
cd yolo-opencv-object-detector
If you downloaded the repository as a ZIP instead, extract it and open a terminal in the extracted folder.

3. Create a virtual environment
A virtual environment keeps this project's packages separate from other Python projects.

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell blocks activation, either run this once for your user account:

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
or use the environment's Python directly without activating it:

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate
4. Install the packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
opencv-python provides the cv2 module. ultralytics provides YOLO and installs its machine-learning dependencies. The first detection run may also download the pretrained model weights.

5. Verify the code
These tests exercise small pieces of the program without opening the webcam:

python -m unittest -v
python detector.py --help
6. Run the webcam detector
python detector.py
The default source is camera 0. Hold up common objects such as a bottle, cup, phone, backpack, or book. Press Q or Esc to close the window.

Open it in full-screen mode:

python detector.py --mirror --fullscreen
The normal window opens 1280 pixels wide. Choose another width if needed:

python detector.py --window-width 1600 --mirror
If the camera does not open:

python detector.py --source 1
Also check that Windows camera privacy settings allow desktop applications to use the camera, and close applications that may already be using it.

7. Useful experiments
Raise the confidence threshold to keep only stronger predictions:

python detector.py --confidence 0.50
Use a smaller model input for more speed, possibly at the cost of missing small objects:

python detector.py --image-size 480
Mirror the webcam view:

python detector.py --mirror
Force CPU inference:

python detector.py --device cpu
Run detection on a video file:

python detector.py --source path\to\sample.mp4
Save an automatically named annotated video in outputs/:

python detector.py --save
Choose the output name yourself:

python detector.py --save outputs\my_demo.mp4
Options can be combined:

python detector.py --source 0 --confidence 0.40 --image-size 480 --mirror --save
How the program works
cv2.VideoCapture opens the camera or video.
capture.read() returns one frame as a NumPy array.
model.predict(frame) asks YOLO to locate objects in that frame.
Each result contains box coordinates, a class ID, and a confidence score.
OpenCV draws the boxes and text on the original frame.
cv2.imshow displays the annotated frame.
The loop repeats until the source ends or you press Q/Esc.
The finally block releases the camera, video writer, and windows even if the program is interrupted.
OpenCV stores color images in BGR channel order rather than the more common RGB order. The coordinates (x1, y1) and (x2, y2) represent the top-left and bottom-right corners of each bounding box.

Settings worth discussing
Setting	Lower value	Higher value
Confidence threshold	More detections and false positives	Fewer, stronger detections
Image size	Usually faster, may miss small objects	Usually slower, may retain more detail
Model size	A nano model favors speed	Larger models may improve accuracy but need more compute
The displayed FPS is a smoothed estimate of the whole loop's throughput, including inference and drawing. It is not just the model's inference time.

Honest limitations
The pretrained model only knows the categories it was trained to recognize.
Poor lighting, motion blur, occlusion, and very small objects can hurt results.
A confidence score is not a guarantee that a prediction is correct.
CPU frame rate depends heavily on the computer and chosen image size.
Object counts are per frame; the current version does not track identity, so it cannot determine whether an object is the same one seen earlier.
Simple evaluation checklist
Record your observations for several short tests:

Test	What to observe
One clear object	Correct label and a stable box
Several objects	Whether each object is detected and counted
Dim lighting	Missed detections or lower confidence
Partial obstruction	Whether the detection disappears or changes
Confidence 0.25 vs 0.60	False positives versus missed objects
Image size 480 vs 640	FPS versus detection stability
This is a qualitative evaluation, not a formal accuracy benchmark, because we do not have a hand-labeled test dataset.

Resume bullets
Built a real-time object-detection application using Python, OpenCV, and a pretrained YOLO model to process webcam and video frames.
Implemented confidence filtering, custom bounding-box annotations, per-frame class counts, smoothed FPS measurement, and annotated video export.
Tested detection behavior under changes in lighting, confidence threshold, and model input size, documenting speed and reliability trade-offs.
Interview practice
Why did you use a pretrained model instead of training one?

The project's goal was to learn and build the complete inference pipeline: capturing frames, running detection, interpreting outputs, drawing results, measuring speed, and saving video. Training a reliable detector would also require collecting and labeling a suitable dataset. A logical next step would be fine-tuning on a small domain-specific dataset.

What does the confidence threshold do?

It removes predictions whose scores are below the selected threshold. Lowering it shows more possible objects but can introduce false positives. Raising it keeps stronger predictions but can miss real objects.

How did you measure real-time performance?

I timed consecutive iterations of the processing loop and converted the time per iteration into frames per second. I used exponential smoothing so the display would be readable instead of jumping sharply every frame.

What would you improve next?

I would add multi-object tracking so objects retain IDs between frames, collect a labeled test set for quantitative evaluation, and compare different model sizes or hardware backends for the best speed/accuracy balance.
