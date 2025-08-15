Object Detection with OpenCV DNN (SSD MobileNetV3 + COCO)
Detect objects in images and real-time webcam feed using OpenCV and a pretrained SSD MobileNetV3 model on the COCO dataset.

Features
Detect objects in static images and webcam video

Bounding boxes and labels for 80 COCO classes

Easy-to-run Python code

Requirements
Python 3.x

opencv-python

matplotlib

Install requirements:

bash
pip install opencv-python matplotlib
Files Needed
Place these files in your project folder:

frozen_inference_graph.pb (model weights)

ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt (model config)

labels.txt (COCO class labels)

sample.jpg (sample image for testing)

Usage
1. Static Image Detection

python
import cv2
import matplotlib.pyplot as plt

# Load model files
frozen_model = "frozen_inference_graph.pb"
config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# Initialize the model
model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Load class labels
with open("labels.txt", "rt") as f:
    classLabels = f.read().rstrip('\n').split('\n')

# Read sample image
img = cv2.imread("sample.jpg")

classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

font_scale = 1
font = cv2.FONT_HERSHEY_SIMPLEX

# Draw bounding boxes and labels
for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40),
                font, font_scale, (0, 255, 0), 3)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
2. Real-time Webcam Detection

python
import cv2

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    if len(classIndex) != 0:
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if classInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[classInd - 1],
                            (boxes + 10, boxes[1] + 40),
                            font, fontScale=font_scale,
                            color=(0, 255, 0), thickness=3)
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Run
For an image: Run the image detection section in your Python IDE.

For webcam: Run the webcam section. Press q to quit.

Notes
If webcam doesn’t open, try changing cv2.VideoCapture(0) to cv2.VideoCapture(1).

Edit "sample.jpg" for your own test images.

This model supports up to 80 COCO classes, as listed in labels.txt.# object-detection
