from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from yolov5 import YOLOv5

app = Flask(__name__)

# Load YOLOv5 model
model = YOLOv5("yolov5s.pt")  # You can use 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt' for better accuracy

@app.route('/detect', methods=['POST'])
def detect_monitors():
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)
    detections = results.pandas().xyxy[0]  # Results in pandas DataFrame

    monitor_count = len(detections[detections['name'] == 'tvmonitor'])
    return jsonify({"monitor_count": monitor_count})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
