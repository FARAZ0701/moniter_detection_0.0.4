from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load the pre-trained YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Handle output layers
unconnected_out_layers = net.getUnconnectedOutLayers()

# Convert the output layers to 1-based index
if isinstance(unconnected_out_layers, np.ndarray):
    unconnected_out_layers = unconnected_out_layers.flatten()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Load the classes (object names)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Monitor label in the COCO dataset
MONITOR_LABEL = 'tvmonitor'  # Update if needed

@app.route('/detect', methods=['POST'])
def detect_monitors():
    # Get image data from the request
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detection results
    monitor_count = 0
    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == MONITOR_LABEL:
                    monitor_count += 1

    return jsonify({"monitor_count": monitor_count})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
