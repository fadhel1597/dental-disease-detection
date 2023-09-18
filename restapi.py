from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

from keras.models import load_model

from PIL import Image
import cv2
import numpy as np
import io
import os
import datetime
import base64

import torch
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename


app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


@app.route("/image-classification",  methods=["POST"])
def image_classification():

    image = request.files["img"]
    filename = secure_filename(image.filename)

    # Save the image to a temporary location
    temp_path = os.path.join("/tmp", filename)
    image.save(temp_path)

    image = cv2.imread(temp_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Resize the image with the specified interpolation method
    image = cv2.resize(image, (150, 150))
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)

    labels = ['Unlabeled', 'crossbite', 'normal', 'openbite', 'overbite', 'underbite']

    keras_model = load_model("weights/VGG16.h5")
    red_prob = keras_model.predict(image)
    y_pred = (red_prob > 0.5).astype(int)

    detected_labels = [labels[i] for i in range(len(labels)) if y_pred[0][i] == 1]

    if detected_labels == []:
        detected_labels = 'No Class Detected'

    # print(detected_labels)

    # Create a dictionary to store the detected labels
    response = {"detected_labels": detected_labels}

    os.remove(temp_path)

    return jsonify(response)
        
        

@app.route("/object-detection", methods=["POST"])
def object_detection():
        
    image = request.files["img"]
    filename = secure_filename(image.filename)

    # Save the image to a temporary location
    temp_path = os.path.join("/tmp", filename)
    image.save(temp_path)

    image = cv2.imread(temp_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.resize(image, (640,640))

    model = YOLO("weights/YOLOv8.pt")
    results = model.predict(image, imgsz=640, conf=0.5, iou=0.5)

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    colors = {name: color for name, color in zip(model.names.values(), colors)}

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotator = Annotator(frame, font_size=0.5)
    boxes = results[0].boxes

    results_data = []

    for box in boxes:
        bbox = box.xyxy[0]
        c = box.cls
        class_name = model.names[int(c)]
        color = colors[class_name] 
        confidence = box.conf[0].item()
        confidence = "{:.2f}".format(confidence)

        box_data = {
            'class': class_name,
            'confidence': confidence
        }

        results_data.append(box_data)

    frame = annotator.result()
    result_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    response_data = {
        'results_data': results_data,
        'annotated_image_base64': base64.b64encode(result_img).decode('utf-8')  # Encode the image as base64
    }

    os.remove(temp_path)

    return response_data


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)  # debug=True causes Restarting with stat