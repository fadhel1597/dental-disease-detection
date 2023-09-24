import requests
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# URL of the API endpoint
api_url = "http://localhost:5000/object-detection"  # Update with your server address

# Path to the image you want to test
image_path = "tests/cavity.jpg"  # Update with your image path

# Send a POST request to the API
with open(image_path, "rb") as image_file:
    files = {"img": (image_path, image_file, "image/jpeg")}
    response = requests.post(api_url, files=files)

# Check if the request was successful
if response.status_code == 200:
    result_data = response.json()
    
    # Decode the annotated image from base64
    annotated_image_base64 = result_data["annotated_image_base64"]
    annotated_image_bytes = base64.b64decode(annotated_image_base64)
    annotated_image = Image.open(BytesIO(annotated_image_bytes))
    
    # Convert the PIL Image to a NumPy array
    annotated_image_np = np.array(annotated_image)

    # Display the annotated image using OpenCV
    cv2.imshow("Annotated Image", annotated_image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print the detection results
    results_data = result_data["results_data"]
    for result in results_data:
        print(f"Class: {result['class']}, Confidence: {result['confidence']}")
else:
    print(f"Error: {response.status_code}")