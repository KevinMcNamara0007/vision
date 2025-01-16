import cv2
import numpy as np
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import Any, Union
import PIL
from typing import List, Tuple
import os
import gdown
import onnxruntime

ImageType = Union[
    torch.Tensor,     
    np.ndarray,       
    PIL.Image.Image,   
    str,              
    bytes              
]

def process_image(image: ImageType) -> np.ndarray:
    if isinstance(image, str): 
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, bytes):
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    
    if len(image.shape) == 4 and image.shape[1] in [1,3,4]:
        image = np.transpose(image, (0, 2, 3, 1))
    elif len(image.shape) == 3 and image.shape[0] in [1,3,4]:
        image = np.transpose(image, (1, 2, 0))

    if len(image.shape) == 4:
        image = np.split(image, image.shape[0], axis=0)
        image = [img.squeeze(0) for img in image]
    
    return image

def compute_bbox(result):
    if len(result.boxes) == 0:
        return None
    boxes = result.boxes
    confs = boxes.conf.cpu().numpy()
    highest_conf_idx = np.argmax(confs)
    box = boxes.xyxy[highest_conf_idx].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    bbox = (x1, y1, x2, y2)
    return bbox

def get_eye_distance(cropped_depth):

    depth_gray = (cropped_depth * 255).astype(np.uint8)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    eyes = eye_cascade.detectMultiScale(
        depth_gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(10, 10),
        maxSize=(cropped_depth.shape[1]//2, cropped_depth.shape[0]//2)
    )
    
    eye_distances = []
    for (ex, ey, ew, eh) in eyes:
        eye_center_x = ex + ew//2
        eye_center_y = ey + eh//2
        eye_depth = cropped_depth[eye_center_y, eye_center_x]
        if not np.isnan(eye_depth):
            eye_distances.append(eye_depth)
    
    if eye_distances:
        return np.mean(eye_distances)
    
    h, w = cropped_depth.shape
    center_region = cropped_depth[h//3:2*h//3, w//3:2*w//3]
    return np.median(center_region[~np.isnan(center_region)])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init_models():
    yolo_model = YOLO('models/yolov11n-face.onnx', task='detect')
    depth_model = onnxruntime.InferenceSession('models/dav2.onnx', providers=['CPUExecutionProvider'])
    squinting_model = onnxruntime.InferenceSession('models/squint_detector.onnx', providers=['CPUExecutionProvider'])
    return yolo_model, depth_model, squinting_model

def compute_inference_engine(image, yolo_model, depth_model, squinting_model):

    # Process Image
    image = process_image(image)

    # Compute YOLO Model
    yolo_results = yolo_model(image, conf=0.25, device='cpu',verbose=False)
    bboxes = [compute_bbox(res) for res in yolo_results]
    if bboxes is None:
        return None, None
    bbox = bboxes[0]

    # Compute Depth Model
    image = image.astype(np.float32)
    input_name = depth_model.get_inputs()[0].name
    input_data = {input_name: image}
    depth_map = depth_model.run(None, input_data)[0]
    cropped_depth = depth_map[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    distance_measure = get_eye_distance(cropped_depth*42.72)

    # Compute Squinting Model
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    image = cv2.resize(image, (256, 256))
    image = image[np.newaxis, np.newaxis, ...]
    image = image.astype(np.float32) / 255.0
    mean, std = 0.395, 0.189
    image = (image - mean) / std
    input_name = squinting_model.get_inputs()[0].name
    input_data = {input_name: image}
    pred = sigmoid(squinting_model.run(None, input_data)[0][0][0])

    # Return Distance and Squinting Prediction
    return distance_measure, pred

async def get_inference_response(image):
    yolo_model, depth_model, squinting_model = init_models()
    frame = image
    distance, pred = compute_inference_engine(frame, yolo_model, depth_model, squinting_model)
    return float(distance), float(pred)

def main():
    yolo_model, depth_model, squinting_model = init_models()
    frame = cv2.imread('image.jpg')
    distance, pred = compute_inference_engine(frame, yolo_model, depth_model, squinting_model)
    print(distance, pred)

if __name__ == '__main__':
    main()
