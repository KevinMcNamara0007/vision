from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import Union, List, Tuple, Dict
import os
import gdown

ImageType = Union[torch.Tensor, np.ndarray, str, bytes]

# Define emotion labels
EMOTIONS = ['sad', 'angry', 'happy', 'focused', 'surprised', 'bored']

def download_weights():
    url = 'https://drive.google.com/uc?id=1y53MeW4ZVoZ7vz-N4dDwWqTCpUIKtctY'
    path = 'weights/yolov11n-face.pt'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        gdown.download(url, path, quiet=False)

def process_image(image: ImageType) -> np.ndarray:
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, bytes):
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if len(image.shape) == 3 else cv2.COLOR_BGRA2RGBA)
    
    if image.ndim == 4:
        image = image.squeeze(0)
    
    return image

def init_yolo(download: bool = False) -> YOLO:
    if download:
        download_weights()
    return YOLO('weights/yolov11n-face.pt')

@torch.no_grad()
def compute_yolo(image: ImageType, conf: float = 0.25, model: YOLO = None, download: bool = False) -> List[Dict[str, Union[Tuple[int], str]]]:
    if model is None:
        model = init_yolo(download)
    
    processed_image = process_image(image)
    
    height, width = processed_image.shape[:2]
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_size = (640, int(640 / aspect_ratio))
    else:
        new_size = (int(640 * aspect_ratio), 640)
    resized_image = cv2.resize(processed_image, new_size, interpolation=cv2.INTER_LINEAR)
    
    padded_image = np.full((640, 640, 3), 128, dtype=np.uint8)
    x_offset = int((640 - resized_image.shape[1]) / 2)
    y_offset = int((640 - resized_image.shape[0]) / 2)
    padded_image[y_offset:y_offset+resized_image.shape[0], x_offset:x_offset+resized_image.shape[1]] = resized_image
    
    results = model(padded_image, conf=conf, verbose=False)
    
    detections = []
    for result in results:
        if result.boxes:
            box = result.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # Convert back to original image coordinates
            x1 = int((x1 - x_offset) * width / resized_image.shape[1])
            x2 = int((x2 - x_offset) * width / resized_image.shape[1])
            y1 = int((y1 - y_offset) * height / resized_image.shape[0])
            y2 = int((y2 - y_offset) * height / resized_image.shape[0])
            
            # Here we'd typically use another model for emotion detection
            # For this example, we'll simulate emotion detection based on facial features:
            face = processed_image[y1:y2, x1:x2]
            emotion = detect_emotion(face)  # Placeholder function

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'emotion': emotion
            })
        else:
            detections.append({'bbox': None, 'emotion': 'unknown'})
    
    return detections

def detect_emotion(face: np.ndarray) -> str:
    # This is a very simplified, placeholder function for emotion detection
    # In a real scenario, you'd use a pre-trained model for emotion recognition
    # Here, we'll just use some basic image analysis for simulation:
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.count_nonzero(edges)
    
    if edge_count < 500:  # Arbitrary threshold for simplicity
        return 'bored'  # Less facial movement
    elif np.mean(face) > 150:  # Brightness could indicate happiness
        return 'happy'
    elif np.mean(face) < 50:  # Darker could indicate sadness
        return 'sad'
    elif cv2.Laplacian(gray, cv2.CV_64F).var() > 100:  # High variance might indicate surprise or focus
        return 'surprised' if np.random.random() > 0.5 else 'focused'
    else:
        return 'angry'  # Default to angry if nothing else fits

def main():
    model = init_yolo(download=True)
    path = 'example_image.jpg'
    detections = compute_yolo(image=path, model=model)
    for detection in detections:
        print(f"Bounding Box: {detection['bbox']}, Emotion: {detection['emotion']}")

if __name__ == '__main__':
    main()