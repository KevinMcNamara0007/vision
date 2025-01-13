from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import Any, Union
import PIL
from typing import List, Tuple
import os
import gdown

ImageType = Union[
    torch.Tensor,
    np.ndarray,
    PIL.Image.Image,
    str,
    bytes
]

def download_weights():
    id = '1y53MeW4ZVoZ7vz-N4dDwWqTCpUIKtctY'
    download_path = 'weights/yolov11n-face.pt'
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    if not os.path.exists(download_path):
        url = f'https://drive.google.com/uc?id={id}'
        gdown.download(url, download_path, quiet=False)

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

def init_yolo(download: bool=False, use_onnx: bool=True) -> YOLO:

    if use_onnx:
        return YOLO('models/yolov11n-face.onnx', task='detect')
    else:
        if download:
            download_weights()
        return YOLO('weights/yolov11n-face.pt')

@torch.no_grad()
def compute_yolo(
        image: ImageType,
        conf: float = 0.25,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu'),
        model: YOLO = None,
        download: bool=False,
        preprocess: bool=True,
        use_onnx: bool=True,
) -> List[Tuple[int]]:

    if model is None:
        model = init_yolo(download=download, use_onnx=use_onnx)
    if preprocess:
        image = process_image(image)

    if use_onnx:
        results = model(image, conf=conf, device='cpu',verbose=False)
    else:
        results = model(image, conf=conf, device=device, verbose=False)
    bboxes = [compute_bbox(res) for res in results]

    return bboxes