from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import Any, Union
import PIL
from typing import List, Tuple
import os
import gdown
from DAV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

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

def download_weights():
    id = '17CjD-85mMkv1h7aK6hWsH3IxTvs5GXrn'
    download_path = 'models/depth_anything_v2_metric_hypersim_vits.pth'
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    if not os.path.exists(download_path):
        url = f'https://drive.google.com/uc?id={id}'
        gdown.download(url, download_path, quiet=False)

def init_dav2(
        device: torch.device = torch.device('cpu'),
        download: bool = False, 
) -> DepthAnythingV2: 

    if download:
        download_weights()

    max_depth = 20

    model_config = {'vits' : {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
    depth_model = DepthAnythingV2(**{**model_config['vits'], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load('models/depth_anything_v2_metric_hypersim_vits.pth',
                                            map_location=device, weights_only=True))
    depth_model = depth_model.to(device).eval()
    return depth_model

@torch.no_grad()
def compute_dav2(
    image: ImageType,
    model: DepthAnythingV2 = None,
    download: bool=False,
    preprocess: bool=True,
) -> List[Tuple[int]]:
    
    '''
        compute_dav2()

            input:
                image: (REQUIRED) torch.Tensor (any shape), np.ndarray (any shape), 
                        PIL.Image.Image, str (path), bytes (base64)  
                model: (OPTIONAL) Creates DAV2 model on the fly. For repeated calls, use init_dav2() THEN pass
                        that model in as a paramter to avoid repeated initialization 
                        model = init_dav2()
                        compute_dav2(model = model)
                download: (OPTIONAL) bool - True or False if we need the weights
                preprocess: (OPTIONAL) bool - True if we want to preprocess image, false if its already processed
            output:
                bboxes: 'numpy.ndarray': pixel map of distances from camera (unscaled) 
                        
    '''

    if model is None:
        model = init_dav2(download=download)
    if preprocess:
        image = process_image(image)
    depth_map = model.infer_image(image)

    return depth_map