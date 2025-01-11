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

def download_weights():
    id = '17CjD-85mMkv1h7aK6hWsH3IxTvs5GXrn'
    download_path = 'DAV2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth'
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    if not os.path.exists(download_path):
        url = f'https://drive.google.com/uc?id={id}'
        gdown.download(url, download_path, quiet=False)

def init_dav2(
        device: torch.device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu'),
        download: bool = False, 
        use_onnx: bool = True,
) -> DepthAnythingV2: 

    if use_onnx:
        return onnxruntime.InferenceSession('onnx_models/dav2.onnx', providers=['CPUExecutionProvider'])
    
    else:
        if download:
            download_weights()

        max_depth = 20

        model_config = {'vits' : {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        depth_model = DepthAnythingV2(**{**model_config['vits'], 'max_depth': max_depth})
        depth_model.load_state_dict(torch.load('DAV2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth', 
                                                map_location=device, weights_only=True))
        depth_model = depth_model.to(device).eval()
        return depth_model

@torch.no_grad()
def compute_dav2_torch(
    image: ImageType,
    model: DepthAnythingV2 = None,
    download: bool=False,
    preprocess: bool=True,
) -> List[Tuple[int]]:

    if model is None:
        model = init_dav2(download=download)
    if preprocess:
        image = process_image(image)
    
    else:
        depth_map = model.infer_image(raw_image=image)

    return depth_map
    
def compute_dav2_onnx(
    image: ImageType,
    preprocess: bool=True,
    model: onnxruntime.InferenceSession = None,
) -> List[Tuple[int]]:
    
    if model is None:
        model = init_dav2(use_onnx=True)
    if preprocess:
        image = process_image(image)
    
    image = image.astype(np.float32)
        
    input_name = model.get_inputs()[0].name
    input_data = {input_name: image}
    
    depth_map = model.run(None, input_data)[0]
    return depth_map
    
def main():

    model = init_dav2(use_onnx=True)
    path = 'example_image.jpg'
    depth_map = compute_dav2_onnx(image=path, model=model)
    print(depth_map)

    # WORKS FOR ANY IMAGE 
    # ImageType = Union[
    #     torch.Tensor,     
    #     np.ndarray,       
    #     PIL.Image.Image,   
    #     str,              
    #     bytes              
    # ]

if __name__ == '__main__':
    main()