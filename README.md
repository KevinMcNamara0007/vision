# vision
Open Source repo demoing vision enablement in any digital surface 

# Distance Detection Model

This model combines YOLO face detection with Depth Anything V2 (finetuned on metric depth) to estimate distances to detected faces in images.

## Installation

1. Create and activate a virtual environment (python3 -m venv env && source env/bin/activate)

2. Install dependencies (pip install -r requirements.txt)

## Usage

The model accepts various input formats:
- File paths (str)
- NumPy arrays
- PIL Images
- Torch Tensors
- Bytes

First initialize models (Call these only once on server startup)
```python
    yolo_model = init_yolo()
    depth_model = init_dav2()
```

Then, simple call compute_distance each time you want to call the model. Note: The model defaults to onnx. 
```python
    distance, _ = compute_distance(frame, yolo_model, depth_model)
```

## Installation
#### requires
- python >= 3.9
### PIP
- pip install -r requirements.txt
- Refresh if needed then run app
- #### OR
- pip install virtualenv
- virtualenv venv
- ./venv/Scripts/activate
- pip install -r requirements.txt

## Running APP
uvicorn src.asgi:vision --host=127.0.0.1 --port=8080 --reload


### WEB PAGES
- http://127.0.0.1:8080/vision/
- http://127.0.0.1:8080/docs

### Directories:
- config - contains env settings will be needed for model paths
- react - contains the web app files where you can modify react code
- static - contains web build pack which is initialized on backend
- src - contains backend python service and API(s)
