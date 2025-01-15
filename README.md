# vision
Open Source repo demoing vision enablement in any digital surface 

# AI Inferfence Engine Usage

This model combines YOLO face detection with Depth Anything V2 (finetuned on metric depth) to estimate distances to detected faces in images. It also features a custom trained squinting detection model.

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
    yolo_model, depth_model, squinting_model = init_models()
```

Then, simple call compute_distance_engine each time you want to call the model. Note: The model defaults to onnx. 
```python
    distance, pred = compute_inference_engine(frame, yolo_model, depth_model, squinting_model)
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
