# vision
Open Source repo demoing vision enablement in any digital surface 

# Distance Detection Model

This model combines YOLO face detection with Depth Anything V2 to estimate distances to detected faces in images.

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

First initialize models once when the server spins up

```python
    yolo_model = init_yolo()
    depth_model = init_dav2()
```

Then, simple compute to get output each time you want to call the model. It defaults to onnx model.

```python
    distance, _ = compute_distance(frame, yolo_model, depth_model)
```


