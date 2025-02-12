import io
from PIL import Image
import numpy as np
from src.utilities.inference_engine import get_inference_response
from src.utilities.brightness_model import get_brightness_adjustment


async def evaluate_image_service(image):
    # Read the file contents
    contents = await image.read()
    # Open the image with PIL
    pil_image = Image.open(io.BytesIO(contents))
    # Convert to NumPy array
    frame = np.array(pil_image)
    # Pass the NumPy array to the `get_distance` function
    # distance = await get_distance(frame)
    details = await get_inference_response(frame)
    pupil_dialation = await get_brightness_adjustment(frame)
    print(details, pupil_dialation)
    return {"distance": details[0], "squint": details[1], "iris": pupil_dialation}