import io

from PIL import Image
import numpy as np

from src.utilities.distance_model import get_distance


async def evaluate_image_service(image):
    # Read the file contents
    contents = await image.read()
    # Open the image with PIL
    pil_image = Image.open(io.BytesIO(contents))
    # Convert to NumPy array
    frame = np.array(pil_image)
    # Pass the NumPy array to the `get_distance` function
    distance = await get_distance(frame)

    return {"distance": distance, "squint": 1, "iris": 1}
