import cv2
import numpy as np
import onnxruntime as ort

# Hardcoded average pupil size based on analysis
AVERAGE_PUPIL_SIZE = 1082.0

async def analyze_pupil_size(current_pupil_size):
    """
    Compares the current pupil size against the hardcoded average
    and determines brightness adjustment.

    :param current_pupil_size: Detected pupil size (area)
    :return: 1 (increase brightness), 0 (maintain), -1 (decrease brightness)
    """
    change = (current_pupil_size - AVERAGE_PUPIL_SIZE) / AVERAGE_PUPIL_SIZE

    if change > 0.15:
        return 1  # Increase brightness
    elif change < -0.15:
        return -1  # Decrease brightness
    return 0  # Maintain brightness

def get_best_provider():
    """Returns the best available ONNX provider."""
    available_providers = ort.get_available_providers()
    preferred_order = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
    sorted_providers = sorted(available_providers, key=lambda x: preferred_order.index(x) if x in preferred_order else len(preferred_order))
    return sorted_providers[0] if sorted_providers else None

def get_pupil_size(segmentation_mask):
    """
    Extracts pupil size from a segmentation mask.
    Assumes the pupil is the largest detected object.
    """
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.contourArea(largest_contour)
    return 0  # No pupil detected

# Get the best available ONNX provider
best_provider = get_best_provider()
if best_provider:
    ort_session = ort.InferenceSession("models/pupil_masking_model.onnx", providers=[best_provider])
    print(f"Using provider: {best_provider}")
else:
    print("No suitable provider found.")


async def predict_mask_onnx(frame):
    """Runs ONNX inference and returns predicted pupil mask."""
    input_size = (200, 200)
    input_image = cv2.resize(frame, input_size) / 255.0
    input_image = np.expand_dims(input_image, axis=-1).astype(np.float32)
    input_image = np.expand_dims(input_image, axis=0)
    onnx_input = {ort_session.get_inputs()[0].name: input_image}
    output = ort_session.run(None, onnx_input)[0]
    binary_mask = (output > 0.5).astype(np.uint8)
    return cv2.resize(np.squeeze(binary_mask), (frame.shape[1], frame.shape[0]))

async def get_brightness_adjustment(frame):
    """Determines brightness adjustment based on pupil size."""
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = await predict_mask_onnx(gray_frame)
        pupil_size = get_pupil_size(mask)
        return await analyze_pupil_size(pupil_size) if pupil_size > 0 else 0
    except Exception as e:
        print(e)
        return 0
