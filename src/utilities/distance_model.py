from src.utilities.distance_detection import compute_dav2_onnx, compute_dav2_torch, init_dav2, process_image
from src.utilities.face_detection import compute_yolo, init_yolo
import cv2
import numpy as np


def get_eye_distance(cropped_depth):
    depth_gray = (cropped_depth * 255).astype(np.uint8)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    eyes = eye_cascade.detectMultiScale(
        depth_gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(10, 10),
        maxSize=(cropped_depth.shape[1] // 2, cropped_depth.shape[0] // 2)
    )

    eye_distances = []
    for (ex, ey, ew, eh) in eyes:
        eye_center_x = ex + ew // 2
        eye_center_y = ey + eh // 2
        eye_depth = cropped_depth[eye_center_y, eye_center_x]
        if not np.isnan(eye_depth):
            eye_distances.append(eye_depth)

    if eye_distances:
        return np.mean(eye_distances)

    h, w = cropped_depth.shape
    center_region = cropped_depth[h // 3:2 * h // 3, w // 3:2 * w // 3]
    return np.median(center_region[~np.isnan(center_region)])


def compute_distance(frame, yolo_model, depth_model, use_onnx: bool = True):
    '''
        compute_distance()

        inputs:
            frame: (any image type/file name for image)
            yolo_model: init_yolo(download=False)
            depth_model: init_dav2(download=False)
        output:
            distance: float scalar for distance in cm to camera

    '''

    frame = process_image(frame)

    bbox = compute_yolo(image=frame, model=yolo_model, preprocess=False)[0]
    if bbox is None:
        return None, None

    if use_onnx:
        depth_map = compute_dav2_onnx(image=frame, model=depth_model, preprocess=False)
    else:
        depth_map = compute_dav2_torch(image=frame, model=depth_model, preprocess=False)

    cropped_depth = depth_map[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    distance_measure = get_eye_distance(cropped_depth * 42.72)
    distance = f"{distance_measure:.2f}cm"

    return distance, bbox


def run_live_detection():
    yolo_model = init_yolo(download=False)
    depth_model = init_dav2(download=False)

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            distance, bbox = compute_distance(frame, yolo_model, depth_model)

            if distance is None:
                continue

            x1, y1, x2, y2 = bbox

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, distance, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            display_frame = frame if bbox is not None else frame
            cv2.imshow('Face Detection & Distance', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


async def get_distance(image):
    yolo_model = init_yolo(download=False)
    depth_model = init_dav2(download=False)
    frame = image
    distance, bbox = compute_distance(frame, yolo_model, depth_model)
    return distance
