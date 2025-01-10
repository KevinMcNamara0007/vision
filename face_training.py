import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Check TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"Number of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# Load the pre-trained DNN model for face detection
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
config_path = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

def detect_and_crop_faces(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # Crop the face from the image
            face = image[startY:endY, startX:endX]
            
            # Resize the cropped face to 300x300 while maintaining aspect ratio
            h, w = face.shape[:2]
            if h > w:
                face = cv2.resize(face, (int(300 * w / h), 300))
            else:
                face = cv2.resize(face, (300, int(300 * h / w)))
            
            # Create a blank canvas of 300x300
            canvas = np.zeros((300, 300, 3), dtype=np.uint8)
            
            # Calculate the padding for centering the face
            y_offset = (300 - face.shape[0]) // 2
            x_offset = (300 - face.shape[1]) // 2
            
            # Paste the resized face onto the center of the canvas
            canvas[y_offset:y_offset+face.shape[0], x_offset:x_offset+face.shape[1]] = face
            
            return canvas
    return None  # If no face detected

def custom_preprocessing_function(img):
    img = img.astype(np.uint8)  # Ensure image is in uint8 format for OpenCV
    cropped_face = detect_and_crop_faces(img)
    if cropped_face is not None:
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)  # Convert back to RGB
        return cropped_face / 255.0  # Normalize to [0, 1]
    else:
        return np.zeros((300, 300, 3))  # Return black image if no face detected

def load_data(filepath):
    data_generator = ImageDataGenerator(
        preprocessing_function=custom_preprocessing_function,
        validation_split=0.2
    )

    train_data = data_generator.flow_from_directory(
        filepath,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_data = data_generator.flow_from_directory(
        filepath,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    return train_data, validation_data

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_data, validation_data, epochs=20):
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    history = model.fit(
        train_data,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        validation_data=validation_data,
        validation_steps=validation_data.samples // validation_data.batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

def evaluate_model(model, validation_data):
    scores = model.evaluate(validation_data, steps=validation_data.samples // validation_data.batch_size)
    print(f"Test Loss: {scores[0]}")
    print(f"Test Accuracy: {scores[1]}")

if __name__ == "__main__":
    filepath = 'path_to_your_image_folder'

    # Load and preprocess data
    train_data, validation_data = load_data(filepath)

    # Define input shape and number of classes
    input_shape = (300, 300, 3)  # Changed to match the new cropped size
    num_classes = len(train_data.class_indices)

    # Create and compile the model
    model = create_model(input_shape, num_classes)
    model.summary()

    # Train the model
    history = train_model(model, train_data, validation_data)

    # Evaluate the model
    evaluate_model(model, validation_data)

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()