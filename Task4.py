import os
import zipfile # Import zipfile module for extraction
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import cv2

# Set dataset path
data_dir = "/content/train.zip"
IMG_SIZE = 64  # Resize all images to 64x64
NUM_CLASSES = 10

# Function to load and preprocess the dataset
def load_data(data_dir):
    images = []
    labels = []

    # Extract the zip file
    with zipfile.ZipFile(data_dir, 'r') as zip_ref:
        zip_ref.extractall('/content/train') # Extract to /content/train directory


    extracted_data_dir = '/content/train/train'  # Navigate to the gesture subfolder

    for gesture_folder in tqdm(os.listdir(extracted_data_dir)):
        if not gesture_folder.split('_')[0].isdigit():
            continue

        gesture_path = os.path.join(extracted_data_dir, gesture_folder)
        # Ensure label is within the valid range (0-9)
        label = int(gesture_folder.split('_')[0]) % NUM_CLASSES  # Extract label from folder name and apply modulo

        for img_file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            images.append(img)
            labels.append(label)

    images = np.array(images, dtype="float32") / 255.0  # Normalize pixel values
    labels = np.array(labels)
    return images, labels
# Load dataset
print("Loading data...")
X, y = load_data(data_dir)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Add channel dimension for grayscale images
print("Data loaded.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
data_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=3
)
print("Model training completed.")

# Evaluate the model
print("Evaluating the model...")
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=[f"Gesture {i}" for i in range(NUM_CLASSES)]))

# Save the model
model.save("hand_gesture_model.h5")
print("Model saved as hand_gesture_model.h5.")
