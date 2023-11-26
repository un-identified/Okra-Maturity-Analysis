import tensorflow as tf #y
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import cv2
import numpy as np
model = load_model('maturity_model.h5')
def load_single_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  # Assuming the target size is (64, 64)
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    return img
# Make predictions on a single test image
test_image_path = filedialog.askopenfilename(
    title="Select a test image",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
)
test_image = load_single_image(test_image_path)
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension for prediction
raw_predictions = model.predict(test_image)
print("Raw Predictions:", raw_predictions)
prediction = model.predict(test_image)

# Print prediction
class_name = ['young', 'mature', 'average'][np.argmax(prediction)]
confidence = prediction[0, np.argmax(prediction)]
print(f"Predicted class: {class_name}, Confidence: {confidence}")