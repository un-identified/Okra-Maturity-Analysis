import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import cv2
import numpy as np

# Load the trained model
model = load_model('image_maturity_model.h5')
# Function to load a single image
def load_single_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  
    img = img / 255.0  
    return img
# Ask user to select a test image
test_image_path = filedialog.askopenfilename(
    title="Select a test image",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
)
test_image = load_single_image(test_image_path)
test_image = np.expand_dims(test_image, axis=0) 

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('image_maturity_model.tflite', 'wb') as f:
    f.write(tflite_model)
# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Define the input tensor shape
input_shape = interpreter.get_input_details()[0]['shape']
input_tensor = np.zeros(input_shape, dtype=np.float32)

# Set the input tensor with the test image
# Convert the input data to FLOAT32 and set the input tensor
test_image_float32 = test_image.astype(np.float32)
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], test_image_float32)


# Run inference
interpreter.invoke()

# Get the output tensor
output_tensor = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# Print the raw predictions
print("Raw Predictions (TFLite):", output_tensor)

# Get the predicted class and confidence
predicted_class = np.argmax(output_tensor)
confidence = output_tensor[0, predicted_class]

class_name = ['young', 'mature', 'average'][predicted_class]
print(f"Predicted class (TFLite): {class_name}, Confidence: {confidence}")
