import streamlit as st #m
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load the pre-trained model
model = load_model('maturity_model.h5')

# Function to preprocess a single image
def load_single_image(uploaded_file):
    # Convert the BytesIO object to an OpenCV image
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    # Resize and normalize the image
    img = cv2.resize(image, (64, 64))  # Assuming the target size is (64, 64)
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    return img

# Streamlit app
def main():
    st.title("Image Maturity Prediction")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the selected image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        image = load_single_image(uploaded_file)
        image = np.expand_dims(image, axis=0)  # Add batch dimension for prediction

        # Make predictions
        raw_predictions = model.predict(image)
        st.write("Raw Predictions:", raw_predictions)

        # Get the final prediction
        predicted_class = ['young', 'mature', 'average'][np.argmax(raw_predictions)]
        confidence = raw_predictions[0, np.argmax(raw_predictions)]
        
        # Display the result
        st.write(f"Predicted class: {predicted_class}, Confidence: {confidence}")

if __name__ == "__main__":
    main()
