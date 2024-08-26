# Okra Maturity Analysis

## Project Overview

This project aims to analyze the maturity of okra plants using thermal imaging and machine learning. It leverages a pre-trained TensorFlow Lite model to classify the maturity stage of okra plants based on their thermal images.

## Features

- **Thermal Image Classification:** Utilizes a TensorFlow Lite model to categorize okra plant maturity into stages such as "young", "developed", and "average".
- **Model Training:**  The project includes a Python script (`okra_model_trainer.py`) for training the machine learning model.
- **Model Deployment:**  A TensorFlow Lite model (`image_maturity_model.tflite`) is provided for deployment and inference.
- **Streamlit Web App:** A Streamlit web application (`streamlit_site.py`) allows for interactive analysis and visualization of thermal images.

## Installation


1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the model (Optional):**

   ```bash
   python okra_model_trainer.py
   ```

2. **Run the Streamlit web app:**

   ```bash
   streamlit run streamlit_site.py
   ```

3. **Run the inference script with the pre-trained model:**

   ```bash
   python Run_with_model.py 
   ```

This will load the pre-trained model (`image_maturity_model.tflite`) and apply it to the thermal image data located in the `Thermal_image` directory. 

**Note:** The `Thermal_image` directory contains image data labeled with maturity stages ("young", "developed", and "average"). These images serve as the input for the model.

