import tensorflow as tf #a
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import cv2
import numpy as np
# Define the directory structure
dataset_directory = r"C:\Users\naman\OneDrive\Desktop\Thermal_image\Thermal_image"
def load_single_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  
    img = img / 255.0 
    return img

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

target_size = (64, 64)

train_generator = train_datagen.flow_from_directory(
    dataset_directory,
    target_size=target_size,
    batch_size=3, 
    class_mode='categorical',
    shuffle=True
)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu')) 
model.add(layers.Dense(128, activation='relu'))  
model.add(layers.Dense(3, activation='softmax'))

optimizer = Adam(learning_rate=0.001)  
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
)
model.save('image_maturity_model.h5')
model = load_model('image_maturity_model.h5')
def load_single_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  
    img = img / 255.0  
    return img
test_image_path = filedialog.askopenfilename(
    title="Select a test image",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
)
test_image = load_single_image(test_image_path)
test_image = np.expand_dims(test_image, axis=0) 
raw_predictions = model.predict(test_image)
print("Raw Predictions:", raw_predictions)
prediction = model.predict(test_image)
class_name = ['young', 'mature', 'average'][np.argmax(prediction)]
confidence = prediction[0, np.argmax(prediction)]
print(f"Predicted class: {class_name}, Confidence: {confidence}")