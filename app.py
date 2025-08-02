import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[-1] for line in f.readlines()]

# App title
st.title("♻️ Biodegradable vs Non-Biodegradable Classifier")

# Select input method
option = st.radio("Choose Image Input Method:", ["📁 Upload Image", "📸 Use Camera"])

image = None

# Upload image from device
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Convert uploaded image to PIL
            bytes_data = uploaded_file.read()
            image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error("Error loading image. Please try a different file.")

# Capture image from camera
elif option == "📸 Use Camera":
    image_data = st.camera_input("Take a photo")
    if image_data is not None:
        try:
            image = Image.open(image_data).convert("RGB")
            st.image(image, caption="Captured Image", use_column_width=True)
        except Exception as e:
            st.error("Error loading captured image. Try retaking the photo.")

# Process and predict
if image is not None and st.button("Get Result"):
    input_shape = input_details[0]['shape']
    resized = image.resize((input_shape[2], input_shape[1]))  # width x height
    input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index]) * 100.0
    label = labels[predicted_index]

    st.success(f"✅ **Prediction:** {label}")
    st.info(f"🔍 **Confidence:** {confidence:.2f}%")
