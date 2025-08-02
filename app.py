import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[-1] for line in f.readlines()]

# Title
st.title("♻️ Biodegradable vs Non-Biodegradable Classifier")

# Image selection method
option = st.radio("Choose Image Input Method:", ["📁 Upload Image", "📸 Use Camera"])

image = None

# 📁 Upload image from file
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except UnidentifiedImageError:
            st.error("❌ Error: Invalid image format. Please upload a JPG or PNG.")

# 📸 Take photo using camera
elif option == "📸 Use Camera":
    image_data = st.camera_input("Take a picture")
    if image_data:
        try:
            image = Image.open(image_data).convert("RGB")
            st.image(image, caption="Captured Image", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Camera error: {e}")

# ✅ Prediction button
if image is not None:
    if st.button("Get Result"):
        # Preprocess
        input_shape = input_details[0]["shape"]
        resized_image = image.resize((input_shape[2], input_shape[1]))
        input_data = np.expand_dims(resized_image, axis=0).astype(np.float32) / 255.0

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.squeeze(output)

        # Results
        top_index = np.argmax(prediction)
        confidence = float(prediction[top_index]) * 100
        label = labels[top_index]

        st.success(f"✅ **Prediction:** {label}")
        st.info(f"🔍 **Confidence:** {confidence:.2f}%")
