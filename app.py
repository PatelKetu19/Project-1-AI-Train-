import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[-1] for line in f.readlines()]

# Streamlit App Title
st.title("♻️ Biodegradable vs Non-Biodegradable Classifier")

# Choose input method
option = st.radio("Choose Image Input Method:", ["📁 Upload Image", "📸 Use Camera"])

# Initialize image placeholder
image = None

# Upload image from file
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Capture image from camera
elif option == "📸 Use Camera":
    image_data = st.camera_input("Take a picture")
    if image_data is not None:
        image = Image.open(image_data).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

# Show Get Result button only if image is loaded
if image is not None and st.button("Get Result"):
    # Resize and normalize image
    input_shape = input_details[0]['shape']
    resized_image = image.resize((input_shape[2], input_shape[1]))  # width x height
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32) / 255.0

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output_data)

    # Extract top result
    top_index = np.argmax(prediction)
    confidence = float(prediction[top_index]) * 100
    label = labels[top_index]

    # Show result
    st.success(f"✅ **Prediction:** {label}")
    st.info(f"🔍 **Confidence:** {confidence:.2f}%")
