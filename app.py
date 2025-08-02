import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read labels from file
with open("labels.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[-1] for line in f.readlines()]

# Title
st.title("♻️ Biodegradable or Non-Biodegradable Classifier")

# Upload image file
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Once a file is uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Show "Get Result" button
    if st.button("Get Result"):
        # Resize to model input size
        input_shape = input_details[0]['shape']
        image_resized = image.resize((input_shape[2], input_shape[1]))  # (width, height)

        # Normalize image if needed (assuming float32 model)
        input_data = np.expand_dims(image_resized, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.squeeze(output_data)

        # Get highest score and label
        top_index = np.argmax(prediction)
        confidence = float(prediction[top_index]) * 100
        label = labels[top_index]

        # Show result
        st.markdown(f"### ✅ Prediction: **{label}**")
        st.markdown(f"### 🔍 Confidence: `{confidence:.2f}%`")
