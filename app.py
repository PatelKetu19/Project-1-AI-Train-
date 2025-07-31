import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

@st.cache_data
def load_labels(path="labels.txt"):
    with open(path, "r") as f:
        return [line.strip().split(" ", 1)[1] for line in f]

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

st.title("Teachable Machine Image Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width =True)
    
    try:
        model = load_model()
        labels = load_labels()
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = float(predictions[0][predicted_index])

        st.success(f"Prediction: {labels[predicted_index]}")
        st.info(f"Confidence: {confidence * 100:.2f}%")
    except Exception as e:
        st.error(f"Model load/predict failed: {e}")

