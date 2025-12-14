import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Trash Classification",
    layout="centered"
)

st.title("Trash Classification App")
st.write("Upload an image of trash to classify it.")

class_names = ['battery', 
               'biological', 
               'brown-glass', 
               'cardboard', 'clothes', 
               'green-glass', 
               'metal', 
               'paper', 
               'plastic', 
               'shoes', 
               'trash', 
               'white-glass']

IMG_SIZE = 224

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "output" / "saved_model" / "new_final_model_no_more.keras"

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model(MODEL_PATH)


def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

#
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    input_tensor = preprocess_image(image)

    predictions = model.predict(input_tensor)
    predicted_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    predicted_class = class_names[predicted_idx]

    st.markdown("### Prediction Result")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")