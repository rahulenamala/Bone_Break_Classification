import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{b64_encoded});
            background-size: cover;
            color: white; 
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)



def classify_image(image, model, class_names):
    image = image.convert('RGB')
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class_idx = np.argmax(predictions)
    confidence_score = predictions[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence_score
