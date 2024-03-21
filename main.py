import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# from util import classify, set_background
from util import classify_image, set_background

set_background('./background/bgs_o.jpg')

# # set title
# st.title('Bone Break classification')
# # set header
# st.header('Please upload your X-ray image')

# Make all text white:
st.markdown("""<style>body { color: white; }</style>""", unsafe_allow_html=True)

white_heading_html = """
<h1 style="color: white; font-weight: bold;">Ouch! What Happened There?</h1>
"""

white_description_html = """
<p style="font-size: 1.2rem;">Don't worry, we're not radiologists, but upload your X-ray and we'll give it our best shot (pun intended) at guessing your fracture type.</p>
"""

st.markdown(white_heading_html, unsafe_allow_html=True)
st.markdown(white_description_html, unsafe_allow_html=True)

# upload file
uploaded_file=st.file_uploader('Upload an X-ray image', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/bone_break_classifier.h5')


with open('./model/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
    f.close()


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Resize image to (256, 256) before prediction
    image = image.resize((256, 256))

    st.image(image, caption='Uploaded Image', use_column_width=True)
    class_name, confidence_score = classify_image(image, model, class_names)
    st.write(f'Prediction: {class_name}')
    st.write(f'Confidence Score: {confidence_score * 100:.2f}%')

