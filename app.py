import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 256

MODEL_PATH = "low_light_lol_model.h5"
LOW_DATASET_PATH = r"C:\Users\vinod\Desktop\LowLightImageEnhancement\our485\low"
OUTPUT_PATH = r"C:\Users\vinod\Desktop\LowLightImageEnhancement\output"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ===============================
# LOAD MODEL (Inference only)
# ===============================
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH, compile=False)

model = load_cnn_model()

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Low Light Image Enhancement", layout="wide")

st.title("🌙 Low-Light Image Enhancement")
st.write("Select a dataset image OR upload your own low-light image to enhance it using CNN.")

# ===============================
# IMAGE INPUT OPTIONS
# ===============================
option = st.radio(
    "Choose image source:",
    ("Select from Dataset", "Upload Image")
)

image_np = None
image_name = None

# -------- Option 1: Dataset image --------
if option == "Select from Dataset":
    image_files = sorted(os.listdir(LOW_DATASET_PATH))

    selected_image = st.selectbox(
        "Select an image from dataset",
        image_files
    )

    if selected_image:
        img_path = os.path.join(LOW_DATASET_PATH, selected_image)
        img = cv2.imread(img_path)
        image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_name = selected_image

# -------- Option 2: Upload image --------
else:
    uploaded_file = st.file_uploader(
        "Upload a low-light image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_name = uploaded_file.name

# ===============================
# PROCESS & DISPLAY
# ===============================
if image_np is not None:
    st.subheader("Original Low-Light Image")
    st.image(image_np, use_container_width=True)

    if st.button("✨ Enhance Image"):
        # Preprocess
        img_resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
        img_norm = img_resized / 255.0

        # Predict
        pred = model.predict(np.expand_dims(img_norm, axis=0))[0]

        # Postprocess
        enhanced = (pred * 255).astype(np.uint8)

        # Save output
        save_path = os.path.join(OUTPUT_PATH, "enhanced_" + image_name)
        cv2.imwrite(save_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

        st.subheader("Enhanced Image")
        st.image(enhanced, use_container_width=True)

        st.success(f"Enhanced image saved to: {save_path}")
