import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(
    page_title="EcoVision - Waste Classifier", page_icon="♻️", layout="centered"
)


# Load model (cached to avoid reload on every run)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("artifacts/model.h5")
    return model


model = load_model()
CLASS_NAMES = ["Organic", "Recyclable"]
IMG_SIZE = (224, 224)

# --------------------------
# HEADER
# --------------------------
st.title("🌱 EcoVision: AI Waste Classifier")
st.markdown("""
Upload an image of waste material, and **EcoVision** will classify it as **Organic** or **Recyclable** using a deep learning model (ResNet50).
""")

# --------------------------
# FILE UPLOAD
# --------------------------
uploaded_file = st.file_uploader(
    "📸 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_resized = image_data.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # --------------------------
    # DISPLAY RESULT
    # --------------------------
    st.markdown("---")
    st.subheader(f"🧠 Prediction: **{pred_class}**")
    st.progress(int(confidence))
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Optional — show probabilities
    st.write("### Class Probabilities")
    for label, prob in zip(CLASS_NAMES, preds[0]):
        st.write(f"- {label}: {prob * 100:.2f}%")

    # --------------------------
    # EXTRA DETAILS
    # --------------------------
    st.markdown("---")
    st.caption("""
    *Model:* ResNet50 (Transfer Learning, ImageNet Pretrained)  
    *Image Size:* 224×224  
    *Developed by:* [@bhuvn24](https://github.com/bhuvn24)
    """)

else:
    st.info("👆 Upload an image to get started.")
