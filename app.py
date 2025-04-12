import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
import os
import pandas as pd
import geocoder

# Set up page
st.set_page_config(page_title="🌿 Plant Identifier", layout="centered")
st.title("🌿 Plant Identifier")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_model.h5")

model = load_model()

# Class label mapping — update with all classes from your model
class_labels = {
    "0": "Abies concolor",
    "1": "Acer rubrum",
    "2": "Pinus strobus",
    "3": "Quercus alba",
    "4": "Betula papyrifera",
    "5": "Quercus rubra",
    "6": "Ulmus americana",
    "7": "Fraxinus americana",
    "8": "Tilia americana",
    "9": "Carya ovata",
    "170": "Fagus grandifolia"
}

log_file = "plant_trees.csv"
os.makedirs("uploads", exist_ok=True)

# Upload image
uploaded_file = st.file_uploader("Upload a leaf or plant image", type=["jpg", "jpeg", "png"])

# Auto geolocation from IP
g = geocoder.ip('me')
latitude, longitude = "", ""
if g.ok and g.latlng:
    latitude, longitude = map(str, g.latlng)
else:
    st.warning("⚠️ Could not detect your location automatically.")

# Prediction and logging
if uploaded_file and latitude and longitude:
    # Save image
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_path = os.path.join("uploads", f"{timestamp}_{uploaded_file.name}")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show uploaded image
    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    # Preprocess & predict
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label = str(np.argmax(prediction))
    plant_name = class_labels.get(label, "Unknown Plant")

    st.success(f"🌱 Prediction: {plant_name} (class {label})")

    # Log prediction
    row = pd.DataFrame([[img_path, latitude, longitude, plant_name, timestamp]],
                       columns=["image_path", "latitude", "longitude", "prediction", "timestamp"])
    if not os.path.exists(log_file):
        row.to_csv(log_file, index=False)
    else:
        row.to_csv(log_file, mode="a", header=False, index=False)

    # Embed map
    if os.path.exists("reforestation_map.html"):
        with open("reforestation_map.html", "r", encoding="utf-8") as f:
            html_content = f.read()
            components.html(html_content, height=600, scrolling=True)

else:
    st.info("Please upload an image and wait for location detection.")
