import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
import os
import pandas as pd
import geocoder
import folium
from streamlit_folium import folium_static

# Set up page with wider layout
st.set_page_config(page_title="🌿 Plant Identifier", layout="wide")
st.title("🌿 Plant Identifier")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_model.h5")

model = load_model()

# Class label mapping with more details
class_labels = {
    "0": {"name": "Abies concolor", "type": "Conifer"},
    "1": {"name": "Acer rubrum", "type": "Deciduous"},
    "2": {"name": "Pinus strobus", "type": "Conifer"},
    "3": {"name": "Quercus alba", "type": "Deciduous"},
    "4": {"name": "Betula papyrifera", "type": "Deciduous"},
    "5": {"name": "Quercus rubra", "type": "Deciduous"},
    "6": {"name": "Ulmus americana", "type": "Deciduous"},
    "7": {"name": "Fraxinus americana", "type": "Deciduous"},
    "8": {"name": "Tilia americana", "type": "Deciduous"},
    "9": {"name": "Carya ovata", "type": "Deciduous"},
    "170": {"name": "Fagus grandifolia", "type": "Deciduous"}
}

log_file = "plant_observations.csv"
os.makedirs("uploads", exist_ok=True)

# Image input section
st.sidebar.header("Image Input")
option = st.sidebar.radio("Select input method:", 
                         ("Upload an image", "Take a photo with camera"),
                         index=0)

img_file = None
if option == "Take a photo with camera":
    img_file = st.sidebar.camera_input("Take a picture of the plant")
else:
    img_file = st.sidebar.file_uploader("Upload a plant image", 
                                       type=["jpg", "jpeg", "png"])

# Main content columns
col1, col2 = st.columns(2)

# Image display and processing
with col1:
    st.subheader("Plant Image")
    if img_file:
        # Save image
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_ext = "jpg" if option == "Take a photo with camera" else img_file.name.split('.')[-1]
        img_path = os.path.join("uploads", f"{timestamp}_observation.{file_ext}")
        
        with open(img_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Display image with enhancement options
        img = Image.open(img_path)
        st.image(img, caption="Your plant photo", use_column_width=True)
        
        # Simple enhancement options
        enhance = st.selectbox("Enhance image", 
                             ["Original", "Grayscale", "Contrast"],
                             index=0)
        
        if enhance == "Grayscale":
            img = img.convert("L")
        elif enhance == "Contrast":
            img = ImageEnhance.Contrast(img).enhance(1.5)
            
        st.image(img, caption=f"{enhance} view", use_column_width=True)

# Location and prediction
with col2:
    st.subheader("Location Data")
    
    # Auto geolocation from IP
    g = geocoder.ip('me')
    if g.ok and g.latlng:
        latitude, longitude = g.latlng
        st.success("📍 Location automatically detected")
        
        # Create interactive map
        m = folium.Map(location=[latitude, longitude], zoom_start=12)
        folium.Marker(
            [latitude, longitude],
            popup="Your location",
            icon=folium.Icon(color="green", icon="leaf")
        ).add_to(m)
        
        # Display map
        folium_static(m, width=400, height=300)
        
        # Manual override option
        if st.checkbox("Adjust location manually"):
            lat = st.number_input("Latitude", 
                                min_value=-90.0, 
                                max_value=90.0, 
                                value=float(latitude))
            lon = st.number_input("Longitude", 
                                min_value=-180.0, 
                                max_value=180.0, 
                                value=float(longitude))
            latitude, longitude = lat, lon
            
            # Update map with manual location
            m = folium.Map(location=[latitude, longitude], zoom_start=12)
            folium.Marker(
                [latitude, longitude],
                popup="Adjusted location",
                icon=folium.Icon(color="red", icon="flag")
            ).add_to(m)
            folium_static(m, width=400, height=300)
    else:
        st.warning("⚠️ Could not detect your location automatically")
        latitude, longitude = None, None
    
    # Only proceed with prediction if we have both image and location
    if img_file and latitude is not None and longitude is not None:
        # Prediction section
        st.subheader("Plant Identification")
        with st.spinner('Analyzing plant features...'):
            img = Image.open(img_path).resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            label = str(np.argmax(prediction))
            plant_data = class_labels.get(label, {"name": "Unknown Plant", "type": "Unknown"})
            
            # Display results in a nice card
            st.success(f"""
            **🌱 Identified Plant:** {plant_data['name']}  
            **🌳 Type:** {plant_data['type']}  
            **🔍 Confidence:** {np.max(prediction)*100:.1f}%
            """)
            
            # Log observation
            observation = {
                "image_path": img_path,
                "latitude": latitude,
                "longitude": longitude,
                "species": plant_data['name'],
                "plant_type": plant_data['type'],
                "timestamp": timestamp
            }
            
            if st.button("Save Observation"):
                df = pd.DataFrame([observation])
                if not os.path.exists(log_file):
                    df.to_csv(log_file, index=False)
                else:
                    df.to_csv(log_file, mode="a", header=False, index=False)
                st.success("Observation saved!")
                
                # Show recent observations
                if os.path.exists(log_file):
                    with st.expander("View recent observations"):
                        st.dataframe(pd.read_csv(log_file).tail(3))

# Footer with additional info
st.sidebar.markdown("---")
st.sidebar.info("""
This app helps identify plants from photos. 
All observations contribute to biodiversity research.
""")
