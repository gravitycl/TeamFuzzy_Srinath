from flask import Flask, request, jsonify, send_file
from tensorflow import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array

from flask_cors import CORS
import numpy as np
import os
import datetime
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load your trained model
model = load_model("plant_model.h5")
log_file = "planted_trees.csv"

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # or use mobilenet_v2.preprocess_input if you used that during training
    return x

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    f = request.files["file"]
    latitude = request.form.get("latitude", "")
    longitude = request.form.get("longitude", "")

    # Ensure filename is safe and unique
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{f.filename}"
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(img_path)

    # Run prediction
    try:
        pred = model.predict(preprocess(img_path))
        prediction = int(np.argmax(pred))  # Convert to int for cleaner JSON
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Log to CSV
    try:
        log_entry = pd.DataFrame([[img_path, latitude, longitude, prediction, timestamp]],
                                 columns=["image_path", "latitude", "longitude", "prediction", "timestamp"])
        log_entry.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)
    except Exception as e:
        return jsonify({"error": f"Logging failed: {str(e)}"}), 500

    return jsonify({"prediction": prediction})

@app.route("/map", methods=["GET"])
def show_map():
    if not os.path.exists("reforestation_map.html"):
        return jsonify({"error": "Map file not found"}), 404
    return send_file("reforestation_map.html")

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
