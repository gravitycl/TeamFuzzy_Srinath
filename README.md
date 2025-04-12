# 🌱 Plant.Dec

![App Interface](./assets/app-screenshot.png)

A smart plant identification system that combines computer vision and geolocation to recognize tree species from leaf images.

## Table of Contents
- [Features](#-features)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Model Details](#-model-details)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Image Processing
- 📸 Real-time camera capture
- 🖼️ Image upload (JPG, PNG)
- 🌐 URL image loading
- 🛠️ Image enhancement tools
  - Grayscale conversion
  - Contrast adjustment
  - Edge detection

### Geolocation
- 🌍 Automatic IP-based location detection
- 🗺️ Interactive Folium maps
- 📍 Manual location adjustment
- 🕒 Timezone-aware timestamps

### AI Capabilities
- 🔍 Multi-species identification
- 📊 Confidence scoring
- 🏆 Top-5 predictions display
- 🧠 Model interpretability (Grad-CAM)

## 🎥 Demo

[![Video Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://youtu.be/YOUR_VIDEO_ID)

## 📂 Dataset

Using the [LeafSnap Dataset](https://www.kaggle.com/datasets/xhlulu/leafsnap-dataset):

| Dataset Split | Images | Species |
|--------------|--------|---------|
| Lab          | 7,719  | 185     |
| Field        | 23,147 | 185     |
| **Total**    | 30,866 | 185     |

**Sample Species Classes:**
```python
['Acer rubrum', 'Quercus alba', 'Pinus strobus', 
 'Fagus grandifolia', 'Betula papyrifera', ...]
