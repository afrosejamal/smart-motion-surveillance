# 🎥 Smart Motion & Object Detection System

> An AI-powered real-time motion detection and object tracking system built with **YOLOv8**, **OpenCV**, and **PyTorch** for intelligent surveillance and monitoring.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📌 Overview

Smart Motion & Object Detection is a real-time computer vision application that combines **motion detection**, **YOLOv8 object detection**, **face detection**, and **multi-object tracking** into a single intelligent surveillance system.

Instead of detecting every object continuously, the system first detects motion using background subtraction and then identifies moving objects using **YOLOv8**, making the application both efficient and accurate.

Whenever motion is detected, the application automatically captures screenshots of detected objects and stores them for future analysis.

---

# ✨ Features

* 🎥 Real-time webcam monitoring
* 🚶 Motion Detection using Background Subtraction (MOG2)
* 🤖 YOLOv8 Object Detection
* 😀 Face Detection using Haar Cascade
* 📍 Multi-Object Tracking with Centroid Tracker
* 📸 Automatic Screenshot Capture
* ✂️ Cropped Object Image Saving
* ⚡ Live FPS Monitoring
* 🖥 Clean Information Dashboard
* 🔍 Motion Region Analysis
* 🚗 Detects People, Vehicles & Other Objects

---

# 🛠 Tech Stack

### Programming Language

* Python

### Computer Vision

* OpenCV

### Deep Learning

* YOLOv8 (Ultralytics)
* PyTorch

### Libraries

* NumPy
* Requests

---

# 🏗 System Workflow

```text
Webcam
   │
   ▼
Capture Video Frame
   │
   ▼
Background Subtraction (MOG2)
   │
   ▼
Motion Detection
   │
   ▼
YOLOv8 Object Detection
   │
   ▼
Face Detection
   │
   ▼
Centroid Tracking
   │
   ▼
Screenshot Capture
   │
   ▼
Save Full Frame + Cropped Object
```

---

# 📂 Project Structure

```text
smart-motion-object-detection/

│── movement_analysis.py
│── requirements.txt
│── README.md
│── LICENSE
│── .gitignore
│
├── demo/
│     demo.mp4
│
├── screenshots/
│     home.png
│     detection.png
│     tracking.png
│     screenshot_capture.png
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/afrosejamal/smart-motion-object-detection.git

cd smart-motion-object-detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run

```bash
python movement_analysis.py
```

---

# 🎯 How It Works

1. Captures live video from webcam.
2. Applies Background Subtraction (MOG2).
3. Detects motion regions.
4. Runs YOLOv8 on moving objects.
5. Detects faces using Haar Cascade.
6. Tracks each object using Centroid Tracking.
7. Assigns a unique ID to every object.
8. Captures screenshots automatically.
9. Saves:

   * Full Frame
   * Cropped Object Image

---



# 📊 Key Features Implemented

✔ Motion Detection

✔ YOLOv8 Object Detection

✔ Face Detection

✔ Multi-Object Tracking

✔ Automatic Screenshot Capture

✔ Object Cropping

✔ FPS Monitoring

✔ Motion Analysis

✔ Live Information Panel

✔ Webcam Surveillance

---

# 🚀 Future Improvements

* Email Alert System
* Telegram Notifications
* Intrusion Detection
* Cloud Storage
* Multi-Camera Support
* Person Re-identification
* Web Dashboard
* Streamlit Interface
* Docker Deployment
* GPU Optimization

---

# 💻 Skills Demonstrated

* Computer Vision
* Deep Learning
* Object Detection
* Motion Detection
* Multi-Object Tracking
* YOLOv8
* OpenCV
* PyTorch
* Image Processing
* Python Development

---

# 👩‍💻 Author

**Afrose Fathima J**

Artificial Intelligence & Data Science Graduate

LinkedIn:
https://www.linkedin.com/in/afrose-fathima-jamal-492b57291

---

