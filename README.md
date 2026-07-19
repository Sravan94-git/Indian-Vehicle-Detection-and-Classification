# 🚗 Indian Vehicle Detection & Classification System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-Flask-red.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://www.tensorflow.org/)
[![Computer Vision](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)

This project presents an **AI-powered Indian Vehicle Detection & Classification System** designed to detect and classify vehicles commonly found on Indian roads. Built using **TensorFlow**, **OpenCV**, and **Flask**, the system employs a two-stage deep learning pipeline that first detects vehicles and then classifies them into categories such as cars, buses, trucks, motorcycles, auto-rickshaws, and other vehicle types. The web-based application allows users to upload images or videos and receive real-time detection results with annotated outputs.

***

## ✨ Key Features

- 🚗 **Two-Stage Detection Pipeline:** Detects vehicles before classifying them for improved accuracy.
- 🎯 **Indian Vehicle Classification:** Recognizes vehicle categories commonly found on Indian roads.
- 📷 **Image & Video Support:** Processes both images and videos for vehicle detection.
- ⚡ **Real-Time Processing:** Optimized for fast inference suitable for live traffic analysis.
- 💻 **Flask Web Interface:** Interactive web application for easy uploads and visualization.
- 📦 **Multi-Vehicle Detection:** Detects multiple vehicles simultaneously in a single frame.
- 📍 **Bounding Box Annotation:** Displays detected vehicles with labels and confidence scores.
- 🚦 **Traffic Monitoring Ready:** Suitable for intelligent transportation and smart city applications.

***

## 🛠️ Tech Stack

- **Programming Language:** Python 3
- **Deep Learning Framework:** TensorFlow
- **Backend Framework:** Flask
- **Computer Vision:** OpenCV
- **Frontend:** HTML, CSS
- **Image Processing:** NumPy
- **Development Environment:** VS Code, Jupyter Notebook

***

## 🚀 Project Workflow

1. User uploads an image or video through the Flask web application.
2. The uploaded file is securely stored on the server.
3. The vehicle detection model identifies all vehicles present in the frame.
4. The classification model categorizes each detected vehicle.
5. Bounding boxes and class labels are drawn around detected vehicles.
6. The annotated image or processed video is generated and displayed to the user.

***

## 📂 Project Structure

```text
Indian-Vehicle-Detection-and-Classification/
│
├── static/
│   ├── uploads/
│   ├── outputs/
│   └── css/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── models/
│   ├── detection_model/
│   └── classification_model/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

***

## 📊 Applications

- 🚦 Smart Traffic Monitoring
- 🛣️ Intelligent Transportation Systems
- 🚓 Traffic Law Enforcement
- 📈 Traffic Density Analysis
- 🚘 Vehicle Counting Systems
- 🏙️ Smart City Surveillance
- 🎥 Highway Monitoring
- 📊 Urban Traffic Analytics

***

## 🎯 Future Enhancements

- 🎥 Live CCTV camera integration.
- 📹 Real-time video stream detection.
- 🚁 Drone-based traffic monitoring.
- 📍 GPS-enabled vehicle tracking.
- ☁️ Cloud deployment with analytics dashboard.
- 📱 Mobile application support.
- 🚨 Automatic traffic violation detection.
- 📊 Traffic statistics and reporting dashboard.

***

## 💻 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/Indian-Vehicle-Detection-and-Classification.git

cd Indian-Vehicle-Detection-and-Classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

***

## 📸 Output

- Upload an image or video containing vehicles.
- The system detects and classifies all visible vehicles.
- The application displays:
  - Original Image/Video
  - Annotated Output with Bounding Boxes
  - Vehicle Labels and Confidence Scores

***

## 📈 Results

- Accurate detection and classification of Indian vehicle types.
- Fast inference suitable for real-time traffic applications.
- Robust performance in dense traffic environments.
- User-friendly Flask-based web interface.
- Scalable solution for intelligent transportation and smart city initiatives.

***

## 🤝 Contributing

Contributions are welcome!

Feel free to fork the repository, create a feature branch, and submit a pull request.

***

## 📜 License

This project is licensed under the **MIT License**.

***

## 👨‍💻 Author

**Sravan**

AI | Machine Learning | Deep Learning | Computer Vision | Full Stack Developer

If you found this project useful, don't forget to ⭐ the repository!
