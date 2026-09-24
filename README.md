# 🚗 Vehix - Indian Vehicle Detection & Classification Using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb.svg)](https://vitejs.dev/)
[![Model](https://img.shields.io/badge/Model-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)

Vehix is an AI-powered **FastAPI and React application** for detecting and classifying vehicles in Indian road imagery. It uses **YOLOv8n** to locate vehicles and a trained classifier to identify each detected vehicle type. Users can upload images or videos and review annotated results, confidence scores, and vehicle crops.

***

## ✨ Key Features

* **Vehicle detection:** Locates vehicles in road images and video frames using YOLOv8n.
* **Vehicle classification:** Identifies detected vehicles across 12 trained classes.
* **Image analysis:** Returns an annotated image, vehicle crops, labels, and confidence scores.
* **Video analysis:** Processes uploaded MP4, MOV, or AVI files frame by frame.
* **Readiness screen:** Waits for the Render backend and its ML models to become available.
* **Interactive API documentation:** FastAPI provides automatic documentation at `/docs`.

***

## 🛠️ Technologies Used

* **Programming Language:** Python 3.12
* **Deep Learning:** TensorFlow, Keras, and Ultralytics YOLOv8
* **Backend Framework:** FastAPI
* **Frontend:** React and Vite
* **Computer Vision:** OpenCV
* **Image Processing:** Pillow and NumPy
* **API Server:** Uvicorn

***

## 🚀 Project Workflow

1. The browser accepts an image or video and prepares the upload.
2. The React frontend checks the backend readiness endpoint.
3. FastAPI validates the uploaded file and prepares the media.
4. YOLOv8n locates vehicles in the image or each video frame.
5. Detected vehicle crops are passed to the trained classifier.
6. The API returns annotated media and detection or classification results.
7. The frontend displays the report with confidence values and vehicle details.

### Architecture

```text
React frontend: upload media and display reports
					  |
					  v
FastAPI endpoints: /api/analyze/image and /api/analyze/video
					  |
					  v
YOLOv8n detection + trained vehicle classification
					  |
					  v
Annotated media, crops, labels, and confidence scores
```

### Vehicle classes

The classifier recognizes the following vehicle categories:

| Class | Class | Class |
| --- | --- | --- |
| Motorized two-wheeler | Ambassador taxi | Autorickshaw |
| Bicycle | Bus | Car |
| Minitruck | Motorvan | Rickshaw |
| Toto | Truck | Van |

***

## 📂 Project Structure

```text
Vehix/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── weights/
│       ├── classifier.h5
│       └── yolov8n.pt
│
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

***

## 📊 Applications

* 🚦 Traffic monitoring and vehicle counting
* 🛣️ Indian road-scene analysis
* 🏙️ Smart-city transportation systems
* 📹 Traffic video processing
* 🔍 Vehicle dataset exploration
* 📈 Road-safety and mobility research

***

## 💻 Installation

### Clone the Repository

```bash
git clone https://github.com/Sravan94-git/Indian-Vehicle-Detection-and-Classification.git
cd Indian-Vehicle-Detection-and-Classification
```

### Install Backend Dependencies

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://localhost:8000` and its interactive documentation is available at `http://localhost:8000/docs`.

### Start the Frontend

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open the frontend URL shown by Vite, usually `http://localhost:5173`.

***

## 📸 Output

* Upload an image containing one or more vehicles.
* Review the annotated image and detected vehicle crops.
* Upload a supported video and review the processed result.
* Inspect vehicle labels and detection and classification confidence scores.

***

## 🎯 Future Enhancements

* Real-time camera and CCTV stream detection.
* Vehicle counting and traffic-density analytics.
* GPS-based traffic mapping.
* Mobile application integration.
* Historical analysis and reporting dashboards.

***

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository, create a feature branch, and submit a pull request.

***

## 📜 License

This project is licensed under the MIT License.

***

## 👨‍💻 Author

**Sravan Kumar Sunkara**
