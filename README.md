# 🚗 Indian Vehicle Detection & Classification System

This project is a real-time vehicle detection and classification system specifically tailored for the diverse traffic conditions found in India. By leveraging a two-stage deep learning pipeline, it accurately identifies and classifies a wide range of vehicles from both image and video inputs. This system is a valuable tool for applications in intelligent transportation and traffic management.

## 🌟 Key Features

* **Two-Stage Pipelined Architecture**: The system first detects vehicles and then classifies them, ensuring a strong balance of efficiency and accuracy.
* **Localized Vehicle Classification**: The model is trained to recognize vehicle types common to the Indian context, such as cars, buses, auto-rickshaws, and two-wheelers.
* **Real-Time Performance**: Optimized to process video streams in real-time for immediate traffic analysis and insights.
* **User-Friendly Interface**: A Flask-based web application allows users to easily upload images or videos to see the system in action.
* **Scalability**: Designed to handle multi-vehicle predictions in a single frame, making it suitable for dense traffic scenarios.

## 💻 Tech Stack

| Category                  | Tools & Libraries                                   |
| ------------------------- | --------------------------------------------------- |
| **Programming Language** | Python                                              |
| **Deep Learning Framework** | TensorFlow / PyTorch                              |
| **Computer Vision** | OpenCV, NumPy                                       |
| **Web Framework** | Flask                                               |

## 🚀 How to Run the Project

Follow these steps to get the project up and running on your local machine.

### 1. Clone the repository

```bash
git clone [https://github.com/your-username/indian-vehicle-detection.git](https://github.com/your-username/indian-vehicle-detection.git)
cd indian-vehicle-detection
```

### 2. Install dependencies
```bash

pip install -r requirements.txt
```
### 3. Run the Flask application
```bash

python app.py
```
