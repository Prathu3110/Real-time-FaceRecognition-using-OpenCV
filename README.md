# Real-time-FaceRecognition-using-OpenCV
Real-Time Face Recognition Using OpenCV and face_recognition

# Real-Time Face Recognition Using OpenCV and face_recognition

This project implements a real-time face recognition system using OpenCV and the `face_recognition` library in Python. It detects and recognizes faces from a webcam feed by comparing them to pre-encoded reference images stored in a `faces/` directory.

## 🔧 Requirements

- Python 3.6+
- OpenCV (`cv2`)
- numpy
- face_recognition
- face_recognition_models

Install the dependencies using pip:

pip install opencv-python face_recognition numpy

Note: For GPU acceleration with dlib (used by face_recognition), CUDA support must be properly set up on your machine. Replace model='cuda' with 'hog' if not using GPU.


project-folder/
│
├── faces/                 # Folder containing reference face images (e.g., person1.jpg, person2.jpg)
├── face_recognition_app.py  # Main script
├── README.md
