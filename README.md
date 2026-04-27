# **Face Mask Detection System**

## &#x20;Objective

The objective of this project is to detect whether individuals are wearing face masks in real-time using a webcam. This system helps automate mask compliance in public or workplace environments.

---

##  Tech Stack

* Python
* OpenCV
* TensorFlow / Keras
* Convolutional Neural Network (CNN)
* Haar Cascade Classifier

---

##  How It Works

1. A webcam captures live video feed.
2. OpenCV uses a Haar Cascade classifier to detect faces in each frame.
3. Each detected face is preprocessed and passed to a trained CNN model.
4. The model predicts whether the person is wearing a mask or not.
5. The system displays:

   * 🟢 Green box → Mask
   * 🔴 Red box → No Mask

---

##  Key Features

* Real-time face mask detection using webcam
* CNN trained on labeled dataset (With Mask / Without Mask)
* Live video feed with bounding boxes and labels
* Simple and efficient implementation

---

## Project Structure

```
mask-detection/
│
├── detect_mask.py
├── mask_detector.h5
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/mask-detection.git
cd mask-detection
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the project

```
python detect_mask.py
```

---

## Requirements

```
opencv-python
tensorflow
numpy
```

---

##  Model Details

* Model Type: Convolutional Neural Network (CNN)
* Input Size: 128 × 128 images
* Accuracy: ~90–92% validation accuracy
* Training Platform: Kaggle

---

##  Use Case

This system can be used in:

* Public places (airports, malls, stations)
* Corporate offices
* Entry checkpoints

to ensure mask compliance automatically.

---

## Future Improvements

* Add alert sound for “No Mask” detection
* Improve accuracy with deeper models
* Replace Haar Cascade with DNN-based face detection
* Deploy as a web application

---

##  Author

Your Name

---

##  Note

The trained model (`mask_detector.h5`) is required to run this project. Make sure it is placed in the same directory as the script.
