# 🎥 Face Mask Detection System

##  Objective

The objective of this project is to detect whether individuals are wearing face masks in real-time using a webcam. The system helps automate mask compliance in public spaces or workplace environments.

---

##  Tech Stack

* Python
* OpenCV
* TensorFlow / Keras
* Convolutional Neural Network (CNN)
* Haar Cascade Classifier
* Streamlit

---

## How It Works

1. The system captures video using a webcam or accepts image uploads.
2. Haar Cascade is used to detect faces in the frame.
3. Each detected face is preprocessed and passed to a trained CNN model.
4. The model predicts whether the person is wearing a mask or not.
5. The result is displayed with bounding boxes:

   * 🟢 Green → Mask
   * 🔴 Red → No Mask
   * 🟡 Yellow → Uncertain

---

##  Key Features

* Real-time face mask detection using webcam
* Image upload support via Streamlit web app
* CNN trained on labeled dataset (With Mask / Without Mask)
* Live video feed with bounding boxes and labels
* Confidence-based prediction system

---

##  Project Structure

```
mask_detection_project/
│
├── app.py
├── detect_mask.py
├── mask_detector.h5
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

---

##  Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/AK-0110/Face-Mask-Detection-System.git
cd Face-Mask-Detection-System
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the Streamlit web app

```
streamlit run app.py
```

---

##  Requirements

```
opencv-python
tensorflow
numpy
streamlit
pillow
```

---

##  Model Details

* Model Type: Convolutional Neural Network (CNN)
* Input Size: 128 × 128 images
* Accuracy: ~90–92% validation accuracy
* Training Platform: Kaggle

---

##  Use Cases

* Public places (airports, malls, stations)
* Corporate offices
* Entry checkpoints

This system helps automate mask compliance and reduce manual monitoring.

---

##  Future Improvements

* Improve accuracy using deeper models
* Replace Haar Cascade with advanced face detectors
* Deploy fully online using web-based camera streaming
* Add alert system for “No Mask” detection

---

##  Author

Arushi Khethavath

---

## 📌 Note

Ensure that the trained model (`mask_detector.h5`) and Haar Cascade file are placed in the same directory as `app.py` before running the application.

