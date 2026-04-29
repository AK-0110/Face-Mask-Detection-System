import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.title("🎥 Face Mask Detection (Web App)")

model = load_model("mask_detector.h5", compile=False)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    pred = model.predict(img)
    conf = pred[0][0]

    if conf > 0.6:
        st.success(f"Mask Detected  ({conf:.2f})")
    elif conf < 0.4:
        st.error(f"No Mask Detected  ({1-conf:.2f})")
    else:
        st.warning("Uncertain ")

run = st.checkbox("Use Webcam")

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    for _ in range(1000):   
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not working")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            cv2.putText(frame, "No Face Detected", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            face = cv2.resize(face, (128,128))
            face = face / 255.0
            face = face.reshape(1,128,128,3)

            pred = model.predict(face)
            conf = pred[0][0]

            if conf > 0.6:
                label = f"Mask ({conf:.2f})"
                color = (0,255,0)
            elif conf < 0.4:
                label = f"No Mask ({1-conf:.2f})"
                color = (0,0,255)
            else:
                label = "Uncertain"
                color = (255,255,0)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        stframe.image(frame, channels="BGR")

    cap.release()