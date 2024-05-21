import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
from playsound import playsound
import time
from PIL import Image

st.title("Sign Language Translator")
st.subheader("የምልክት ቋንቋ ወደ አማርኛ ይተርጉሙ!")


if 'detecting' not in st.session_state:
    st.session_state.detecting = False

def detect_gesture():
    voice = pyttsx3.init()
    voices = voice.getProperty('voices')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open webcam. Please check if the camera is connected properly.")
        return

    detector = HandDetector(maxHands=1)
    classifier = Classifier("model/keras_model.h5", "model/labels.txt")
    offset = 20
    imgSize = 300
    labels = ["I love you", "hello", "sorry", "please", "yes", "Thank you"]
    audios = ["I love you (mp3cut.net).m4a", "hello (mp3cut.net).m4a", "Yikrta (mp3cut.net).m4a",
              "Ebakh(please) (mp3cut.net).m4a", "AWO (mp3cut.net).m4a", "AMESEGNALEW (mp3cut.net).m4a"]

    voice.setProperty('voice', voices[10].id)
    voice.setProperty('rate', 150)

    frame_window = st.image([])  # Streamlit placeholder for video frames

    while st.session_state.detecting:
        success, img = cap.read()
        if not success:
            st.error("Failed to read from webcam. Please check if the camera is working properly.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.size == 0:
                continue

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            highest_value_index = prediction.index(max(prediction))
            highest_value = max(prediction)

            if highest_value * 100 > 97:
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[highest_value_index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.7,
                            (0, 0, 0), 2)
                playsound(audios[highest_value_index])

            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 0), 4)

        # Convert the image to RGB format for Streamlit
        imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        frame_window.image(imgOutput)  # Update Streamlit image placeholder

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    frame_window.empty()  # Clear the Streamlit image placeholder when detection stops

start_button = st.empty()  # Placeholder for start button
stop_button = st.empty()   # Placeholder for stop button

if st.session_state.detecting:
    stop_button.button('መተርጎም አቁም', on_click=lambda: st.session_state.update({'detecting': False}))
else:
    start_button.button('መተርጎም ጀምር', on_click=lambda: st.session_state.update({'detecting': True, 'start_detection': True}))

if st.session_state.detecting:
    detect_gesture()

