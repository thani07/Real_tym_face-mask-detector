import streamlit as st
import cv2
import numpy as np
import threading
import time
from tensorflow.keras.models import load_model
from playsound import playsound
from PIL import Image

# === Load model and cascade ===
model = load_model("model/mask_detection_cnn_final.h5")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# === Label map ===
label_map = {
    0: "with_mask",
    1: "without_mask"
}

# === Audio paths ===
AUDIO_FILES = {
    "with_mask": "audio/with_mask.wav",
    "without_mask": "audio/without_mask.wav"
}

def play_audio_async(path):
    threading.Thread(target=playsound, args=(path,), daemon=True).start()

# === Streamlit UI ===
st.set_page_config(page_title="Face Mask Detection", layout="centered")
st.title("😷 Real-Time Face Mask Detection with Audio")
FRAME_WINDOW = st.image([])

start_button = st.button("Start Camera")

# === Run logic only when Start is pressed ===
if start_button:
    cap = cv2.VideoCapture(0)
    st.info("📷 Camera started. Press the STOP button in browser to exit.")

    last_label = None
    last_audio_time = 0
    COOLDOWN = 3.0  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("❌ Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            face = cv2.resize(roi, (224, 224))
            inp  = np.expand_dims(face, axis=0) / 255.0

            prob = model.predict(inp, verbose=0)[0][0]
            cls  = 1 if prob > 0.5 else 0
            label = label_map[cls]

            now = time.time()
            if label != last_label and (now - last_audio_time) >= COOLDOWN:
                play_audio_async(AUDIO_FILES[label])
                last_label = label
                last_audio_time = now

            # Draw bounding box and label
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            text  = "With Mask" if cls == 0 else "No Mask"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display frame in Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Stop manually by refreshing page or closing browser (no STOP button in Streamlit for now)

    cap.release()
    cv2.destroyAllWindows()
