import streamlit as st
import requests
from PIL import Image
import cv2
import numpy as np
import io
import time

FASTAPI_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="Real-Time Drowsiness Detection")
st.title(" Driver Drowsiness Detection")
st.markdown("**Note:** This uses webcam and FastAPI for frame-by-frame prediction.")

run = st.checkbox("Start Webcam Detection")

frame_window = st.image([])

closed_counter = 0
warning_threshold = 15

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb_frame)
        resized_img = image_pil.resize((150, 150))
        
        # Convert to bytes
        buffer = io.BytesIO()
        resized_img.save(buffer, format="JPEG")
        buffer.seek(0)

        files = {"file": ("frame.jpg", buffer, "image/jpeg")}
        try:
            response = requests.post(FASTAPI_URL, files=files)
            result = response.json()
            top = max(result["predictions"], key=lambda x: x["probability"])
            label = top["class"]
            prob = top["probability"]

            if label == "Sleepy Eye":
                closed_counter += 1
            else:
                closed_counter = 0

            if closed_counter > warning_threshold:
                cv2.putText(frame, " Drowsiness Detected!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Add label to frame
            cv2.putText(frame, f"{label} ({prob:.2%})", (10, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            cv2.putText(frame, "Error: No connection to FastAPI", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.05)

    cap.release()
else:
    st.warning(" Check the box above to start detection.")
