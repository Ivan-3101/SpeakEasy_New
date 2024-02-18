import streamlit as st
import cv2
import numpy as np

# Function to start video capture
def start_video_capture(stop_button):
    cap = cv2.VideoCapture(0)
    # Create an empty image placeholder
    video_placeholder = st.image([], channels="BGR", use_column_width=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the live video feed
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Check if the stop button is pressed
        if stop_button:
            break

    cap.release()

# Streamlit app
st.title("Live Video Capture App")

# Button to start video capture
start_button = st.button("Start Video Capture", key="start_button")

# Button to stop video capture
stop_button = st.button("Stop Video Capture", key="stop_button")

# Check if the start button is pressed
if start_button:
    # Call the function to start video capture
    start_video_capture(stop_button)
