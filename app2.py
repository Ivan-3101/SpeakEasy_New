import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from modules import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from tensorflow.keras.models import load_model
import pygame
from gtts import gTTS
import os

# Load the trained model
model = load_model("action_new.h5")

# Initialize variables
actions = ["hello", "thanks", "yes"]
threshold = 0.7
sequence = []
predicted_words = []

# Set the output directory for Marathi audio files
output_directory = "marathi_audio_files"

# Ensure the output directory exists
if not os.path.exists(output_directory):
    st.error(f"Error: Output directory '{output_directory}' not found. Make sure to run the script that generates the audio files.")
    st.stop()

# Generate the paths for existing Marathi audio files
marathi_audio_files = {
    "hello": os.path.join(output_directory, "hello.mp3"),
    "thanks": os.path.join(output_directory, "thanks.mp3"),
    "yes": os.path.join(output_directory, "yes.mp3"),
}

pygame.mixer.init()

def play_marathi_audio(word):
    audio_file = marathi_audio_files.get(word)
    if audio_file:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

cap = cv2.VideoCapture(0)

# Setting mediapipe model
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    st.title("Sign Language Interpretation App")

    # Create a state variable to store the last 5 predicted words
    last_five_words = st.empty()

    # Button to start video capture
    start_button = st.button("Start Video Capture")

    # Button to stop video capture
    stop_button = st.button("Stop Video Capture")

    # Display UI for video feed
    video_placeholder = st.image([], channels="BGR", use_column_width=True)

    while cap.isOpened() and not stop_button:
        if start_button:
            ret, frame = cap.read()
            if not ret:
                break

            # Making detections
            image, results = mediapipe_detection(frame, holistic)

            # Drawing landmarks
            draw_styled_landmarks(image, results)

            # Defining the Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Ensuring the sequence length is maintained

            res = None
            predicted_word = None

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                if len(res) > 0:
                    predicted_word = actions[np.argmax(res)]

                    # Visual and audio logic
                    if res[np.argmax(res)] > threshold:
                        predicted_words.append(predicted_word)
                        play_marathi_audio(predicted_word)
                    else:
                        st.warning("Empty prediction result.")

            # Display UI
            video_placeholder.image(image, channels="BGR", use_column_width=True)

            # Update the last 5 predicted words display
            last_five_words.write("Last 5 Predicted Words:")
            last_five_words.write(" ".join(predicted_words[-5:]))

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()