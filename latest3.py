import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from modules import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from tensorflow.keras.models import load_model
from marathi_audio import play_marathi_audio  # Import the play_marathi_audio function
import os

# Load the trained model
model = load_model("action.h5")

# Initialize variables
actions = ["hello", "thanks", "yes"]
threshold = 0.7
sequence = []
predicted_words = []

# Center the title
st.markdown("<h1 style='text-align: center;'>SpeakEasy</h1>", unsafe_allow_html=True)

# Add "How to Use the App" section
st.sidebar.subheader("How to Use the App")
st.sidebar.write(
    "To use SpeakEasy, follow these steps:\n"
    "1. Click on the 'Start Video Capture' button to begin capturing video.\n"
    "2. Perform sign language gestures in front of your camera.\n"
    "3. SpeakEasy will recognize the gestures and display the corresponding spoken words.\n"
    "4. Click on the 'Stop Video Capture' button to stop capturing video.\n"
)

# Add About section
st.sidebar.subheader("About")
st.sidebar.write(
    "SpeakEasy is a Streamlit app for real-time sign language interpretation. "
    "It uses computer vision and machine learning techniques to recognize sign language gestures "
    "and translates them into spoken language."
)

cap = cv2.VideoCapture(0)

# Setting mediapipe model
holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

with holistic as holistic:

    # Create a container for the predicted word
    predicted_word_container = st.empty()

    # Display UI for video feed
    video_placeholder = st.empty()

    # Create empty columns
    empty_col1, btn_col1, btn_col2, empty_col2 = st.columns([1, 1, 1, 1])

    # Center the buttons
    with btn_col1:
        start_button = st.button("Start Video Capture")

    with btn_col2:
        stop_button = st.button("Stop Video Capture")

    # Initialize an expander for predicted words
    with st.expander("Predicted Words", expanded=False) as predicted_words_expander:
        predicted_words_text = st.empty()

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

            # Update the predicted word display
            if predicted_word:
                styled_text = f"<h3 style='text-align: center; color:green;'>Predicted Word: {predicted_word}</h3>"
                predicted_word_container.markdown(styled_text, unsafe_allow_html=True)

            # Update predicted words inside the expander
            predicted_words_text.write(" ".join(predicted_words), unsafe_allow_html=True)
