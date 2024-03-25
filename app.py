import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
# from modules import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from tensorflow.keras.models import load_model
import os
import pygame  # Import pygame for audio playback
import gtts

# # Initialize pygame mixer
# pygame.init()
# pygame.mixer.init()



# Load the trained model
model = load_model("m26dropout.h5")

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


# Initialize variables
actions = ["नमस्कार", "धन्यवाद", "होय"]
threshold = 0.7
sequence = []
predicted_words = []


# Function to play audio
def play_audio(filename):
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()


# Center the title
st.markdown("<h1 style='text-align: center;'>SpeakEasy</h1>", unsafe_allow_html=True)
# Add "How to Use the App" card section
st.sidebar.markdown(
    """
    <div style='padding: 10px; border-radius: 10px; background-color: #070F2B;'>
        <h3 style='text-align: center;margin-bottom: 10px;'>How to Use the App</h3>
        <p>To use SpeakEasy, follow these steps:</p>
        <ol>
            <li>Click on the <strong>'Start Video Capture'</strong> button to begin capturing video.</li>
            <li>Perform sign language gestures in front of your camera.</li>
            <li>SpeakEasy will recognize the gestures and display the corresponding spoken words.</li>
            <li>Click on the <strong>'Stop Video Capture'</strong> button to stop capturing video.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add About card section
st.sidebar.markdown(
    """
    <div style='padding: 10px; margin-top: 20px; border-radius: 10px; background-color: #070F2B;'>
        <h3 style='text-align: center; margin-bottom: 10px;'>About</h3>
        <p>SpeakEasy is a Streamlit app for real-time sign language interpretation. It uses computer vision and machine learning techniques to recognize sign language gestures and translates them into spoken language.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cap = cv2.VideoCapture(0)

# Setting mediapipe model
holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
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
                        audio_file_path = f"marathi_audio_files/{predicted_word}.mp3"
                        play_audio(audio_file_path)

                    else:
                        st.warning("Empty prediction result.")

            # Display UI
            video_placeholder.image(image, channels="BGR", use_column_width=True)

            # Update the predicted word display
            if predicted_word:
                styled_text = f"<h3 style='text-align: center; color:green;'>Predicted Word: {predicted_word}</h3>"
                predicted_word_container.markdown(styled_text, unsafe_allow_html=True)
