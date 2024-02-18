import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os
import pygame
from tensorflow.keras.models import load_model
import streamlit as st

# Initialize the audio mixer
pygame.mixer.init()
actions = np.array(['hello', 'thanks', 'yes'])

# Function to play Marathi audio
def play_marathi_audio(word):
    audio_file = marathi_audio_files.get(word)
    if audio_file:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

# Load the pre-trained model
model = load_model('action_new.h5')  # Replace 'path_to_your_model.h5' with the actual path

# Set up Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to perform Mediapipe detection on an image
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw styled landmarks on an image
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

# Function to extract keypoints from the Mediapipe results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Load the Marathi audio files
marathi_audio_files = {
    'hello': 'path/to/hello.mp3',
    'thanks': 'path/to/thanks.mp3',
    'yes': 'path/to/yes.mp3'
}

# Create Streamlit app
st.title("Sign Language Recognition App")

# Add a button to start video capture
start_button = st.button("Start Video Capture")

if start_button:
    cap = cv2.VideoCapture(0)

    # Setting the variables for the sentence and defining the threshold value for the predicted word to be shown
    sequence = []
    sentence = []
    threshold = 0.8

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Reading the feed
            ret, frame = cap.read()

            # Making detections
            image, results = mediapipe_detection(frame, holistic)

            # Drawing landmarks
            draw_styled_landmarks(image, results)

            # Defining the Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Ensuring the sequence length is maintained

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_word = actions[np.argmax(res)]
                print(predicted_word)

                # Visual and audio logic
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if predicted_word != sentence[-1]:
                            sentence.append(predicted_word)
                            play_marathi_audio(predicted_word)
                    else:
                        sentence.append(predicted_word)
                        play_marathi_audio(predicted_word)

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Output
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully on "q" keypress
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Add a button to stop video capture
stop_button = st.button("Stop Video Capture")

if stop_button:
    st.text("Video Capture Stopped")
