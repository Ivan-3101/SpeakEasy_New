import cv2
import numpy as np
import os
import mediapipe as mp
import pygame
from gtts import gTTS
from tensorflow.keras.models import load_model

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

# Load the trained model
model = load_model("action_new.h5")

# Initialize variables
word_text = {
    "hello": "नमस्कार",
    "thanks": "धन्यवाद",
    "yes": "होय"
}

actions = list(word_text.keys())
threshold = 0.7
sequence = []
sentence = []

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

# Set the output directory for Marathi audio files
output_directory = "marathi_audio_files"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Generate and save Marathi audio files for each word
marathi_audio_files = {}
for word, text in word_text.items():
    tts = gTTS(text, lang="mr")
    audio_file_path = os.path.join(output_directory, f"{word}.mp3")
    tts.save(audio_file_path)
    marathi_audio_files[word] = audio_file_path

pygame.mixer.init()

def play_marathi_audio(word):
    audio_file = marathi_audio_files.get(word)
    if audio_file:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

cap = cv2.VideoCapture(0)

# Setting mediapipe model
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
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
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if len(res) > 0:
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
            else:
                print("Warning: Empty prediction result.")

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(
            image,
            " ".join(sentence),
            (3, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Output
        cv2.imshow("OpenCV Feed", image)

        # Break gracefully on "q" keypress
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
