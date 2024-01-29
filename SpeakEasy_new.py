import cv2
import numpy as np
import os
import mediapipe as mp
import pygame
from gtts import gTTS
from tensorflow.keras.models import load_model

from modules import mediapipe_detection, draw_styled_landmarks, extract_keypoints

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Load the trained model
model = load_model("action.h5")

# Initialize variables
word_text = {"hello": "नमस्कार", "thanks": "धन्यवाद", "yes": "होय"}

actions = list(word_text.keys())
threshold = 0.7
sequence = []
sentence = []

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

# Load images
image_directory = "marathi_images"
image_files = {word: os.path.join(image_directory, f"{word}.png") for word in word_text.keys()}
word_images = {word: cv2.imread(image_files[word]) for word in word_text.keys()}

pygame.mixer.init()

def play_marathi_audio(word):
    audio_file = marathi_audio_files.get(word)
    if audio_file:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

cap = cv2.VideoCapture(0)

# Setting mediapipe model
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

        res = None  # Define res before using it
        predicted_word = None  # Define predicted_word outside of the condition

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if len(res) > 0:
                predicted_word = actions[np.argmax(res)]

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

        # Display UI
        ui_text = " ".join(sentence)
        ui_color = (0, 255, 0) if res is not None and res[np.argmax(res)] > threshold else (0, 0, 255)
        cv2.rectangle(image, (0, 0), (640, 40), ui_color, -1)
        cv2.putText(
            image,
            f"Marathi: {ui_text}",
            (3, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Iterate over words and images
        for i, (word, img) in enumerate(zip(word_text.keys(), word_images.values())):
            # Resize word images
            img_resized = cv2.resize(img, (40, 40))
            
            # Display word images along with the output
            start_x = i * 60
            end_x = start_x + 40
            image[40:80, start_x:end_x] = img_resized

            # Display predicted word for each image
            if predicted_word:
                cv2.putText(
                    image,
                    f"Predicted: {predicted_word}",
                    (start_x, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Output
        cv2.imshow("OpenCV Feed", image)

        # Break gracefully on "q" keypress
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
