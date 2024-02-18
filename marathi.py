from gtts import gTTS
import os

# Define Marathi translations for each word
marathi_translations = {
    "hello": "नमस्कार",
    "thanks": "धन्यवाद",
    "yes": "होय",
}

# Set the output directory for Marathi audio files
output_directory = "marathi_audio_files"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Generate and save Marathi audio files for each word
for word, translation in marathi_translations.items():
    tts = gTTS(translation, lang="mr")
    audio_file_path = os.path.join(output_directory, f"{word}.mp3")
    tts.save(audio_file_path)
    print(f"Audio file saved for '{word}'")

print("Marathi audio files created successfully.")
