import pygame
import os

pygame.mixer.init()

# Set the output directory for Marathi audio files
output_directory = "marathi_audio_files"

# Ensure the output directory exists
if not os.path.exists(output_directory):
    print(f"Error: Output directory '{output_directory}' not found. Make sure to run the script that generates the audio files.")

# Generate the paths for existing Marathi audio files
marathi_audio_files = {
    "hello": os.path.join(output_directory, "hello.mp3"),
    "thanks": os.path.join(output_directory, "thanks.mp3"),
    "yes": os.path.join(output_directory, "yes.mp3"),
}

def play_marathi_audio(word):
    audio_file = marathi_audio_files.get(word)
    if audio_file:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
