import wave
import math
import struct
import os

# Crée le dossier sounds si il n'existe pas
os.makedirs("sounds", exist_ok=True)

filename = "sounds/beep.wav"
framerate = 44100
duration = 0.5  # demi-seconde
frequency = 440  # La note A4
amplitude = 32767

with wave.open(filename, "w") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(framerate)

    for i in range(int(framerate * duration)):
        value = int(amplitude * math.sin(2 * math.pi * frequency * i / framerate))
        wav_file.writeframes(struct.pack('<h', value))

print(f"Fichier audio créé : {filename}")