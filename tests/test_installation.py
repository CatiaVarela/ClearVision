vscode_lsp_terminal_prompt_tracker= {}
import sys
import torch
import cv2
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except: print("PyTorch")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except: print("OpenCV")

try:
    from ultralytics import YOLO
    print(f"Ultralytics: {YOLO.__version__}")
except: print(" Ultralytics")

try:
    import numpy
    print(f"✅ NumPy: {numpy.__version__}")
except: print("❌ NumPy")

try:
    import pygame
    print(f"✅ Pygame: {pygame.version.ver}")
except: print("❌ Pygame")