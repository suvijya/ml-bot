import cv2
import os

# Print OpenCV installation path
opencv_path = os.path.dirname(cv2.__file__)
print("OpenCV installation path:", opencv_path)

# Path to Haar cascades
haar_cascade_path = os.path.join(opencv_path, 'data', 'haarcascades', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Check if Haar cascade loaded
if face_cascade.empty():
    print("Error: Haar cascade not loaded. Check the path.")
else:
    print("Haar cascade loaded successfully.")
