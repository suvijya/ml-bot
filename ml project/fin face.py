import cv2
import face_recognition
import numpy as np
import time
import pickle
import os

# Initialize face database path
db_path = "face_database.pkl"

def load_face_database():
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            face_db = pickle.load(f)
        print("Loaded face database.")
    else:
        face_db = {'names': [], 'encodings': []}
    return face_db

def save_face_database(face_db):
    with open(db_path, "wb") as f:
        pickle.dump(face_db, f)
    print("Face database saved.")

def recognize_face(face_db, tolerance=0.6):
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return

    start_time = time.time()  # Track elapsed time
    max_runtime = 3  # Max runtime in seconds

    while True:
        ret, frame = video_capture.read()

        # Check if frame was successfully captured
        if not ret:
            print("Failed to capture image. Retrying...")
            time.sleep(1)  # Delay before the next attempt
            continue

        # Resize frame and convert to RGB
        rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Optional resizing for faster processing
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)

        print(f"Detected face locations: {face_locations}")  # Debug print

        if face_locations:
            # Extract face encodings for all detected faces
            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                print(f"Face encodings extracted: {face_encodings}")  # Debug print

                # Proceed with face recognition for each encoding
                for face_encoding in face_encodings:
                    distances = face_recognition.face_distance(face_db['encodings'], face_encoding)
                    print(f"Distances from known faces: {distances}")  # Debug print for distances

                    # Find the best match
                    if len(distances) > 0:
                        min_distance = np.min(distances)
                        min_distance_index = np.argmin(distances)

                        if min_distance < tolerance:
                            name = face_db['names'][min_distance_index]
                            print(f"Hello, {name}! Recognized with distance: {min_distance}")
                        else:
                            print("New face detected!")
                            name = input("Enter name for the new face: ")  # Store user input for new face name
                            if name:
                                add_new_face_to_database(name, face_encoding, face_db)
                    else:
                        # Case for when no known faces are in the database
                        print("New face detected!")
                        name = input("Enter name for the new face: ")  # Store user input for new face name
                        if name:
                            add_new_face_to_database(name, face_encoding, face_db)

            except Exception as e:
                print(f"Error in computing face encodings: {e}")

        if time.time() - start_time > max_runtime:
            print("Max runtime reached. Exiting.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

def add_new_face_to_database(name, face_encoding, face_db):
    face_db['names'].append(name)
    face_db['encodings'].append(face_encoding)
    save_face_database(face_db)
    print(f"Added {name} to the database.")

# Load the face database
face_db = load_face_database()

# Run the main recognition function
recognize_face(face_db)
