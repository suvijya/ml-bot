import cv2
import face_recognition
import numpy as np
import time
import pickle
import os
import speech_recognition as sr
import pyttsx3

# Initialize face database path
db_path = "face_database.pkl"

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

def speak(text):
    """Make the computer speak the given text."""
    try:
        print(f"Speaking: {text}")  # Debug print
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error during speech: {e}")

def load_face_database():
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            face_db = pickle.load(f)
        print("Loaded face database.")
    else:
        # Initialize the database structure
        face_db = {'names': [], 'branches': [], 'encodings': []}
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
    max_runtime = 15  # Max runtime in seconds

    while True:
        ret, frame = video_capture.read()

        # Check if frame was successfully captured
        if not ret:
            print("Failed to capture image. Retrying...")
            time.sleep(1)  # Delay before the next attempt
            continue

        # Resize frame and convert to RGB
        rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)

        print(f"Detected face locations: {face_locations}")  # Debug print

        if face_locations:
            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                print(f"Face encodings extracted: {face_encodings}")  # Debug print

                for face_encoding in face_encodings:
                    distances = face_recognition.face_distance(face_db['encodings'], face_encoding)
                    print(f"Distances from known faces: {distances}")  # Debug print

                    if len(distances) > 0:
                        min_distance = np.min(distances)
                        min_distance_index = np.argmin(distances)

                        if min_distance < tolerance:
                            name = face_db['names'][min_distance_index]
                            if min_distance_index < len(face_db['branches']):
                                branch = face_db['branches'][min_distance_index]
                            else:
                                branch = "unknown"  # Fallback if there's an index error
                                print("Warning: Branch index is out of range.")
                            print(f"Hello, {name}! Recognized with distance: {min_distance}")
                            speak(f"Hello, {name}! You study {branch}.")  # Voice greeting with branch
                        else:
                            print("New face detected!")
                            speak("New face detected! Please say your name:")  # Prompt for name
                            name = get_name_from_speech()  # Get name through speech
                            if name:
                                speak("Please say your branch of study:")  # Prompt for branch
                                branch = get_branch_from_speech()  # Get branch through speech
                                if branch:
                                    add_new_face_to_database(name, branch, face_encoding, face_db)
                    else:
                        print("New face detected!")
                        speak("New face detected! Please say your name:")  # Prompt for name
                        name = get_name_from_speech()  # Get name through speech
                        if name:
                            speak("Please say your branch of study:")  # Prompt for branch
                            branch = get_branch_from_speech()  # Get branch through speech
                            if branch:
                                add_new_face_to_database(name, branch, face_encoding, face_db)

            except Exception as e:
                print(f"Error in computing face encodings: {e}")

        if time.time() - start_time > max_runtime:
            print("Max runtime reached. Exiting.")
            break

    video_capture.release()
    cv2.destroyAllWindows()


def add_new_face_to_database(name, branch, face_encoding, face_db):
    # Add entries to the database
    face_db['names'].append(name)
    face_db['branches'].append(branch)
    face_db['encodings'].append(face_encoding)
    print(f"Before saving: {face_db}")  # Debug print to check database structure
    save_face_database(face_db)
    print(f"Added {name} with branch {branch} to the database.")

import speech_recognition as sr
import time

import speech_recognition as sr
import time

def get_name_from_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while True:
            # ... (rest of your code)

            confirmation_attempts = 3
            while confirmation_attempts > 0:
                try:
                    audio_confirmation = recognizer.listen(source, timeout=5)
                    confirmation = recognizer.recognize_google(audio_confirmation)
                    confirmation = confirmation.lower()

                    if confirmation in ["yes", "correct", "yeah", "yep", "absolutely"]:
                        return name
                    elif confirmation in ["no", "incorrect", "nope", "nah"]:
                        speak("Please say your name again.")
                        break
                    else:
                        speak("I didn't quite catch that. Please say yes or no.")
                        confirmation_attempts -= 1
                except sr.UnknownValueError:
                    print("Could not understand the audio.")
                    speak("I didn't catch that. Please say yes or no.")
                    confirmation_attempts -= 1
                except sr.RequestError as e:
                    print(f"Could not request results: {e}")
                    speak("There was an error. Please try again.")
                    return None

            if confirmation_attempts == 0:
                speak("I couldn't understand your confirmation. Please try again later.")
                return None



def get_branch_from_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say your branch of study:")
        audio = recognizer.listen(source)

        try:
            branch = recognizer.recognize_google(audio)
            print(f"Recognized branch: {branch}")
            return branch
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError:
            print("Could not request results; check your network connection.")
    return None

# Load the face database
face_db = load_face_database()

# Run the main recognition function
recognize_face(face_db)
