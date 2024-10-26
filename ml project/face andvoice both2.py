import cv2
import face_recognition
import numpy as np
import time
import pickle
import os
import speech_recognition as sr
import pyttsx3
from vosk import Model, KaldiRecognizer
import json

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

# Load Vosk model
vosk_model_path = "vosk-model-small-en-us-0.15"  # Path to Vosk model
if not os.path.exists(vosk_model_path):
    raise Exception("Vosk model not found. Download it and place it in the specified path.")
vosk_model = Model(vosk_model_path)

def load_face_database():
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            face_db = pickle.load(f)
        print("Loaded face database.")
    else:
        face_db = {'names': [], 'branches': [], 'encodings': []}
    return face_db

def save_face_database(face_db):
    with open(db_path, "wb") as f:
        pickle.dump(face_db, f)
    print("Face database saved.")

def get_name_from_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        speak("Please say your name.")
        
        while True:
            try:
                audio = recognizer.listen(source, timeout=10)
                name = recognizer.recognize_google(audio)
                print(f"Recognized name: {name}")
                
                speak(f"You said your name is {name}. Is that correct?")
                if vosk_yes_no_confirmation():
                    return name
                else:
                    speak("Okay, let's try again.")
            except sr.UnknownValueError:
                speak("I couldn't understand. Please say your name again.")
            except sr.WaitTimeoutError:
                speak("I didn't hear anything. Please try again.")
            except sr.RequestError as e:
                speak("There was an issue with the network connection.")
                return None

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

def vosk_yes_no_confirmation():
    """Uses Vosk to confirm yes or no response."""
    with sr.Microphone() as source:
        recognizer = KaldiRecognizer(vosk_model, 16000)
        recognizer.SetWords(True)
        print("Please say 'yes' or 'no':")  # Prompt user
        speak("Please say 'yes' or 'no'.")

        while True:
            audio = source.listen(source, timeout=5)
            if recognizer.AcceptWaveform(audio.get_wav_data()):
                result = json.loads(recognizer.Result())
                if "text" in result:
                    answer = result["text"]
                    print(f"Vosk detected response: {answer}")
                    if "yes" in answer:
                        return True
                    elif "no" in answer:
                        return False
                    else:
                        speak("I didn't quite catch that. Please say yes or no.")
    return False

# Load the face database
face_db = load_face_database()

# Run the main recognition function
recognize_face(face_db)
