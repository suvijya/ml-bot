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

# Initialize face database path and text-to-speech engine
db_path = "face_database.pkl"
tts_engine = pyttsx3.init()

def speak(text):
    """Make the computer speak the given text."""
    print(f"Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# Load Vosk model for yes/no responses
vosk_model_path = "vosk-model-small-en-us-0.15"
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
                speak(f"You said your name is {name}. Is that correct?")
                if vosk_yes_no_confirmation():
                    return name
                else:
                    speak("Okay, let's try again.")
            except sr.UnknownValueError:
                speak("I couldn't understand. Please say your name again.")
            except sr.WaitTimeoutError:
                speak("I didn't hear anything. Please try again.")
            except sr.RequestError:
                speak("There was an issue with the network connection.")
                return None

def vosk_yes_no_confirmation():
    """Uses Vosk for yes or no confirmation."""
    import pyaudio  # Ensure PyAudio is available

    # Set up the PyAudio stream
    p = pyaudio.PyAudio()
    stream = p.open(rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=8192)
    recognizer = KaldiRecognizer(vosk_model, 16000)
    
    speak("Please say 'yes' or 'no'.")  # Prompt user before listening
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            if "text" in result:
                answer = result["text"]
                print(f"Vosk detected response: {answer}")
                if "yes" in answer:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return True
                elif "no" in answer:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return False
                else:
                    speak("Please respond with yes or no.")
    
    # Close the stream in case of exit
    stream.stop_stream()
    stream.close()
    p.terminate()
    return False


def get_branch_from_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Please say your branch of study.")
        audio = recognizer.listen(source)
        try:
            branch = recognizer.recognize_google(audio)
            speak(f"You said {branch}. Is that correct?")
            if vosk_yes_no_confirmation():
                return branch
            else:
                speak("Let's try again.")
        except sr.UnknownValueError:
            speak("I couldn't understand. Please say your branch again.")
        except sr.RequestError:
            speak("There was an issue with the network connection.")
    return None

def add_new_face_to_database(name, branch, face_encoding, face_db):
    face_db['names'].append(name)
    face_db['branches'].append(branch)
    face_db['encodings'].append(face_encoding)
    save_face_database(face_db)
    print(f"Added {name} with branch {branch} to the database.")

def recognize_face(face_db, tolerance=0.6):
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return

    start_time = time.time()
    max_runtime = 15  # Max runtime in seconds (adjust as needed)
    recognized = False  # Flag to track successful recognition

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture image. Retrying...")
            time.sleep(1)
            continue

        rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)

        if face_locations:
            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    distances = face_recognition.face_distance(face_db['encodings'], face_encoding)

                    if len(distances) > 0:
                        min_distance = np.min(distances)
                        min_distance_index = np.argmin(distances)

                        if min_distance < tolerance:
                            name = face_db['names'][min_distance_index]
                            branch = face_db['branches'][min_distance_index]
                            speak(f"Hello, {name}! You study {branch}.")
                            
                            recognized = True  # Set recognized flag to true after greeting
                            break  # Exit the loop after recognition

                        else:
                            speak("New face detected! Please say your name.")
                            name = get_name_from_speech()
                            if name:
                                speak("Please say your branch of study.")
                                branch = get_branch_from_speech()
                                if branch:
                                    add_new_face_to_database(name, branch, face_encoding, face_db)
                                    recognized = True  # Set flag if new face added
                                    break
                    else:
                        speak("New face detected! Please say your name.")
                        name = get_name_from_speech()
                        if name:
                            speak("Please say your branch of study.")
                            branch = get_branch_from_speech()
                            if branch:
                                add_new_face_to_database(name, branch, face_encoding, face_db)
                                recognized = True  # Set flag if new face added
                                break

            except Exception as e:
                print(f"Error in computing face encodings: {e}")

        # Break from the main loop after recognition
        if recognized or (time.time() - start_time > max_runtime):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# Load the face database and run recognition
face_db = load_face_database()
recognize_face(face_db)
