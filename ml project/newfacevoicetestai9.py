import cv2
#import face_recognition
import numpy as np
import time
import pickle
import os
import speech_recognition as sr
import pyttsx3
from vosk import Model, KaldiRecognizer
import json
import requests
import datetime
import pyaudio


# Initialize face database path and text-to-speech engine
db_path = "face_database.pkl"
tts_engine = pyttsx3.init()

# Define the path to the Haar cascade file directly
haar_cascade_path = r'C:\Users\Suvijya A\.conda\envs\face_voice_env\Lib\site-packages\cv2\data\haarcascades\haarcascade_frontalface_default.xml'
# Load the Haar cascade
face_cascade = cv2.CascadeClassifier(haar_cascade_path)


def speak(text):
    """Make the computer speak the given text."""
    print(f"Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# Load Vosk model for yes/no responses
# Define paths for each model
vosk_model_general_path = "vosk-model-small-en-us-0.15"  # General English model
vosk_model_indian_path = "vosk-model-en-in-0.5"    # Indian English model
vosk_model_hindi_path = "vosk-model-small-hi-0.22"    # Hindi model

# Load each model
vosk_model_general = Model(vosk_model_general_path)
vosk_model_indian = Model(vosk_model_indian_path)
vosk_model_hindi = Model(vosk_model_hindi_path)

def recognize_speech(model):
    """Use the specified Vosk model for speech recognition."""
    recognizer = KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=8192)

    stream.start_stream()
    print("Listening...")
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            if "text" in result:
                print("Recognized:", result["text"])
                break  # Break after one recognition or continue as needed
    stream.stop_stream()
    stream.close()
    p.terminate()
    return result.get("text", "")

def recognize_language(language="english"):
    """Select model based on the language specified."""
    if language.lower() == "hindi":
        print("Using Hindi model:")
        return recognize_speech(vosk_model_hindi)
    elif language.lower() == "indian english":
        print("Using Indian English model:")
        return recognize_speech(vosk_model_indian)
    else:
        print("Using General English model:")
        return recognize_speech(vosk_model_general)

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

def get_face_encoding(face_image):
    """Compute the encoding for a detected face using OpenCV."""
    # Resize face image and normalize it to standard dimensions (for this example, 128x128)
    resized_face = cv2.resize(face_image, (128, 128))
    normalized_face = resized_face.flatten() / 255.0  # Flatten and normalize to [0, 1]
    return normalized_face

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

def vosk_yes_no_confirmation(language="en"):
    """Uses Vosk for yes or no confirmation, supports English and Hindi models."""
    import pyaudio  # Ensure PyAudio is available

    # Set up the PyAudio stream
    p = pyaudio.PyAudio()
    stream = p.open(rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=8192)

    # Select the appropriate Vosk model based on language
    recognizer = KaldiRecognizer(vosk_model_general if language == "en" else vosk_model_hindi, 16000)
    
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


def get_command_from_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=10)
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            speak("I couldn't understand. Please say it again.")
            return None
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Please try again.")
            return None
        except sr.RequestError:
            speak("There was an issue with the network connection.")
            return None

import datetime

def get_time():
    """Fetch the current time and speak it."""
    now = datetime.datetime.now()
    current_time = now.strftime("%I:%M %p")  # Format time as HH:MM AM/PM
    speak(f"The current time is {current_time}.")

def get_date():
    """Fetch the current date and speak it."""
    now = datetime.datetime.now()
    current_date = now.strftime("%B %d, %Y")  # Format date as Month Day, Year
    speak(f"Today's date is {current_date}.")


import requests

def get_city_from_speech():
    """Prompt the user to say the city name and return it."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Please say the city you want the weather for.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            audio = recognizer.listen(source, timeout=10)
            city = recognizer.recognize_google(audio)
            speak(f"You said {city}. Is that correct?")
            if vosk_yes_no_confirmation():
                return city
            else:
                speak("Let's try again.")
                return get_city_from_speech()  # Retry if incorrect
        except sr.UnknownValueError:
            speak("I couldn't understand. Please say the city again.")
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Please try again.")
        except sr.RequestError:
            speak("There was an issue with the network connection.")
            return None
    return None

def get_weather():
    """Fetch the current weather information for a specific city and speak it."""
    api_key = "53b87d8341ebc5c55fdddb82cf37439a"  # Replace with your actual API key

    # Prompt user for the city name
    city = get_city_from_speech()
    if not city:
        speak("Sorry, I couldn't get the city name.")
        return

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        weather_data = response.json()

        if weather_data.get("cod") != 200:
            speak("I'm sorry, I couldn't fetch the weather information.")
            print("Error:", weather_data.get("message"))
            return

        temperature = weather_data["main"]["temp"]
        description = weather_data["weather"][0]["description"]
        speak(f"The current temperature in {city} is {temperature} degrees Celsius with {description}.")
    except Exception as e:
        speak("There was an error fetching the weather information.")
        print("Exception:", e)

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
    if name in face_db['names']:
        idx = face_db['names'].index(name)
        
        # Append the new encoding to existing encodings
        if isinstance(face_db['encodings'][idx], list):
            face_db['encodings'][idx].append(face_encoding)
        else:
            face_db['encodings'][idx] = [face_db['encodings'][idx], face_encoding]
        
        # Update the average encoding
        avg_encoding = np.mean(face_db['encodings'][idx], axis=0)
        face_db['encodings'][idx] = avg_encoding
    else:
        # For a new face, add the name, branch, and initial encoding
        face_db['names'].append(name)
        face_db['branches'].append(branch)
        face_db['encodings'].append([face_encoding])  # Save as list of encodings
    
    save_face_database(face_db)
    print(f"Added/Updated {name} with branch {branch} to the database.")

def recognize_face(face_db, initial_tolerance=0.5, max_tolerance=0.6):
    print("Initializing video capture...")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return

    start_time = time.time()
    max_runtime = 15  # Max runtime in seconds
    recognized = False  # Flag to end recognition after success

    # Check if face database is empty
    if not face_db['encodings']:
        print("Face database is empty. Starting with a new face entry.")
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture initial image. Exiting...")
                video_capture.release()
                return

            # Convert frame to grayscale for Haar Cascade detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_image = frame[y:y + h, x:x + w]
                face_encoding = get_face_encoding(face_image)

                if face_encoding is not None:
                    speak("New face detected! Please say your name.")
                    name = get_name_from_speech()
                    if name:
                        speak("Please say your branch of study.")
                        branch = get_branch_from_speech()
                        if branch:
                            speak(f"You said your name is {name} and your branch is {branch}. Is that correct?")
                            if vosk_yes_no_confirmation():
                                add_new_face_to_database(name, branch, face_encoding, face_db)
                                recognized = True
                                print(f"Added {name} with branch {branch} to the database.")
                                break
                            else:
                                speak("Let's try again.")

            if time.time() - start_time > max_runtime:
                print("Max runtime reached while adding new face. Exiting...")
                break

    # Face recognition loop
    while not recognized and face_db['encodings']:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image. Retrying...")
            time.sleep(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_image = frame[y:y + h, x:x + w]
            face_encoding = get_face_encoding(face_image)

            if face_encoding is not None:
                distances = [np.linalg.norm(face_encoding - np.array(enc)) for enc in face_db['encodings']]
                min_distance = min(distances) if distances else float("inf")
                min_distance_index = distances.index(min_distance) if distances else -1

                if min_distance < initial_tolerance:
                    name = face_db['names'][min_distance_index]
                    branch = face_db['branches'][min_distance_index]
                    print(f"Recognized {name}, branch: {branch}")
                    speak(f"Hello, {name}! You study {branch}.")
                    recognized = True
                    break

        if recognized:
            break  # Break out of face_encoding loop if recognized

    # If face not recognized, prompt for new face entry
    if not recognized:
        speak("New face detected! Please say your name.")
        name = get_name_from_speech()
        if name:
            speak("Please say your branch of study.")
            branch = get_branch_from_speech()
            if branch:
                add_new_face_to_database(name, branch, face_encoding, face_db)
                recognized = True  # Mark recognized after new entry
                print(f"Added {name} with branch {branch} to the database.")

    # Continue to ask for commands until the user says "stop" or "exit"
    if recognized:
        while True:
            speak("What would you like to hear? You can ask about the time, date, weather, or anything else.")
            command = get_command_from_speech()  # You'll need to implement this method
            if command is None:
                continue
            
            command = command.lower()
            if "weather" in command:
                get_weather()  # Fetch weather info
            elif "time" in command:
                get_time()  # Fetch current time (implement this function)
            elif "date" in command:
                get_date()  # Fetch current date (implement this function)
            elif "stop" in command or "exit" in command:
                speak("Goodbye!")
                break
            else:
                speak("Sorry, I didn't understand that. Please ask again.")

    # Release video capture and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()
    print("Video capture ended.")

def get_user_request():
    """Capture what the user wants to hear."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        speak("What would you like to hear?")
        
        while True:
            try:
                audio = recognizer.listen(source, timeout=10)
                request = recognizer.recognize_google(audio)
                print(f"You said: {request}")
                return request
            except sr.UnknownValueError:
                speak("I couldn't understand that. Please say it again.")
            except sr.WaitTimeoutError:
                speak("I didn't hear anything. Please try again.")
            except sr.RequestError:
                speak("There was an issue with the network connection.")
                return None

def handle_user_request(request):
    """Handle the user's request based on the spoken input."""
    if "weather" in request.lower():
        get_weather()
    elif "time" in request.lower():
        current_time = time.strftime("%H:%M:%S", time.localtime())
        speak(f"The current time is {current_time}.")
    elif "date" in request.lower():
        current_date = time.strftime("%Y-%m-%d", time.localtime())
        speak(f"Today's date is {current_date}.")
    else:
        speak("I'm not sure how to respond to that. Please ask about the weather, time, or date.")

def main():
    face_db = load_face_database()
    recognize_face(face_db)

if __name__ == "__main__":
    main()
