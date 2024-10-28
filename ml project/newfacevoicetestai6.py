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
import requests
import datetime


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
    api_key = "key"  # Replace with your actual API key

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

    # Check if the face database is empty, if so, add a new face
    if not face_db['encodings']:
        print("Face database is empty. Starting with a new face entry.")
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture initial image. Exiting...")
                video_capture.release()
                return

            # Resize frame and detect face locations
            rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            if face_encodings:
                speak("New face detected! Please say your name.")
                name = get_name_from_speech()
                if name:
                    speak("Please say your branch of study.")
                    branch = get_branch_from_speech()
                    if branch:
                        speak(f"You said your name is {name} and your branch is {branch}. Is that correct?")
                        if vosk_yes_no_confirmation():
                            add_new_face_to_database(name, branch, face_encodings[0], face_db)
                            recognized = True  # Mark as recognized to break out of loops
                            print(f"Added {name} with branch {branch} to the database.")
                            break  # Exit new face entry loop after adding
                        else:
                            speak("Let's try again.")
            else:
                print("No face detected. Retrying...")

            # Check for max runtime during new face addition
            if time.time() - start_time > max_runtime:
                print("Max runtime reached while adding new face. Exiting...")
                break

    # If database is not empty, attempt recognition
    while not recognized and face_db['encodings']:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image. Retrying...")
            time.sleep(1)
            continue

        # Resize frame and convert to RGB
        rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print(f"Detected {len(face_locations)} faces in frame.")

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                # Average encodings for each person in the database
                avg_encodings = [np.mean(encs, axis=0) if isinstance(encs, list) else encs for encs in face_db['encodings']]

                # Try recognition with initial tolerance, increasing if not recognized
                tolerance = initial_tolerance
                while tolerance <= max_tolerance:
                    distances = face_recognition.face_distance(avg_encodings, face_encoding)
                    print(f"Distances: {distances} with tolerance {tolerance}")

                    if distances.size > 0:
                        min_distance = np.min(distances)
                        min_distance_index = np.argmin(distances)
                        print(f"Minimum distance: {min_distance}")

                        if min_distance < tolerance:
                            # Recognize the known face
                            name = face_db['names'][min_distance_index]
                            branch = face_db['branches'][min_distance_index]
                            print(f"Recognized {name}, branch: {branch}")
                            speak(f"Hello, {name}! You study {branch}.")
                            recognized = True  # Mark recognized to exit loop
                            break  # Break out of tolerance loop
                        tolerance += 0.05

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
                            break  # Exit recognition loop

        # Exit loop after recognition or reaching max runtime
        if recognized or (time.time() - start_time > max_runtime):
            print("Ending recognition loop.")
            break

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
