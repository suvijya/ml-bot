import cv2
import face_recognition
import speech_recognition as sr
import os

def test_microphone_and_save():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        print("Please say something:")
        
        # Capture the audio
        audio = recognizer.listen(source)
        
        # Try recognizing the speech
        try:
            print("Recognizing speech...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
        except sr.RequestError:
            print("Could not request results; check your network connection.")
        
        # Save the captured audio
        save_audio(audio)

def save_audio(audio_data):
    # Save audio data to a file
    with open("test_audio.wav", "wb") as f:
        f.write(audio_data.get_wav_data())
    print("Audio saved as 'test_audio.wav'. You can listen to it.")


# Function to recognize faces
def recognize_face(face_db):
    # Load the image from the webcam
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face is in the database
            matches = face_recognition.compare_faces(face_db['encodings'], face_encoding)
            name = "Unknown"

            # If a match is found, get the name
            if True in matches:
                first_match_index = matches.index(True)
                name = face_db['names'][first_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to recognize speech and return the name
def ask_and_store_name():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say your name:")
        audio = recognizer.listen(source)

        try:
            name = recognizer.recognize_google(audio)
            print(f"Name recognized: {name}")
            return name
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None

# Example face database (to be replaced with your actual data)
face_db = {
    'encodings': [],  # Populate with actual face encodings
    'names': []       # Corresponding names
}

# Call the recognition function
recognize_face(face_db)
test_microphone_and_save()

