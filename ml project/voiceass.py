import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os

# Initialize the recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to capture audio and convert to text
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            query = recognizer.recognize_google(audio, language="en-US")
            print(f"User said: {query}\n")
        except sr.UnknownValueError:
            speak("Sorry, I did not catch that. Please repeat.")
            return "None"
        return query.lower()

# Main function to process voice commands
def process_command():
    query = listen()

    if 'time' in query:
        # Tells the current time
        time = datetime.datetime.now().strftime('%I:%M %p')
        speak(f"The time is {time}")

    elif 'open browser' in query:
        # Open the default web browser
        speak("Opening your browser")
        webbrowser.open("https://www.google.com")

    elif 'play music' in query:
        # Assuming you have music in a folder called "Music"
        music_dir = os.path.expanduser("~/Music")
        songs = os.listdir(music_dir)
        if songs:
            os.startfile(os.path.join(music_dir, songs[0]))  # Play the first song
            speak("Playing music")
        else:
            speak("No music found in the Music folder.")

    elif 'exit' in query or 'stop' in query:
        speak("Goodbye!")
        exit()

    else:
        speak("I am not sure how to respond to that.")

# Run the assistant in a loop
if __name__ == "__main__":
    speak("Hello! I am your voice assistant. How can I help you today?")
    while True:
        process_command()
