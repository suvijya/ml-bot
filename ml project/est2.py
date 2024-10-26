import speech_recognition as sr

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
        
        # Save the captured audio
        save_audio(audio)

def save_audio(audio_data):
    # Save audio data to a file
    with open("test_audio.wav", "wb") as f:
        f.write(audio_data.get_wav_data())
    print("Audio saved as 'test_audio.wav'. You can listen to it.")

# Run the function
test_microphone_and_save()
