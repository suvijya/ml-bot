import cv2
import face_recognition
import os
import pickle

# Load or create a face database
def load_face_database(db_path='face_database.pkl'):
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_face_database(face_database, db_path='face_database.pkl'):
    with open(db_path, 'wb') as f:
        pickle.dump(face_database, f)

def ask_and_store_name():
    if recognize_face(face_db) is None:
        name = ask_and_store_name()
        if name:
            add_new_face_to_database(name, face_db)
        

# Function to capture face encoding from camera
def capture_and_encode_face():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            video_capture.release()
            cv2.destroyAllWindows()
            return face_encodings[0], frame  # Return first face encoding found
        else:
            cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

# Add a new person to the database
def add_new_face_to_database(name, face_database):
    face_encoding, face_image = capture_and_encode_face()
    face_database[name] = face_encoding
    save_face_database(face_database)
    print(f"Face for {name} added to the database.")

# Recognize a face
def recognize_face(face_database):
    face_encoding, _ = capture_and_encode_face()
    for name, db_encoding in face_database.items():
        match = face_recognition.compare_faces([db_encoding], face_encoding)[0]
        if match:
            print(f"Hello, {name}!")
            return name
    print("Face not recognized. Would you like to add this person to the database?")
    return None

# Main
if __name__ == "__main__":
    face_db = load_face_database()

    # Recognize face or add new face
    person_name = recognize_face(face_db)
    if person_name is None:
        name = input("Enter your name: ")
        add_new_face_to_database(name, face_db)
