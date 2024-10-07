#!/usr/bin/python3
import datetime
time_start = datetime.datetime.now()

import cv2
import logging
import sys
import time
import datetime
import timeit
import os
import numpy as np

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        print(f"time to execute {func.__name__}: {timeit.default_timer() - start_time} seconds")
        return result
    return wrapper

# Logging configuration
logging.basicConfig(
    filename="/home/nikitamitin/PycharmProjects/larva/logfile.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.debug('Facial recognition script started')

# Constants
MAX_ATTEMPTS = 5
REQUIRED_MATCHES = 3
SLEEP_INTERVAL = 1
IMAGE_PATH = "/home/nikitamitin/PycharmProjects/larva/faces/"
FLAG_FILE = "/tmp/face_recognized"

# Loading the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

print("Time to init: ", datetime.datetime.now() - time_start)

def detect_face(image_path):
    """
    Detects a face in an image and returns the cropped face.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Unable to read image: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (200, 200))  # Resize for consistency
        return face_img
    else:
        logging.warning("No face detected in image: " + image_path)
        return None

def capture_face(user_id):
    """
    Captures 10 images of the user's face and saves them.
    """
    video_capture = cv2.VideoCapture(0)

    try:
        for i in range(10):
            ret, frame = video_capture.read()
            if not ret:
                logging.error("Failed to capture frame from webcam")
                break
            time.sleep(SLEEP_INTERVAL)

            # Face detection before saving
            face_img = detect_face_from_frame(frame)
            if face_img is not None:
                face_image_path = f"{IMAGE_PATH}{user_id}_{i}.jpg"
                cv2.imwrite(face_image_path, face_img)
                logging.debug(f"Saved face image: {face_image_path}")
            else:
                logging.warning("No face detected. Skipping save.")
    finally:
        video_capture.release()

def detect_face_from_frame(frame):
    """
    Detects a face in a given video frame and returns the cropped face.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (200, 200))  # Resize for consistency
        return face_img
    else:
        return None

def apply_clahe(image):
    """
    Applies CLAHE to enhance image contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def train_recognizer(user_id):
    """
    Trains the recognizer on 10 images of the specified user ID.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []

    for i in range(10):
        known_face_path = f"{IMAGE_PATH}{user_id}_{i}.jpg"
        if not os.path.exists(known_face_path):
            logging.warning(f"Stored image not found: {known_face_path}")
            continue
        face_img = detect_face(known_face_path)
        if face_img is not None:
            faces.append(face_img)
            labels.append(user_id)
        else:
            logging.warning(f"No face detected in image: {known_face_path}")

    if not faces:
        logging.error("No training data available")
        return None

    recognizer.train(faces, np.array(labels))
    logging.info(f"Recognizer trained for user {user_id} with {len(faces)} images.")
    return recognizer

@timing_decorator
def authenticate_user(recognizer):
    """
    Authenticates the user by comparing captured face images with trained recognizer.
    """
    logging.info("Authentication started!")
    video_capture = cv2.VideoCapture(0)
    attempts = 0

    if recognizer is None:
        logging.error("Recognizer not trained.")
        return 1

    try:
        while attempts < MAX_ATTEMPTS:
            ret, frame = video_capture.read()
            if not ret:
                logging.error("Failed to capture frame from webcam")
                return 1

            # Face detection in the captured frame
            face_img = detect_face_from_frame(frame)
            if face_img is None:
                logging.warning("No face detected in captured frame.")
                attempts += 1
                continue

            # Prediction using the recognizer
            label, confidence = recognizer.predict(face_img)
            logging.debug(f"Predicted label: {label}, Confidence: {confidence}")

            # Confidence threshold for successful match
            if confidence < 60:  # The lower the value, the better the match
                logging.info(f"Verification success for user with ID {label}")
                create_flag_file()
                return 0
            else:
                logging.info("Face did not match the trained user.")

            attempts += 1

        logging.warning("Too many attempts, please use a passcode.")
        return 1

    finally:
        video_capture.release()

def create_flag_file():
    """
    Creates a flag file that PAM checks for successful authentication.
    """
    try:
        with open(FLAG_FILE, 'w') as f:
            f.write('authenticated')
        os.chmod(FLAG_FILE, 0o600)
        logging.debug(f"Flag file {FLAG_FILE} created successfully.")
    except Exception as e:
        logging.error(f"Failed to create flag file: {e}")

def main():
    """
    Main function to initiate the authentication process.
    """
    user_id = 1
    #capture_face(user_id)
    recognizer = train_recognizer(user_id)
    result = authenticate_user(recognizer)
    sys.exit(result)

if __name__ == "__main__":
    main()

