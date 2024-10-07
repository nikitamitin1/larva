#!/usr/bin/python3
import datetime
time_start = datetime.datetime.now()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow warnings

import cv2
from deepface import DeepFace
import logging
import sys
import time
import datetime
import timeit

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print(f"time to execute {func.__name__}: {end_time - start_time} seconds")
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
MAX_ATTEMPTS = 5           # Maximum number of authentication attempts
REQUIRED_MATCHES = 1       # Number of successful matches required for authentication
SLEEP_INTERVAL = 1         # Pause between frames in seconds
IMAGE_PATH = "/home/nikitamitin/PycharmProjects/larva/faces/"
FLAG_FILE = "/tmp/face_recognized"  # File flag for PAM authentication

# Load Haar Cascade for face detection

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# DeepFace model configuration
MODEL_NAME = "Dlib"  # Using the Facenet model
logging.debug(f'Loading model: {MODEL_NAME}')
# model = DeepFace.build_model(MODEL_NAME)

print("Time to init: ", datetime.datetime.now() - time_start)

def detect_face(image_path):
    """
    Detects a face in an image and returns the cropped grayscale face.
    :param image_path: Path to the image file
    :return: Cropped face image or None if no face is detected
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
        face_img = image[y:y + h, x:x + w]  # Return the colored face region
        return face_img
    else:
        logging.warning("No face detected in image: " + image_path)
        return None


def capture_face(user_id):
    """
    Captures 10 face images of the user and saves them.
    :param user_id: User identifier used in naming the saved images
    """
    video_capture = cv2.VideoCapture(0)

    try:
        for i in range(10):
            ret, frame = video_capture.read()
            if not ret:
                logging.error("Failed to capture frame from webcam")
                break
            time.sleep(SLEEP_INTERVAL)

            # Detect face before saving
            face_img = detect_face_from_frame(frame)
            if face_img is not None:
                face_image_path = f"{IMAGE_PATH}{user_id}_{i}.jpg"
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                gray = apply_clahe(gray)
                cv2.imwrite(face_image_path, gray)
                logging.debug(f"Saved face image: {face_image_path}")
            else:
                logging.warning("No face detected. Skipping save.")
    finally:
        video_capture.release()


def detect_face_from_frame(frame):
    """
    Detects a face in a given video frame and returns the cropped face.
    :param frame: Video frame as a numpy array
    :return: Cropped face image or None if no face is detected
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = frame[y:y + h, x:x + w]  # Return the colored face region
        return face_img
    else:
        return None


def apply_clahe(image):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the input image.
    :param image: Grayscale image (numpy array)
    :return: Image with enhanced contrast
    """
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    return clahe.apply(image)

@timing_decorator
def authenticate_user(known_ids):
    """
    Authenticates the user by capturing frames and comparing face regions with stored face images.
    :param known_ids: List of user IDs to authenticate against
    :return: 0 if authentication is successful, 1 otherwise
    """
    logging.info("Authentication started!")
    video_capture = cv2.VideoCapture(0)
    verif_count = 0
    attempts = 0
    temp_file = "/home/nikitamitin/PycharmProjects/larva/temp_face.jpg"

    try:
        while attempts < MAX_ATTEMPTS:
            ret, frame = video_capture.read()
            if not ret:
                logging.error("Failed to capture frame from webcam")
                return 1

            # Detect face in the captured frame
            face_img = detect_face_from_frame(frame)
            if face_img is None:
                logging.warning("No face detected in captured frame.")
                attempts += 1
                continue

            # Save face region to temporary file for comparison
            cv2.imwrite(temp_file, face_img)
            logging.debug(f"Captured and saved face frame {attempts + 1}/{MAX_ATTEMPTS}.")

            # Check captured face against stored user images
            for user_id in known_ids:
                if verif_count >= REQUIRED_MATCHES:
                    logging.info(f"Authenticated: {user_id}")
                    # Create the flag file for PAM
                    create_flag_file()
                    return 0

                for i in range(10):
                    known_face_path = f"{IMAGE_PATH}{user_id}_{i}.jpg"
                    if not os.path.exists(known_face_path):
                        logging.warning(f"Stored image not found: {known_face_path}")
                        continue

                    try:
                        result = DeepFace.verify(
                            temp_file,
                            known_face_path,
                            model_name=MODEL_NAME,
                            enforce_detection=False,
                            detector_backend="opencv",
                            silent=True,
                            normalization='Facenet',
                        )
                        if result['verified']:
                            verif_count += 1
                            logging.info(f"Verification success for {user_id}, count: {verif_count}")
                            if verif_count >= REQUIRED_MATCHES:
                                # Create the flag file for PAM
                                create_flag_file()
                                return 0
                    except Exception as e:
                        logging.error(f"Error processing {user_id}_{i}.jpg: {e}")

            attempts += 1

        logging.warning("Too many attempts, please use a passcode.")
        return 1

    finally:
        video_capture.release()
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def create_flag_file():
    """
    Creates a flag file that PAM checks for successful authentication.
    """
    try:
        with open(FLAG_FILE, 'w') as f:
            f.write('authenticated')
        # Set file permissions to be readable only by root
        os.chmod(FLAG_FILE, 0o600)
        logging.debug(f"Flag file {FLAG_FILE} created successfully.")
    except Exception as e:
        logging.error(f"Failed to create flag file: {e}")


def main():
    """
    Main function to initiate the authentication process.
    """
    user_id = 1
    known_ids = [user_id]
    # Uncomment the following line to capture face images for the user
    # capture_face(user_id)
    result = authenticate_user(known_ids)
    # Exit after authentication attempt
    sys.exit(result)


if __name__ == "__main__":
    main()


