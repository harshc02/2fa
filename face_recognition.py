"""
Per-user face enrollment/verification using OpenCV.

Each username gets its own trained LBPH model file under face_models/,
so verifying user A's face can never accidentally match user B's model -
they're completely separate files, never compared against each other.

Uses:
  - Haar cascade for face detection (built into opencv, no extra download)
  - LBPH (Local Binary Patterns Histogram) recognizer for matching
    (this comes from opencv-contrib-python, NOT plain opencv-python)

This assumes a webcam attached to the machine running the Flask server
(fine for a local demo app like this one; a real web-deployed app would
need to capture frames in the browser and upload them instead).
"""

import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "face_models")
os.makedirs(MODELS_DIR, exist_ok=True)

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(
        "Failed to load Haar cascade file. Tried path:\n  %s\n"
        "File exists on disk: %s\n"
        "If this is missing, reinstall with: pip install \"opencv-contrib-python<5\" "
        "(opencv 5.0.0 shipped a known-broken wheel missing these data files)."
        % (FACE_CASCADE_PATH, os.path.exists(FACE_CASCADE_PATH))
    )


def _model_path(username):
    return os.path.join(MODELS_DIR, f"{username}.yml")


def _largest_face(gray_frame):
    """Return the cropped grayscale face region, or None if no face found."""
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray_frame[y : y + h, x : x + w]


def enroll_face(username, num_samples=35, camera_index=0):
    """Capture several frames of this user's face and train/save THEIR OWN model.
    More samples (35 vs the original 20) gives the model more pose/lighting
    variety to learn from, which helps it tell people apart more reliably -
    move your head slightly (turn a little left/right, tilt up/down) during
    enrollment instead of holding perfectly still, so the samples aren't
    all near-identical frames."""
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError("Could not access webcam (index %d)" % camera_index)

    samples = []
    attempts = 0
    max_attempts = num_samples * 15  # avoid an infinite loop if no face is found

    try:
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            ok, frame = cam.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = _largest_face(gray)
            if face is not None:
                face = cv2.resize(face, (200, 200))
                samples.append(face)
    finally:
        cam.release()

    if not samples:
        raise RuntimeError(
            "No face detected during enrollment. Check lighting/webcam and try again."
        )

    # A fresh recognizer per enrollment, trained only on this user's samples
    # and saved to a file scoped to just this username.
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = np.zeros(len(samples), dtype=np.int32)  # single label per user's own model
    recognizer.train(samples, labels)
    recognizer.save(_model_path(username))
    return True


def verify_face(username, confidence_threshold=50, max_frames=40, camera_index=0, required_consecutive_matches=3):
    """
    Capture frames until CONSECUTIVE confident matches against THIS user's
    own model (or max_frames is hit). Never compares against any other
    user's model.

    LBPH confidence is a DISTANCE, not a probability: LOWER = more similar.
    50 is stricter than the previous default of 70 - it cuts down on false
    accepts between different people, at the cost of occasionally rejecting
    the right person in poor lighting (tighten further/loosen if needed
    after testing with your actual faces and webcam).

    Requiring several consecutive matching frames (not just one lucky frame)
    further reduces the chance of a single noisy frame producing a false
    accept.
    """
    model_path = _model_path(username)
    if not os.path.exists(model_path):
        raise RuntimeError(f"No enrolled face found for '{username}'. Enroll first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError("Could not access webcam (index %d)" % camera_index)

    consecutive = 0
    matched = False
    try:
        for _ in range(max_frames):
            ok, frame = cam.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = _largest_face(gray)
            if face is None:
                consecutive = 0
                continue
            face = cv2.resize(face, (200, 200))
            _label, confidence = recognizer.predict(face)
            if confidence < confidence_threshold:
                consecutive += 1
                if consecutive >= required_consecutive_matches:
                    matched = True
                    break
            else:
                consecutive = 0
    finally:
        cam.release()

    return matched
