import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
IMAGE_SIZE = (100, 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict gender from a face image or webcam.")
    parser.add_argument("--image", help="Path to an image. If omitted, webcam is used.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    return parser.parse_args()


def load_model():
    model_path = MODELS_DIR / "gender_model.pkl"
    encoder_path = MODELS_DIR / "label_encoder.pkl"
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise FileNotFoundError("Missing trained model. Run: python train_gender.py")
    if not encoder_path.exists() or encoder_path.stat().st_size == 0:
        raise FileNotFoundError("Missing label encoder. Run: python train_gender.py")
    return joblib.load(model_path), joblib.load(encoder_path)


def predict_face(model, encoder, gray_face):
    face = cv2.resize(gray_face, IMAGE_SIZE).flatten().reshape(1, -1) / 255.0
    label_index = model.predict(face)[0]
    gender = encoder.inverse_transform([label_index])[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(face)))
    return gender, confidence


def detect_faces(gray):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Could not load Haar cascade for face detection.")
    return face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))


def predict_image(model, encoder, image_path):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        print("No face detected.")
        return

    for i, (x, y, w, h) in enumerate(faces, start=1):
        gender, confidence = predict_face(model, encoder, gray[y : y + h, x : x + w])
        if confidence is None:
            print(f"Face {i}: {gender}")
        else:
            print(f"Face {i}: {gender} ({confidence:.2%} confidence)")


def predict_webcam(model, encoder, camera):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try --camera 1 or check camera permission.")

    print("Webcam prediction started. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        for x, y, w, h in faces:
            gender, confidence = predict_face(model, encoder, gray[y : y + h, x : x + w])
            text = gender if confidence is None else f"{gender} {confidence:.0%}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)

        cv2.imshow("Gender Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    model, encoder = load_model()
    if args.image:
        predict_image(model, encoder, args.image)
    else:
        predict_webcam(model, encoder, args.camera)


if __name__ == "__main__":
    main()
