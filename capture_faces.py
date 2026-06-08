import argparse
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"


def parse_args():
    parser = argparse.ArgumentParser(description="Capture face images for gender training.")
    parser.add_argument("--gender", choices=["male", "female"], required=True)
    parser.add_argument("--name", required=True, help="Person/student folder name, example: student1")
    parser.add_argument("--count", type=int, default=50, help="Number of face images to save")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = DATASET_DIR / args.gender / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Could not load Haar cascade for face detection.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try --camera 1 or check camera permission.")

    saved = 0
    print(f"Capturing {args.count} face images for {args.gender}/{args.name}")
    print("Press 'q' to stop early.")

    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        for x, y, w, h in faces:
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (100, 100))
            out_path = save_dir / f"{args.name}_{saved + 1:03d}.jpg"
            cv2.imwrite(str(out_path), face)
            saved += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 0), 2)
            cv2.putText(frame, f"Saved {saved}/{args.count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)
            break

        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} images in {save_dir}")


if __name__ == "__main__":
    main()
