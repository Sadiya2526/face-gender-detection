from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
IMAGE_SIZE = (100, 100)


def iter_images():
    for gender_dir in DATASET_DIR.iterdir():
        if not gender_dir.is_dir():
            continue
        gender = gender_dir.name.lower()
        if gender not in {"male", "female"}:
            continue
        for image_path in gender_dir.rglob("*"):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                yield image_path, gender


def load_dataset():
    x_values = []
    y_values = []

    for image_path, gender in iter_images():
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        image = cv2.resize(image, IMAGE_SIZE)
        x_values.append(image.flatten() / 255.0)
        y_values.append(gender)

    return np.array(x_values, dtype=np.float32), np.array(y_values)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    x_values, y_values = load_dataset()

    if len(x_values) < 2:
        print("No training output created.")
        print("Reason: dataset has fewer than 2 face images.")
        print("First capture images, for example:")
        print("  python capture_faces.py --gender male --name student1 --count 50")
        print("  python capture_faces.py --gender female --name student3 --count 50")
        return

    if len(set(y_values)) < 2:
        print("No training output created.")
        print("Reason: dataset needs both male and female images.")
        return

    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y_values)

    test_size = 0.25 if len(x_values) >= 8 else 0.5
    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        encoded_y,
        test_size=test_size,
        random_state=42,
        stratify=encoded_y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODELS_DIR / "gender_model.pkl")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.pkl")

    print("Training completed.")
    print(f"Images used: {len(x_values)}")
    print(f"Classes: {', '.join(encoder.classes_)}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, target_names=encoder.classes_, zero_division=0))
    print(f"Saved model: {MODELS_DIR / 'gender_model.pkl'}")
    print(f"Saved labels: {MODELS_DIR / 'label_encoder.pkl'}")


if __name__ == "__main__":
    main()
