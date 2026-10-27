
"""
Hate Speech Detection using Machine Learning
Author: Sai Kumar Kanike

Classifies text into:
0 - Hate Speech
1 - Offensive Language
2 - Neither
"""

from pathlib import Path
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = Path("hate_speech.csv")
MODEL_FILE = Path("hate_speech_model.joblib")

CLASS_NAMES = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}


def prepare_text(text):
    """Normalize text before converting it into machine features."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(file_path):
    """Read the CSV and validate its required columns."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path.resolve()}\n"
            "Place hate_speech.csv in the project folder."
        )

    data = pd.read_csv(file_path)

    required = {"tweet", "class"}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            f"Available columns: {list(data.columns)}"
        )

    data = data[["tweet", "class"]].dropna().copy()
    data["class"] = pd.to_numeric(data["class"], errors="coerce")
    data = data.dropna(subset=["class"])

    data["class"] = data["class"].astype(int)
    data = data[data["class"].isin(CLASS_NAMES)]

    data["processed_text"] = data["tweet"].apply(prepare_text)
    data = data[data["processed_text"] != ""]

    if data["class"].nunique() < 2:
        raise ValueError("The dataset must contain at least two classes.")

    return data


def create_classifier():
    """Create a complete text-processing and classification pipeline."""
    return Pipeline([
        (
            "text_features",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=40000,
                sublinear_tf=True
            )
        ),
        (
            "text_classifier",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=42
            )
        )
    ])


def evaluate_classifier(classifier, test_text, test_labels):
    """Report model performance on held-out data."""
    predictions = classifier.predict(test_text)

    print("\nMODEL EVALUATION")
    print("-" * 45)
    print(f"Accuracy: {accuracy_score(test_labels, predictions):.4f}")

    print("\nClassification report:")
    print(classification_report(
        test_labels,
        predictions,
        labels=list(CLASS_NAMES.keys()),
        target_names=list(CLASS_NAMES.values()),
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        test_labels,
        predictions,
        labels=list(CLASS_NAMES.keys())
    ))


def predict_message(classifier):
    """Allow a user to test individual messages."""
    print("\nEnter text for classification.")
    print("Type 'quit' to exit.")

    while True:
        message = input("\nText: ").strip()

        if message.lower() == "quit":
            break

        cleaned_message = prepare_text(message)

        if not cleaned_message:
            print("Please enter text containing letters.")
            continue

        predicted_class = int(classifier.predict([cleaned_message])[0])
        probabilities = classifier.predict_proba([cleaned_message])[0]
        confidence = max(probabilities) * 100

        print("Predicted category:", CLASS_NAMES[predicted_class])
        print(f"Model confidence estimate: {confidence:.1f}%")


def main():
    print("HATE SPEECH DETECTION")
    print("=" * 45)

    dataset = load_dataset(DATA_FILE)

    print(f"Usable records: {len(dataset)}")
    print("\nExamples per class:")
    print(dataset["class"].value_counts().sort_index())

    features = dataset["processed_text"]
    targets = dataset["class"]

    train_text, test_text, train_labels, test_labels = (
        train_test_split(
            features,
            targets,
            test_size=0.20,
            random_state=42,
            stratify=targets
        )
    )

    classifier = create_classifier()

    print("\nTraining model...")
    classifier.fit(train_text, train_labels)

    evaluate_classifier(classifier, test_text, test_labels)

    joblib.dump(classifier, MODEL_FILE)
    print(f"\nSaved trained model to: {MODEL_FILE}")

    predict_message(classifier)


if __name__ == "__main__":
    main()
