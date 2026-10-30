"""
Evaluate the intent classifier honestly.

The patterns from data/intents.json are split into a training set and a
held-out test set. The model is trained ONLY on the training split, then
accuracy and a per-intent classification report are printed for the test
split. These are the real numbers for this dataset - they are not written
into the README, run this file to see the current result:

    python evaluate.py
"""

from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.model import IntentClassifier, dataset_from_intents, load_intents

PROJECT_ROOT = Path(__file__).resolve().parent
INTENTS_PATH = PROJECT_ROOT / "data" / "intents.json"


def main():
    intents = load_intents(INTENTS_PATH)
    patterns, labels, _responses = dataset_from_intents(intents)

    X_train, X_test, y_train, y_test = train_test_split(
        patterns,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    print(f"Total patterns : {len(patterns)}")
    print(f"Training split : {len(X_train)}")
    print(f"Test split     : {len(X_test)}")
    print("-" * 60)

    model = IntentClassifier(threshold=0.0)  # 0.0 = report raw predictions
    model.fit(X_train, y_train)

    predictions = [model.predict_intent(text)[0] for text in X_test]

    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print("-" * 60)
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
