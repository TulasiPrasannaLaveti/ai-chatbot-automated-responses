"""
Intent classification model.

Approach
--------
1. Read patterns and responses from data/intents.json.
2. Turn each pattern into TF-IDF features, using the custom preprocessing
   pipeline in src/preprocess.py as the vectorizer's analyzer.
3. Train a Logistic Regression classifier on (TF-IDF features -> intent).
4. At prediction time, look at the classifier's probability for its best
   intent. If that confidence is below a threshold, return the "fallback"
   intent instead of guessing - this is what keeps the bot honest on
   questions it was never trained on.
"""

import json
import pickle
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocess import preprocess_to_string

FALLBACK_TAG = "fallback"
DEFAULT_THRESHOLD = 0.20


def load_intents(path):
    """Load an intents JSON file and return the list of intent dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def dataset_from_intents(intents):
    """Flatten intents into (patterns, labels, responses_by_tag)."""
    patterns, labels = [], []
    responses = {}
    for intent in intents:
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            labels.append(tag)
    return patterns, labels, responses


class IntentClassifier:
    """TF-IDF + Logistic Regression intent classifier."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess_to_string,  # our own tokenizer / cleaner
            token_pattern=r"[^ ]+",             # tokens are already cleaned
            ngram_range=(1, 2),       # single words + word pairs
            lowercase=False,          # preprocessing already lowercases
        )
        self.classifier = LogisticRegression(max_iter=1000)
        self.responses = {}

    # ------------------------------------------------------------------
    def fit(self, patterns, labels, responses=None):
        """Train on raw text patterns. Returns self (for chaining)."""
        features = self.vectorizer.fit_transform(patterns)
        self.classifier.fit(features, labels)
        if responses is not None:
            self.responses = responses
        return self

    def train_from_intents(self, intents):
        patterns, labels, responses = dataset_from_intents(intents)
        return self.fit(patterns, labels, responses)

    # ------------------------------------------------------------------
    def predict_intent(self, text):
        """
        Return (intent_tag, confidence) for a user message.

        If the model's best probability is below the threshold, the
        fallback intent is returned instead.
        """
        features = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(features)[0]
        best_index = int(probabilities.argmax())
        best_tag = self.classifier.classes_[best_index]
        confidence = float(probabilities[best_index])

        if confidence < self.threshold:
            return FALLBACK_TAG, confidence
        return best_tag, confidence

    def get_response(self, text):
        """Return (response_text, intent_tag, confidence) for a message."""
        tag, confidence = self.predict_intent(text)
        options = self.responses.get(tag) or self.responses.get(FALLBACK_TAG, [])
        if not options:
            return "Sorry, I did not understand that.", FALLBACK_TAG, confidence
        return random.choice(options), tag, confidence

    # ------------------------------------------------------------------
    def save(self, path):
        """Save the trained model, responses and threshold with pickle."""
        payload = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "responses": self.responses,
            "threshold": self.threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        """Load a model previously written by save()."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model = cls(threshold=payload["threshold"])
        model.vectorizer = payload["vectorizer"]
        model.classifier = payload["classifier"]
        model.responses = payload["responses"]
        return model


def train_and_save(intents_path, model_path):
    """Convenience helper: train from an intents file and save the model."""
    intents = load_intents(intents_path)
    model = IntentClassifier()
    model.train_from_intents(intents)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model
