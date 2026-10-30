"""
Tests for the chatbot - plain asserts, no pytest needed.

Run from the project root:

    python tests/test_chatbot.py

(The functions are also named test_* so pytest can run them if installed.)
"""

import sys
from pathlib import Path

# Make the project root importable no matter where this is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import IntentClassifier, load_intents          # noqa: E402
from src.preprocess import preprocess, stem, tokenize         # noqa: E402

INTENTS_PATH = PROJECT_ROOT / "data" / "intents.json"


def build_model():
    model = IntentClassifier()
    model.train_from_intents(load_intents(INTENTS_PATH))
    return model


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------

def test_tokenize_lowercases_and_removes_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("What is Placements?") == ["what", "is", "placements"]


def test_preprocess_removes_stop_words():
    tokens = preprocess("What is your name?")
    assert "what" not in tokens
    assert "is" not in tokens
    assert "your" not in tokens
    assert "name" in tokens


def test_stemming_is_applied():
    assert stem("placements") == "placement"
    assert "placement" in preprocess("Tell me about the placements")


def test_preprocess_empty_input():
    assert preprocess("") == []
    assert preprocess("?!...") == []


# ----------------------------------------------------------------------
# Intent prediction (after training)
# ----------------------------------------------------------------------

def test_intent_prediction():
    model = build_model()
    assert model.predict_intent("Hello there")[0] == "greeting"
    assert model.predict_intent("Goodbye, see you later")[0] == "goodbye"
    assert model.predict_intent("What courses do you offer?")[0] == "courses"
    assert model.predict_intent("Tell me about placements")[0] == "placements"
    assert model.predict_intent("Thank you so much")[0] == "thanks"


def test_fallback_for_unknown_input():
    model = build_model()
    tag, confidence = model.predict_intent("asdf qwer zxcv plok")
    assert tag == "fallback"
    assert confidence < model.threshold or tag == "fallback"


def test_response_is_returned():
    model = build_model()
    response, tag, confidence = model.get_response("Hi")
    assert isinstance(response, str) and len(response) > 0
    assert tag == "greeting"
    assert 0.0 <= confidence <= 1.0


def test_save_and_load_round_trip(tmp_path=None):
    import tempfile

    model = build_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pkl"
        model.save(path)
        loaded = IntentClassifier.load(path)
    assert loaded.predict_intent("Hello")[0] == "greeting"


# ----------------------------------------------------------------------

ALL_TESTS = [
    test_tokenize_lowercases_and_removes_punctuation,
    test_preprocess_removes_stop_words,
    test_stemming_is_applied,
    test_preprocess_empty_input,
    test_intent_prediction,
    test_fallback_for_unknown_input,
    test_response_is_returned,
    test_save_and_load_round_trip,
]

if __name__ == "__main__":
    for test in ALL_TESTS:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\nAll {len(ALL_TESTS)} tests passed.")
