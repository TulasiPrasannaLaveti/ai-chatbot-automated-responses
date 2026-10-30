"""
Chatbot wrapper and command-line interface.

Usage:
    python app.py            (from the project root)
    python -m src.chatbot    (equivalent)

Type /quit or /exit to end the chat.
"""

from pathlib import Path

from src.model import IntentClassifier, load_intents, train_and_save

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INTENTS_PATH = PROJECT_ROOT / "data" / "intents.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "chatbot_model.pkl"


class Chatbot:
    """A trained chatbot that can be used from code or from the CLI."""

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_intents(cls, intents_path=DEFAULT_INTENTS_PATH):
        """Train a fresh model in memory from an intents JSON file."""
        model = IntentClassifier()
        model.train_from_intents(load_intents(intents_path))
        return cls(model)

    @classmethod
    def from_saved_model(cls, model_path=DEFAULT_MODEL_PATH):
        return cls(IntentClassifier.load(model_path))

    def reply(self, message):
        """Return just the response text for a user message."""
        response, _tag, _confidence = self.model.get_response(message)
        return response

    def reply_with_details(self, message):
        """Return (response, predicted_intent, confidence)."""
        return self.model.get_response(message)


def get_or_train_model():
    """Load the saved model if it exists, otherwise train and save one."""
    if DEFAULT_MODEL_PATH.exists():
        try:
            return IntentClassifier.load(DEFAULT_MODEL_PATH)
        except Exception:
            # Corrupt / incompatible saved model: fall through and retrain.
            pass
    return train_and_save(DEFAULT_INTENTS_PATH, DEFAULT_MODEL_PATH)


def main():
    bot = Chatbot(get_or_train_model())
    print("CampusBot: Hello! Ask me about the college, courses, placements,")
    print("           fees, library or hostel. Type /quit to exit.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCampusBot: Goodbye!")
            break
        if not message:
            continue
        if message.lower() in ("/quit", "/exit", "quit", "exit"):
            print("CampusBot: Goodbye! Have a great day.")
            break
        response, tag, confidence = bot.reply_with_details(message)
        print(f"CampusBot: {response}  [intent: {tag}, confidence: {confidence:.2f}]")


if __name__ == "__main__":
    main()
