# AI Chatbot for Automated Responses

An NLP-based conversational assistant built in Python. It understands a
user's question, identifies the **intent** behind it (for example
`courses`, `fees` or `placements`), and replies with a suitable automated
response in real time — a small college-enquiry chatbot that runs entirely
in the terminal, with no external API.

## Features

- Real-time command-line chat (`/quit` or `/exit` to end)
- Custom text preprocessing pipeline written from scratch:
  tokenization, lowercasing, punctuation removal, stop-word removal and
  simple suffix stemming
- Intent recognition with TF-IDF features + Logistic Regression
  (scikit-learn)
- Confidence threshold: low-confidence questions get an honest
  **fallback** reply instead of a wrong answer
- 11 intents with multiple patterns and responses in a plain JSON file —
  easy to extend without touching code
- Train / test evaluation script that prints real accuracy and a
  per-intent classification report
- Model save / load with `pickle`, so training does not repeat every run
- Plain-Python tests (no extra test framework required)

## Tech Stack

| Part | Technology |
| --- | --- |
| Language | Python 3 |
| ML / NLP | scikit-learn (`TfidfVectorizer`, `LogisticRegression`) |
| Preprocessing | Custom code (`src/preprocess.py`) — regex tokenization, stop-word list, suffix stemming |
| Data | JSON (`data/intents.json`) |
| Model persistence | `pickle` |
| Interface | Command-line (CLI) |

## Project Structure

```
ai-chatbot-automated-responses/
├── app.py                  # Entry point - run this to chat
├── evaluate.py             # Train/test split evaluation, prints accuracy
├── requirements.txt
├── README.md
├── data/
│   └── intents.json        # Intents: patterns (training examples) + responses
├── models/
│   └── .gitkeep            # Trained model (.pkl) is saved here at runtime
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Tokenization, stop-word removal, stemming
│   ├── model.py            # TF-IDF + Logistic Regression, train/save/load/predict
│   └── chatbot.py          # Chatbot wrapper and CLI loop
└── tests/
    └── test_chatbot.py     # Plain-assert tests, runnable with python
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/TulasiPrasannaLaveti/ai-chatbot-automated-responses.git
cd ai-chatbot-automated-responses
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
# Windows:        venv\Scripts\activate
# macOS / Linux:  source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the chatbot

```bash
python app.py
```

On the first run the model is trained from `data/intents.json` and saved
to `models/chatbot_model.pkl`; later runs load the saved model. Delete
that file if you edit the intents and want to retrain.

### 5. (Optional) Evaluate the model

```bash
python evaluate.py
```

### 6. (Optional) Run the tests

```bash
python tests/test_chatbot.py
```

## Sample Conversation

```
CampusBot: Hello! Ask me about the college, courses, placements,
           fees, library or hostel. Type /quit to exit.
You: Hi
CampusBot: Hello! How can I help you today?  [intent: greeting, confidence: ...]
You: What courses do you offer?
CampusBot: The college offers B.Tech programs including Computer Science
           and Engineering, Artificial Intelligence and Machine Learning,
           ...  [intent: courses, confidence: ...]
You: Tell me about placements
CampusBot: The college has a dedicated placement cell that conducts
           aptitude, coding and soft-skills training, mock interviews
           ...  [intent: placements, confidence: ...]
You: What is the weather today?
CampusBot: Sorry, I did not understand that. I can help with questions
           about the college, courses, placements, fees, library and
           hostel.  [intent: fallback, confidence: ...]
You: Thank you
CampusBot: You are welcome! Happy to help.  [intent: thanks, confidence: ...]
You: /quit
CampusBot: Goodbye! Have a great day.
```

*(Confidence values are printed live by the program — they depend on the
input — and replies are picked from the responses for the predicted
intent, so exact wording can vary between runs.)*

## How It Works

### 1. Tokenization & preprocessing — `src/preprocess.py`

Raw text is not fed to the model directly. It is cleaned first:

```
"What courses do you offer?"
  -> lowercase:        "what courses do you offer?"
  -> tokenize:         ["what", "courses", "do", "you", "offer"]
  -> remove stop words: ["courses", "offer"]
  -> stemming:          ["cours", "offer"]
```

- **Tokenization** splits a sentence into individual words and drops
  punctuation (a regular expression keeps only letters and digits).
- **Stop words** like "what", "do", "you", "is" and "the" occur in almost
  every question and say nothing about the intent, so they are removed.
- **Stemming** strips common suffixes (`placements` → `placement`) so
  different forms of the same word become one token. The stemmer here is
  a small rule-based one written for this project, not a library stemmer.

The same function is used for training patterns and for live user input,
so both are always cleaned identically.

### 2. TF-IDF features

`TfidfVectorizer` converts the cleaned tokens into numbers. A word gets a
high weight if it appears often in *this* sentence but rarely across all
training patterns — so a distinctive word like "hostel" or "fees" counts
for much more than a common one. Word pairs (bigrams) are included as
well as single words.

### 3. Intent classification

A **Logistic Regression** classifier is trained on the TF-IDF vectors of
all the patterns in `data/intents.json`, labelled with their intent.
For a new message it outputs a probability for every intent, and the
highest-probability intent wins. For example, "How much are the fees for
admission?" is classified as `fees`, and one of the `fees` responses from
the JSON file is returned at random.

### 4. Confidence threshold & fallback

If even the best intent's probability is below the threshold (0.20 by
default), the chatbot does not guess: it replies with a `fallback`
response and says what topics it *can* help with. That is why unrelated
questions get an honest "I did not understand that" instead of a random
college answer.

### 5. Evaluation

`evaluate.py` splits the patterns into a training set and a held-out test
set (stratified, so every intent appears in both), trains only on the
training split, and prints the accuracy and per-intent
precision/recall/F1 on the test split. **No accuracy figure is quoted in
this README** — run `python evaluate.py` to get the real, current metric
for the dataset in this repository.

### Intents covered

`greeting`, `goodbye`, `name`, `college_info`, `courses`, `placements`,
`fees`, `library`, `hostel`, `thanks`, `fallback` — each with its own
example patterns and responses in `data/intents.json`. To add a topic,
add a new intent object to that file and retrain (delete the saved model).

## Future Improvements

- Add many more patterns per intent and evaluate on real user questions
- Replace the rule-based stemmer with a standard stemmer/lemmatizer and
  compare accuracy
- Try other classifiers (SVM, Naive Bayes, or a small neural network)
  and compare them in `evaluate.py`
- Add a simple web UI (e.g. Flask/Streamlit) on top of the same model
- Support context / follow-up questions ("And what about its fees?")
- Store chat logs (with user consent) to find questions the bot misses
  and turn them into new training patterns

## Author

**Tulasi Prasanna**
GitHub: https://github.com/TulasiPrasannaLaveti

*Final-year B.Tech (Artificial Intelligence & Machine Learning) project —
Natural Language Processing, intent classification and automated response
generation.*
