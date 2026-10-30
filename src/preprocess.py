"""
Text preprocessing pipeline for the chatbot.

This module is intentionally written from scratch (no library tokenizer),
because it is the part interviewers most often ask about:

    raw text
      -> lowercase
      -> tokenize (split into words)
      -> remove punctuation
      -> remove stop words ("is", "the", "what", ...)
      -> simple suffix stemming ("placements" -> "placement")

The exact same functions are used for training data AND for live user
input, so the model always sees text in the same form.
"""

import re

# Common English words that carry very little meaning for intent matching.
# Keeping this list small and explicit makes the behaviour easy to explain.
STOP_WORDS = {
    "a", "an", "the",
    "is", "are", "was", "were", "am", "be", "been",
    "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "my", "your", "his", "her", "our", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "of", "in", "on", "at", "to", "for", "with", "by", "about", "from",
    "and", "or", "but", "if", "then", "so", "as",
    "please", "tell",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase `text` and split it into word tokens (punctuation removed)."""
    if not isinstance(text, str):
        text = str(text)
    return _TOKEN_RE.findall(text.lower())


def stem(word):
    """
    Very simple suffix stemming (not a full Porter stemmer).

    It only removes common English endings so that related forms map to
    the same token, e.g. "placements" -> "placement" and
    "studying" -> "study". The result does not have to be a real English
    word ("courses" -> "cours"); it only has to be consistent, because
    training text and user input go through the same function.
    Rules are applied conservatively and only to longer words.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"          # "facilities" -> "facility"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]                # "studying" -> "study"
    if word.endswith("edly") and len(word) > 6:
        return word[:-4]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]                # "offered" -> "offer"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]                # "courses" -> "cours" (consistent)
    if word.endswith("s") and len(word) > 3:
        return word[:-1]                # "placements" -> "placement"
    return word


def preprocess(text):
    """
    Full pipeline: returns a list of cleaned tokens for `text`.

    This function is passed to scikit-learn's TfidfVectorizer as its
    `analyzer`, so training and prediction share one code path.
    """
    tokens = tokenize(text)
    cleaned = [stem(tok) for tok in tokens if tok not in STOP_WORDS]
    return cleaned


def preprocess_to_string(text):
    """Same as preprocess(), but returns a single space-joined string."""
    return " ".join(preprocess(text))
