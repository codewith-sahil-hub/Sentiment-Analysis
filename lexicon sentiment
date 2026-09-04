"""
Lexicon-Based Sentiment & Emotion Engine
============================================
A transparent, from-scratch implementation of the same core technique
VADER uses: a word-polarity lexicon combined with negation handling
and intensifier scaling. Built offline (no NLTK/VADER downloads needed)
so it's fully self-contained and inspectable.

This is intentionally simple and rule-based -- good for demonstrating
*how* lexicon-based NLP sentiment scoring works, not a production-grade
model. For production use, a trained model (or the real VADER/NLTK
lexicon) would generalize further.
"""

import re

# ----------------------------------------------------------------------
# LEXICONS
# ----------------------------------------------------------------------
POSITIVE_WORDS = {
    "excellent": 3.0, "amazing": 3.0, "perfect": 3.0, "love": 2.8,
    "fantastic": 2.8, "wonderful": 2.7, "best": 2.6, "great": 2.4,
    "awesome": 2.6, "outstanding": 2.7, "impressed": 2.2, "happy": 2.0,
    "good": 1.8, "nice": 1.6, "solid": 1.5, "reliable": 1.6,
    "recommend": 1.8, "satisfied": 1.8, "worth": 1.4, "comfortable": 1.5,
    "fast": 1.2, "easy": 1.3, "quality": 1.3, "beautiful": 2.0,
    "smooth": 1.3, "sturdy": 1.4, "helpful": 1.5, "pleased": 1.8,
}

NEGATIVE_WORDS = {
    "terrible": -3.0, "awful": -3.0, "horrible": -3.0, "worst": -2.9,
    "hate": -2.8, "useless": -2.6, "broken": -2.4, "disappointing": -2.4,
    "disappointed": -2.3, "poor": -2.0, "bad": -1.9, "cheap": -1.4,
    "waste": -2.5, "slow": -1.4, "annoying": -1.8, "flimsy": -1.8,
    "defective": -2.5, "junk": -2.6, "regret": -2.2, "avoid": -2.2,
    "uncomfortable": -1.6, "overpriced": -1.7, "faulty": -2.3,
    "frustrating": -2.0, "unreliable": -2.1, "difficult": -1.3,
}

NEGATIONS = {"not", "no", "never", "none", "hardly", "barely", "n't", "without"}
INTENSIFIERS = {
    "very": 1.4, "extremely": 1.6, "really": 1.3, "so": 1.3,
    "incredibly": 1.5, "totally": 1.3, "absolutely": 1.5, "quite": 1.15,
}
DAMPENERS = {"somewhat": 0.7, "slightly": 0.6, "barely": 0.5, "kind of": 0.7}

EMOTION_LEXICON = {
    "joy": {"love", "happy", "excited", "delighted", "pleased", "glad", "amazing"},
    "anger": {"angry", "furious", "annoyed", "frustrating", "hate", "outraged"},
    "sadness": {"disappointed", "sad", "regret", "unhappy", "letdown"},
    "trust": {"reliable", "recommend", "sturdy", "quality", "consistent"},
    "surprise": {"surprised", "unexpected", "shocked", "wow"},
    "fear": {"worried", "concerned", "risky", "unsafe"},
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


def score_text(text: str) -> dict:
    """
    Returns a dict with: compound score (-1 to 1), label, and per-word breakdown.
    Mirrors VADER's approach: lexicon lookup -> apply preceding intensifier/
    negation modifiers within a small window -> aggregate -> normalize.
    """
    tokens = tokenize(text)
    raw_scores = []

    for i, word in enumerate(tokens):
        base = POSITIVE_WORDS.get(word, NEGATIVE_WORDS.get(word, 0.0))
        if base == 0.0:
            continue

        # Look at up to 2 preceding words for negation/intensifiers
        window = tokens[max(0, i - 2):i]
        negated = any(w in NEGATIONS or w.endswith("n't") for w in window)
        intensity = 1.0
        for w in window:
            if w in INTENSIFIERS:
                intensity *= INTENSIFIERS[w]
            elif w in DAMPENERS:
                intensity *= DAMPENERS[w]

        adjusted = base * intensity
        if negated:
            adjusted *= -0.85  # negation flips and slightly dampens

        raw_scores.append(adjusted)

    if not raw_scores:
        compound = 0.0
    else:
        total = sum(raw_scores)
        # Normalize similarly to VADER's compound score formula
        compound = total / ((total ** 2 + 15) ** 0.5)
        compound = max(-1.0, min(1.0, compound))

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {"compound": round(compound, 3), "label": label}


def detect_emotions(text: str) -> list[str]:
    tokens = set(tokenize(text))
    found = []
    for emotion, words in EMOTION_LEXICON.items():
        if tokens & words:
            found.append(emotion)
    return found or ["none detected"]
