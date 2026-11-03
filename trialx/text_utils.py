"""Small text helpers used by dataset building and scoring."""
import re

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def normalize(text):
    """Lowercase, replace punctuation with spaces, and collapse whitespace."""
    lowered = text.lower()
    spaced = _NON_ALPHANUMERIC.sub(" ", lowered)
    return " ".join(spaced.split())


def appears_in(label, text):
    """True if the normalized label appears as a whole-word phrase in the normalized text."""
    normalized_label = normalize(label)
    if not normalized_label:
        return False
    padded_text = " " + normalize(text) + " "
    return (" " + normalized_label + " ") in padded_text
