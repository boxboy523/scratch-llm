"""
Data preprocessing functions for the Korean-only LLM dataset.
All functions are pure and adhere to the project's coding standards.
"""

import re
from config import MIN_LINE_LEN, LINE_SCORE_THRESHOLD


def remove_references(text: str) -> str:
    """
    Removes reference markers like [1], [2], etc.
    Inputs: text (str)
    Outputs: cleaned text (str)
    """
    return re.sub(r"\[\d+\]", " ", text)


def replace_urls(text: str) -> str:
    """
    Replaces URLs with a single space to prevent the model from learning them.
    Inputs: text (str)
    Outputs: cleaned text (str)
    """
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, " ", text)


def strip_delimiters(text: str) -> str:
    """
    Strips TinyStories <|endoftext|> delimiter.
    Inputs: text (str)
    Outputs: cleaned text (str)
    """
    return text.replace("<|endoftext|>", "")


def normalize_whitespace(text: str) -> str:
    """
    Normalizes multiple whitespaces and newlines into single spaces or clean breaks.
    Inputs: text (str)
    Outputs: normalized text (str)
    """
    return re.sub(r"\s+", " ", text).strip()


def get_quality_score(text: str) -> float:
    """
    Calculates a quality score based on Korean characters, numbers, and structure.
    Inputs: text (str)
    Outputs: quality score (float)
    """
    total = len(text)
    if total == 0:
        return 0.0

    # Korean syllables + allowed punctuation ONLY
    lang_chars = sum(
        1 for c in text if ("\uac00" <= c <= "\ud7a3") or (c in ".(),?! '\"")
    )
    nums = sum(1 for c in text if "0" <= c <= "9")
    newlines = text.count("\n")
    sentences = len(re.findall(r"[\uac00-\ud7a3]\s*\.", text))

    lang_ratio = lang_chars / total
    num_ratio = nums / total
    newline_ratio = newlines / total
    sentence_density = sentences / (total / 50) if total > 0 else 0

    return lang_ratio - newline_ratio * 3 - num_ratio - max(0, 0.3 - sentence_density) * 0.5


def filter_lines(text: str) -> str:
    """
    Filters lines based on length and quality score.
    Inputs: text (str)
    Outputs: filtered text (str)
    """
    lines = text.split("\n")
    valid_lines = []

    for line in lines:
        clean_line = line.strip()
        if len(clean_line) < MIN_LINE_LEN:
            continue
        if get_quality_score(clean_line) < LINE_SCORE_THRESHOLD:
            continue
        valid_lines.append(clean_line)

    return "\n".join(valid_lines)


def clean_text(text: str) -> str:
    """
    Composes pure functions to clean and filter text.
    Inputs: text (str)
    Outputs: fully cleaned text (str)
    """
    text = remove_references(text)
    text = replace_urls(text)
    text = strip_delimiters(text)
    text = filter_lines(text)
    text = normalize_whitespace(text)
    return text
