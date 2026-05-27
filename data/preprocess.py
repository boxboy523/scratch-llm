"""
Data preprocessing functions for the Korean & English LLM dataset.
"""

import re


def get_quality_score(text: str) -> float:
    """
    Calculates a quality score for the given text based on multiple factors.
    Supports both Korean and English characters.
    """
    total = len(text)
    if total == 0:
        return 0.0
        
    # Language characters (Korean + English Alphabets + common punctuation)
    lang_chars = sum(1 for c in text if 
                     ('\uac00' <= c <= '\ud7a3') or 
                     ('a' <= c.lower() <= 'z') or
                     (c in ['.', '(', ')', ',', '?', '!', ' ', '\'', '\"']))
    
    nums = sum(1 for c in text if '0' <= c <= '9')
    newlines = text.count('\n')
    # Count sentences: Language character followed by a period
    sentences = len(re.findall(r'[a-zA-Z\uac00-\ud7a3]\s*\.', text))
    
    lang_ratio = lang_chars / total
    num_ratio = nums / total
    newline_ratio = newlines / total
    # Density of sentences per 50 characters
    sentence_density = sentences / (total / 50) if total > 0 else 0
    
    # Calculate final score (based on user logic)
    score = lang_ratio - newline_ratio * 3 - num_ratio - max(0, 0.3 - sentence_density) * 0.5
    return score


def is_quality_text(text: str, min_score: float = 0.6, min_length: int = 50) -> bool:
    """
    Checks if the text meets the quality criteria based on the quality score.
    """
    if len(text) < min_length:
        return False
        
    score = get_quality_score(text)
    return score >= min_score


def clean_wikipedia_text(text: str) -> str:
    """
    Cleans Wikipedia-specific noise and normalizes whitespace.
    """
    # Remove references like [1], [2], etc.
    text = re.sub(r"\[\d+\]", " ", text)
    
    # Normalize multiple whitespaces to a single space
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()
