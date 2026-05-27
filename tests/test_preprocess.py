"""
Unit tests for the data preprocessing module with advanced quality scoring.
"""

import pytest
from data.preprocess import get_quality_score, is_quality_text, clean_wikipedia_text


def test_get_quality_score():
    """Tests the advanced quality scoring logic."""
    # High quality Korean text
    good_text = "안녕하세요. 오늘 날씨가 참 좋네요. 만나서 반갑습니다."
    score = get_quality_score(good_text)
    assert score > 0.8
    
    # Text with too many numbers
    bad_text = "123 456 789 101112 131415"
    assert get_quality_score(bad_text) < 0.2
    
    # Empty text
    assert get_quality_score("") == 0.0


def test_is_quality_text():
    """Tests the text quality filtering logic based on score."""
    good_text = "이 문장은 테스트를 위한 한국어 문장입니다. 길이가 충분히 길어야 하며 한글 비율이 높아야 합니다." * 2
    assert is_quality_text(good_text, min_score=0.6) is True
    
    # Short text
    assert is_quality_text("짧은 한글 문장", min_length=50) is False


def test_clean_wikipedia_text():
    """Tests the Wikipedia text cleaning logic."""
    raw_text = "위키백과[1]는 누구나 참여할 수 있는[2] 백과사전입니다."
    cleaned = clean_wikipedia_text(raw_text)
    assert "위키백과 는 누구나 참여할 수 있는 백과사전입니다." in cleaned
    assert "[" not in cleaned
