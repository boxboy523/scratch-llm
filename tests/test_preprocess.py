"""
Unit tests for the data preprocessing module.
"""

import pytest
from data.preprocess import get_quality_score, clean_text


def test_get_quality_score():
    """Tests the quality scoring logic for Korean text."""
    # High quality Korean text
    good_text = "안녕하세요. 오늘 날씨가 참 좋네요. 만나서 반갑습니다."
    score = get_quality_score(good_text)
    assert score > 0.5
    
    # Text with too many numbers
    bad_text = "123 456 789 101112 131415"
    assert get_quality_score(bad_text) < 0.2
    
    # Empty text
    assert get_quality_score("") == 0.0


def test_clean_text():
    """Tests the unified text cleaning logic."""
    raw_text = "위키백과[1]는 https://ko.wikipedia.org/ 에서 제공하는 자유로운 백과사전입니다. <|endoftext|>"
    cleaned = clean_text(raw_text)
    
    # Check references removed
    assert "[1]" not in cleaned
    # Check URLs replaced with space
    assert "https" not in cleaned
    # Check delimiters stripped
    assert "<|endoftext|>" not in cleaned
    # Check normalization
    assert "  " not in cleaned
    assert "자유로운 백과사전입니다." in cleaned
