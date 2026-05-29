"""
Unit tests for the dataset loading and processing module.
"""

import pytest
import torch
import os
from unittest.mock import MagicMock, patch
from datasets import Dataset
from data.dataset import compute_source_threshold, group_tinystories, KoIterableDataset


def test_compute_source_threshold():
    """Verifies that the threshold calculation works on a small dataset."""
    # Create a small dummy dataset
    data = {"text": ["안녕", "반가워", "테스트 문장", "매우 긴 문장입니다. " * 10]}
    ds = Dataset.from_dict(data)
    
    # Compute threshold for bottom 50%
    threshold = compute_source_threshold(ds, sample_size=4, ratio=0.5)
    
    assert isinstance(threshold, float)
    assert -5.0 <= threshold <= 1.0 # Reasonable range for quality score


def test_group_tinystories():
    """Verifies that TinyStories rows are grouped correctly until <|endoftext|>."""
    data = [
        {"text": "Once upon a time."},
        {"text": "A little girl played. <|endoftext|>"},
        {"text": "Another story starts."},
        {"text": "It was fun. <|endoftext|>"}
    ]
    
    grouped = list(group_tinystories(data))
    
    assert len(grouped) == 2
    assert "Once upon a time. A little girl played." in grouped[0]["text"]
    assert "Another story starts. It was fun." in grouped[1]["text"]


def test_ko_iterable_dataset():
    """Verifies that KoIterableDataset packs tokens correctly and adds EOS."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [10, 20, 30]
    mock_tokenizer.eos_token_id = 99
    
    raw_dataset = [{"text": "dummy text that is long enough to pass the min length check of fifty chars."}]
    context_len = 2
    
    # context_len=2, tokens=[10, 20, 30, 99] -> should yield two chunks of 2
    iterable_ds = KoIterableDataset(raw_dataset, mock_tokenizer, context_len)
    chunks = list(iterable_ds)
    
    assert len(chunks) == 2
    assert torch.equal(chunks[0], torch.tensor([10, 20], dtype=torch.long))
    assert torch.equal(chunks[1], torch.tensor([30, 99], dtype=torch.long))
