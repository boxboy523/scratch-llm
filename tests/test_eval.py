"""
Unit tests for the model evaluation and generation module.
"""

import pytest
import torch
from unittest.mock import MagicMock
from eval_model import generate


def test_generate_basic():
    """Verifies that the generate function produces output and respects max_new_tokens."""
    mock_model = MagicMock()
    # Mock logits: (batch=1, seq, vocab=100)
    # We'll just return zeros, argmax will be 0
    mock_model.return_value = (torch.zeros(1, 5, 100), None)
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "generated text"
    mock_tokenizer.eos_token_id = 99
    
    device = torch.device("cpu")
    
    output = generate(mock_model, mock_tokenizer, "prompt", max_new_tokens=2, device=device)
    
    assert output == "generated text"
    assert mock_model.call_count == 2 # Called for each new token


def test_repetition_penalty():
    """Checks if the repetition penalty logic executes without error."""
    # This is hard to verify numerically without a real model, 
    # but we can ensure it runs through the loop.
    mock_model = MagicMock()
    mock_model.return_value = (torch.ones(1, 1, 100), None)
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1]
    mock_tokenizer.decode.side_effect = lambda x: str(x)
    mock_tokenizer.eos_token_id = 99
    
    device = torch.device("cpu")
    
    # Generate 5 tokens, force same token to trigger penalty logic
    output = generate(mock_model, mock_tokenizer, "prompt", max_new_tokens=5, device=device)
    
    assert output is not None
