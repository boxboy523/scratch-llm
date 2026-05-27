"""
Dataset loading and tokenization utilities with multi-source support.
"""

import os
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from data.preprocess import is_quality_text, clean_wikipedia_text
import config


def load_multi_source_dataset(streaming=True):
    """Loads and interleaves Wiki (KO/EN) and NamuWiki datasets."""
    ko_wiki = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split="train", streaming=streaming)
    en_wiki = load_dataset(config.DATASET_NAME, config.EN_DATASET_CONFIG, split="train", streaming=streaming)
    namu = load_dataset(config.NAMUWIKI_DATASET, split="train", streaming=streaming)
    
    datasets = [ko_wiki, en_wiki, namu]
    probs = [config.KO_WIKI_RATIO, config.EN_WIKI_RATIO, config.NAMU_RATIO]
    
    return interleave_datasets(datasets, probabilities=probs, seed=42)


def train_bpe_tokenizer(
    vocab_size: int, 
    save_path: str
) -> PreTrainedTokenizerFast:
    """
    Trains a standard Byte-level BPE tokenizer on multi-source dataset.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    
    dataset = load_multi_source_dataset(streaming=True)
    
    def batch_iterator(batch_size=1000):
        batch = []
        for i, item in enumerate(dataset):
            text = item["text"]
            if is_quality_text(text, min_score=config.MIN_QUALITY_SCORE):
                batch.append(clean_wikipedia_text(text))
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if i > 100_000:  # Increased limit for better BPE on diverse data
                break
        if batch:
            yield batch

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )
    fast_tokenizer.save_pretrained(save_path)
    
    return fast_tokenizer


class KoIterableDataset(IterableDataset):
    """
    Iterable dataset for multi-source text, streaming from HuggingFace.
    """

    def __init__(self, tokenizer, context_len=1024):
        self.dataset = load_multi_source_dataset(streaming=True)
        self.tokenizer = tokenizer
        self.context_len = context_len

    def __iter__(self):
        buffer = []
        count = 0
        for item in self.dataset:
            text = item["text"]
            count += 1
            if count % 10 == 0:
                print(".", end="", flush=True)  # Progress indicator
            
            if not is_quality_text(text, min_score=config.MIN_QUALITY_SCORE):
                continue
            
            cleaned_text = clean_wikipedia_text(text)
            tokens = self.tokenizer.encode(cleaned_text)
            buffer.extend(tokens)
            
            while len(buffer) >= self.context_len:
                chunk = buffer[:self.context_len]
                buffer = buffer[self.context_len:]
                yield torch.tensor(chunk, dtype=torch.long)


def get_tokenizer(save_path: str) -> PreTrainedTokenizerFast:
    """Loads the tokenizer from the saved path."""
    if os.path.exists(os.path.join(save_path, "tokenizer.json")):
        return PreTrainedTokenizerFast.from_pretrained(save_path)
    return None
