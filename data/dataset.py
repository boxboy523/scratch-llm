"""
Dataset loading and processing for the Korean-only LLM.
Implements dynamic quality filtering and multi-source interleaving.
"""

import os
import torch
import numpy as np
from collections import deque
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from data.preprocess import get_quality_score, clean_text
import config


def compute_source_threshold(dataset, sample_size: int, percentile: int) -> float:
    """
    Computes a quality threshold based on a sample of the dataset.
    Inputs: dataset, sample_size (int), percentile (int)
    Outputs: threshold (float)
    """
    scores = []
    # Use a small sample to estimate the distribution
    sample = dataset.shuffle(seed=config.SEED).select(range(min(sample_size, len(dataset))))
    for item in sample:
        scores.append(get_quality_score(item["text"]))
    return float(np.percentile(scores, percentile))


def get_filtered_culturax():
    """
    Loads and dedups CulturaX against Wikipedia and NamuWiki.
    Outputs: filtered dataset
    """
    ds = load_dataset(config.CULTURAX_DATASET, "ko", split="train", streaming=False)
    # Filter out Wikipedia and NamuWiki duplicates
    ds = ds.filter(
        lambda x: "ko.wikipedia.org" not in x["url"] and "namu.wiki" not in x["url"]
    )
    return ds.select_columns(["text"])


def load_multi_source_dataset():
    """
    Loads, filters, and interleaves Wikipedia, NamuWiki, CulturaX, and TinyStories.
    Outputs: interleaved and shuffled dataset
    """
    wiki = load_dataset(config.WIKI_DATASET, config.WIKI_KO, split="train").select_columns(["text"])
    namu = load_dataset(config.NAMUWIKI_DATASET, split="train").select_columns(["text"])
    culturax = get_filtered_culturax()
    stories = load_dataset(config.TINYSTORIES_DATASET, split="train").select_columns(["text"])

    # Compute dynamic thresholds (excluding TinyStories as it is synthetic/clean)
    wiki_thr = compute_source_threshold(wiki, config.SAMPLE_SIZE, config.CUT_PERCENTILE)
    namu_thr = compute_source_threshold(namu, config.SAMPLE_SIZE, config.CUT_PERCENTILE)
    cult_thr = compute_source_threshold(culturax, config.SAMPLE_SIZE, config.CUT_PERCENTILE)

    # Apply quality filtering
    wiki = wiki.filter(lambda x: get_quality_score(x["text"]) >= wiki_thr)
    namu = namu.filter(lambda x: get_quality_score(x["text"]) >= namu_thr)
    culturax = culturax.filter(lambda x: get_quality_score(x["text"]) >= cult_thr)

    datasets = [wiki, namu, culturax, stories]
    probs = [config.KO_WIKI_RATIO, config.NAMU_RATIO, config.CULTURAX_RATIO, config.TINYSTORIES_RATIO]

    interleaved = interleave_datasets(datasets, probabilities=probs, seed=config.SEED)
    return interleaved.shuffle(seed=config.SEED, buffer_size=config.SHUFFLE_BUFFER)


def train_bpe_tokenizer(vocab_size: int, save_path: str) -> PreTrainedTokenizerFast:
    """Trains a BPE tokenizer using the combined dataset."""
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<|endoftext|>"]
    )

    dataset = load_multi_source_dataset()

    def batch_iterator(batch_size=1000):
        batch = []
        for i, item in enumerate(dataset):
            text = clean_text(item["text"])
            if len(text) >= config.DOC_MIN_LENGTH:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if i > 100_000: break
        if batch: yield batch

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
        pad_token="[PAD]", mask_token="[MASK]", eos_token="<|endoftext|>"
    )
    fast_tokenizer.save_pretrained(save_path)
    return fast_tokenizer


class KoIterableDataset(IterableDataset):
    """
    Memory-efficient token-packing dataset.
    """
    def __init__(self, dataset, tokenizer, context_len: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.eos_id = tokenizer.eos_token_id

    def __iter__(self):
        buffer = deque()
        for item in self.dataset:
            cleaned = clean_text(item["text"])
            if len(cleaned) < config.DOC_MIN_LENGTH:
                continue
            
            tokens = self.tokenizer.encode(cleaned)
            tokens.append(self.eos_id) # Mark doc boundary
            buffer.extend(tokens)

            while len(buffer) >= self.context_len:
                chunk = [buffer.popleft() for _ in range(self.context_len)]
                yield torch.tensor(chunk, dtype=torch.long)


def get_tokenizer(save_path: str) -> PreTrainedTokenizerFast:
    """Loads a saved tokenizer."""
    if os.path.exists(os.path.join(save_path, "tokenizer.json")):
        return PreTrainedTokenizerFast.from_pretrained(save_path)
    return None
