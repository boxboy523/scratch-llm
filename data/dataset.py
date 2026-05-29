"""
Dataset loading and processing for the Korean-only LLM.
Implements dynamic quality filtering and multi-source interleaving.
"""

import os
import torch
import numpy as np
from collections import deque
from datasets import Dataset, load_dataset, interleave_datasets
from torch.utils.data import IterableDataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from data.preprocess import get_quality_score, clean_text
import config


def compute_source_threshold(dataset, sample_size: int, ratio: float) -> float:
    """
    Computes a quality threshold based on a sample of the dataset.
    Inputs: dataset, sample_size (int), ratio (float, 0.0-1.0)
    Outputs: threshold (float)
    """
    scores = []
    # Use a small sample to estimate the distribution
    sample = dataset.shuffle(seed=config.SEED).select(range(min(sample_size, len(dataset))))
    for item in sample:
        scores.append(get_quality_score(item["text"]))
    return float(np.quantile(scores, ratio))


def compute_all_thresholds(sources: dict, sample_size: int, ratios: dict) -> dict:
    """
    Computes quality thresholds for all provided sources.
    Inputs: sources (dict of datasets), sample_size (int), ratios (dict of float)
    Outputs: thresholds (dict of float)
    """
    thresholds = {}
    for name, ds in sources.items():
        ratio = ratios.get(name, 0.3)
        thresholds[name] = compute_source_threshold(ds, sample_size, ratio)
    return thresholds


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


def group_tinystories(dataset):
    """
    Yields combined text rows as a single document until <|endoftext|> literal is encountered.
    """
    buffer = []
    for item in dataset:
        text = item["text"]
        buffer.append(text)
        if "<|endoftext|>" in text:
            yield {"text": " ".join(buffer)}
            buffer = []
    if buffer:
        yield {"text": " ".join(buffer)}


def load_multi_source_dataset():
    """
    Loads, filters, and interleaves Wikipedia, NamuWiki, CulturaX, and TinyStories.
    Applies source-specific cleaning before interleaving.
    Outputs: interleaved and shuffled dataset
    """
    wiki = load_dataset(config.WIKI_DATASET, config.WIKI_KO, split="train").select_columns(["text"])
    namu = load_dataset(config.NAMUWIKI_DATASET, split="train").select_columns(["text"])
    culturax = get_filtered_culturax()
    stories_raw = load_dataset(config.TINYSTORIES_DATASET, split="train").select_columns(["text"])

    # 1. Group TinyStories to preserve context
    stories = Dataset.from_generator(lambda: group_tinystories(stories_raw))

    # 2. Compute dynamic thresholds
    sources_to_filter = {
        "wiki": wiki,
        "namu": namu,
        "culturax": culturax
    }
    ratios = {
        "wiki": config.WIKI_CUT_RATIO,
        "namu": config.NAMU_CUT_RATIO,
        "culturax": config.CULTURAX_CUT_RATIO
    }
    thresholds = compute_all_thresholds(sources_to_filter, config.SAMPLE_SIZE, ratios)

    # 3. Apply Quality Filtering and Cleaning
    wiki = wiki.filter(lambda x: get_quality_score(x["text"]) >= thresholds["wiki"])
    wiki = wiki.map(lambda x: {"text": clean_text(x["text"], is_synthetic=False)})
    
    namu = namu.filter(lambda x: get_quality_score(x["text"]) >= thresholds["namu"])
    namu = namu.map(lambda x: {"text": clean_text(x["text"], is_synthetic=False)})
    
    culturax = culturax.filter(lambda x: get_quality_score(x["text"]) >= thresholds["culturax"])
    culturax = culturax.map(lambda x: {"text": clean_text(x["text"], is_synthetic=False)})
    
    # TinyStories: No quality filtering, use synthetic cleaning mode
    stories = stories.map(lambda x: {"text": clean_text(x["text"], is_synthetic=True)})

    datasets = [wiki, namu, culturax, stories]
    probs = [config.KO_WIKI_RATIO, config.NAMU_RATIO, config.CULTURAX_RATIO, config.TINYSTORIES_RATIO]

    interleaved = interleave_datasets(datasets, probabilities=probs, seed=config.SEED)
    return interleaved.shuffle(seed=config.SEED, buffer_size=config.SHUFFLE_BUFFER)


def train_bpe_tokenizer(vocab_size: int, save_path: str) -> PreTrainedTokenizerFast:
    """
    Trains a BPE tokenizer using the combined dataset.
    Inputs: vocab_size (int), save_path (str)
    Outputs: PreTrainedTokenizerFast
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", config.EOS_TOKEN]
    )

    dataset = load_multi_source_dataset()

    def batch_iterator(batch_size=config.TOKENIZER_BATCH_SIZE):
        batch = []
        for i, item in enumerate(dataset):
            text = item["text"]
            if len(text) >= config.DOC_MIN_LENGTH:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if i > config.TOKENIZER_TRAIN_LIMIT: break
        if batch: yield batch

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
        pad_token="[PAD]", mask_token="[MASK]", eos_token=config.EOS_TOKEN
    )
    fast_tokenizer.save_pretrained(save_path)
    return fast_tokenizer


class KoIterableDataset(IterableDataset):
    """
    Memory-efficient token-packing dataset.
    Expects pre-cleaned text from the source dataset.
    """
    def __init__(self, dataset, tokenizer, context_len: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.eos_id = tokenizer.eos_token_id

    def __iter__(self):
        buffer = deque()
        for item in self.dataset:
            text = item["text"]
            if len(text) < config.DOC_MIN_LENGTH:
                continue
            
            tokens = self.tokenizer.encode(text)
            tokens.append(self.eos_id) # Mark doc boundary
            buffer.extend(tokens)

            while len(buffer) >= self.context_len:
                chunk = [buffer.popleft() for _ in range(self.context_len)]
                yield torch.tensor(chunk, dtype=torch.long)


def get_tokenizer(save_path: str) -> PreTrainedTokenizerFast:
    """
    Loads a saved tokenizer.
    Inputs: save_path (str)
    Outputs: PreTrainedTokenizerFast or None
    """
    if os.path.exists(os.path.join(save_path, "tokenizer.json")):
        return PreTrainedTokenizerFast.from_pretrained(save_path)
    return None
