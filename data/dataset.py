"""
Memory-efficient lazy dataset loading with manual interleaving.
"""

import os
import torch
import random
import unicodedata
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from data.preprocess import is_quality_text, clean_wikipedia_text
import config

# Force HuggingFace to use minimal memory
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"
os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"

def manual_interleave_generator(tokenizer):
    """
    Manually interleaves streaming datasets with ultra-defensive memory settings.
    """
    try:
        # We wrap each in a try-except to handle network hiccups in streaming
        ko_wiki = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split="train", streaming=True)
        en_wiki = load_dataset(config.DATASET_NAME, config.EN_DATASET_CONFIG, split="train", streaming=True)
        # NamuWiki can be the main memory hog, we'll try to load it separately
        namu = load_dataset(config.NAMUWIKI_DATASET, split="train", streaming=True)
        
        ko_iter = iter(ko_wiki)
        en_iter = iter(en_wiki)
        namu_iter = iter(namu)
    except Exception as e:
        print(f"Error initializing streams: {e}")
        return

    iters = [ko_iter, en_iter, namu_iter]
    probs = [config.KO_WIKI_RATIO, config.EN_WIKI_RATIO, config.NAMU_RATIO]
    
    while True:
        idx = random.choices(range(len(iters)), weights=probs, k=1)[0]
        try:
            item = next(iters[idx])
            text = item.get("text", "")
            if text and is_quality_text(text, min_score=config.MIN_QUALITY_SCORE):
                yield clean_wikipedia_text(text)
        except StopIteration:
            # Re-initialize the stream if it ends
            if idx == 0: ko_iter = iter(ko_wiki)
            elif idx == 1: en_iter = iter(en_wiki)
            else: namu_iter = iter(namu)
            iters[idx] = [ko_iter, en_iter, namu_iter][idx]
            continue
        except Exception:
            continue


def train_bpe_tokenizer(vocab_size: int, save_path: str) -> PreTrainedTokenizerFast:
    """Trains a BPE tokenizer using a strictly lazy generator."""
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    
    # Simple iterator for tokenizer training
    def tokenizer_iterator(limit=100_000):
        gen = manual_interleave_generator(None)
        for i, text in enumerate(gen):
            yield text
            if i >= limit: break
            if i % 1000 == 0: print(f"\rTokenizer training data: {i} docs", end="")

    tokenizer.train_from_iterator(tokenizer_iterator(), trainer=trainer)
    
    if not os.path.exists(save_path): os.makedirs(save_path)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
        pad_token="[PAD]", mask_token="[MASK]",
    )
    fast_tokenizer.save_pretrained(save_path)
    return fast_tokenizer


class KoIterableDataset(IterableDataset):
    """
    Strictly lazy Iterable dataset for bilingual text.
    """
    def __init__(self, tokenizer, context_len=1024):
        self.tokenizer = tokenizer
        self.context_len = context_len

    def __iter__(self):
        buffer = []
        gen = manual_interleave_generator(self.tokenizer)
        
        for text in gen:
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            
            while len(buffer) >= self.context_len:
                chunk = buffer[:self.context_len]
                buffer = buffer[self.context_len:]
                yield torch.tensor(chunk, dtype=torch.long)


def get_tokenizer(save_path: str) -> PreTrainedTokenizerFast:
    if os.path.exists(os.path.join(save_path, "tokenizer.json")):
        return PreTrainedTokenizerFast.from_pretrained(save_path)
    return None
