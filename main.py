"""
Entry point for training the Korean-only 100M LLM.
Integrates tokenizer training, model initialization, and the training loop.
"""

import os
import config
from data.dataset import (
    train_bpe_tokenizer,
    get_tokenizer,
    load_multi_source_dataset,
    KoIterableDataset
)
from model.lm import KoLLM
from train.trainer import run_training


def main():
    """
    Main execution function to orchestrate the training process.
    """
    print("--- Korean 100M LLM from Scratch ---")

    # 1. Tokenizer
    tokenizer = get_tokenizer(config.TOKENIZER_DIR)
    if not tokenizer:
        print("Training tokenizer on multi-source Korean dataset...")
        tokenizer = train_bpe_tokenizer(
            config.VOCAB_SIZE,
            config.TOKENIZER_DIR
        )
    else:
        print("Existing tokenizer loaded.")

    # 2. Model
    model = KoLLM(
        vocab_size=config.VOCAB_SIZE,
        context_len=config.CONTEXT_LEN,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_kv_heads=config.N_KV_HEADS,
        n_layers=config.N_LAYERS,
        d_ffn=config.D_FFN
    )
    
    # Verify parameter count
    param_count = sum(p.numel() for p in model.parameters())
    assert 90_000_000 <= param_count <= 110_000_000, f"Model has {param_count} params, expected [90M, 110M]"
    print(f"Model initialized with {param_count:,} parameters.")

    # 3. Dataset & Training
    print("Preparing multi-source dataset and starting training loop...")
    raw_dataset = load_multi_source_dataset()
    dataset = KoIterableDataset(
        raw_dataset,
        tokenizer,
        config.CONTEXT_LEN
    )

    run_training(model, dataset)


if __name__ == "__main__":
    main()
