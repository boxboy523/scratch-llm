"""
Entry point for training the Korean 100M LLM.
"""

import config
from data.dataset import train_bpe_tokenizer, get_tokenizer, KoIterableDataset
from model.lm import KoLLM, get_parameter_count
from train.trainer import run_training
import os


def main():
    """Main execution function."""
    print("--- Korean 100M LLM from Scratch ---")
    
    # 1. Tokenizer
    if not os.path.exists(os.path.join(config.TOKENIZER_DIR, "tokenizer.json")):
        print("Training tokenizer on multi-source dataset (Wiki KO/EN + Namu)...")
        tokenizer = train_bpe_tokenizer(
            config.VOCAB_SIZE, 
            config.TOKENIZER_DIR
        )
    else:
        print("Loading existing tokenizer...")
        tokenizer = get_tokenizer(config.TOKENIZER_DIR)
        
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
    
    n_params = get_parameter_count(model)
    print(f"Model initialized with {n_params:,} parameters.")
    
    # 3. Dataset & Training
    print("Preparing bilingual dataset and starting training loop...")
    dataset = KoIterableDataset(
        tokenizer, 
        config.CONTEXT_LEN
    )
    
    run_training(model, dataset)


if __name__ == "__main__":
    main()
