"""
Evaluation and text generation for the Korean LLM.
Includes autoregressive generation with repetition penalty.
"""

import torch
from model.lm import KoLLM
from data.dataset import get_tokenizer
from train.checkpoint import load_checkpoint
import config


def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device):
    """
    Generates text autoregressively with a repetition penalty.
    Inputs: model, tokenizer, prompt (str), max_new_tokens (int), device
    Outputs: generated text (str)
    """
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate context to CONTEXT_LEN
            current_input = input_ids[:, -config.CONTEXT_LEN:]
            logits, _ = model(current_input)
            next_token_logits = logits[0, -1, :]
            
            # Apply repetition penalty from config
            for token_id in set(generated):
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= config.REPETITION_PENALTY
                else:
                    next_token_logits[token_id] *= config.REPETITION_PENALTY
            
            next_token = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated.append(next_token.item())
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated)


def main():
    """Main evaluation script."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = get_tokenizer(config.TOKENIZER_DIR)
    model = KoLLM(
        vocab_size=config.VOCAB_SIZE,
        context_len=config.CONTEXT_LEN,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_kv_heads=config.N_KV_HEADS,
        n_layers=config.N_LAYERS,
        d_ffn=config.D_FFN
    )
    
    checkpoint_path = f"{config.CHECKPOINT_DIR}/latest.pt"
    # Load model weights only (optimizer=None) as per TODO.md
    load_checkpoint(model, None, checkpoint_path)
        
    model.to(device)
    
    prompts = [
        "대한민국의 수도는 ",
        "인공지능 기술의 미래는 ",
        "오늘 날씨가 아주 "
    ]
    
    for prompt in prompts:
        output = generate(model, tokenizer, prompt, 50, device)
        print(f"Prompt: {prompt}")
        print(f"Generated: {output}\n")


if __name__ == "__main__":
    main()
