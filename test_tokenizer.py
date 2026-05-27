"""
Interactive test script for Standard BPE Tokenizer.
"""
import config
from data.dataset import get_tokenizer
import os

def main():
    print("--- Standard BPE Tokenizer Interactive Tester ---")
    
    tokenizer_path = config.TOKENIZER_DIR
    if not os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        print(f"Error: Tokenizer not found at {tokenizer_path}.")
        print("Please run 'python main.py' first to train the tokenizer.")
        return
        
    tokenizer = get_tokenizer(tokenizer_path)
    print("Tokenizer loaded successfully.")
    print("종료하려면 'exit'를 입력하세요.\n")
    
    while True:
        text = input("입력 문장: ")
        if text.lower() == 'exit':
            break
            
        if not text.strip():
            continue
            
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        print(f"\n[1] 토큰화 결과 (IDs):")
        print(f"    {tokens}")
        
        print(f"\n[2] 토큰화 결과 (Subwords):")
        print(f"    {tokenizer.convert_ids_to_tokens(tokens)}")
        
        print(f"\n[3] 디코딩 결과 (복원):")
        print(f"    {decoded}")
        print("-" * 50)

if __name__ == "__main__":
    main()
