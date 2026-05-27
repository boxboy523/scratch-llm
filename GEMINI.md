## Gemini CLI Rules

### 1. Communication & Behavior
- **Polite Language**: Always communicate with the user using polite language (honorifics/존댓말).
- **No Hallucination**: If information is unknown, do not speculate or hallucinate. Clearly state that the information is unknown and request further details from the user.
- **CONCISE OUTPUT**: Keep explanations brief and focus on the task at hand.

### 2. Coding Standards & Principles
- **Pure Functions**: Implement features using pure functions as much as possible to ensure predictability and testability.
- **Single Responsibility Principle (SRP)**: Every function must perform exactly **one task**. Avoid functions that perform multiple actions.
- **Modularization**: Limit a single file or class to a maximum of **10 functions** (including class methods). If more are needed, refactor and decompose into separate modules or classes.
- **Class Implementation**: When using classes, treat methods as functions subject to the 10-function limit. Minimize internal state changes and prioritize immutability and composition over complex inheritance or state management.
- **Argument Limit**: A function or method should have at most **6 parameters**. If more are needed, group related data into an object or refactor.
- **Function Length**: A function's body should not exceed **30 lines**. This ensures the function remains focused and readable.
- **Indentation Limit**: Keep the indentation depth to a maximum of **2 levels**. Use early returns or extract logic into smaller functions to avoid deep nesting.
- **Immutability First**: Favor `const` over `let`. Avoid variable reassignment and side effects. When changing data, return a new object or array instead of mutating the existing one.
- **No Magic Numbers/Strings**: Do not use hard-coded numbers or strings directly in the logic. Define them as descriptive constants at the top of the file or in a configuration module.
- **Mandatory Testing**: Always include corresponding test code (e.g., unit tests) for every piece of logic created.
- **Immediate Verification**: After generating or modifying code, immediately execute tests to verify correctness and prevent regressions.
- **Function Documentation**: Every function must include a preceding comment describing its purpose, inputs, and outputs.
- **Mandatory Git Push**: Always execute `git push` after feature implementation to maintain version control.

---

## Project: Korean 100M LLM from Scratch

### Environment
- Runtime managed by `flake.nix` + `uv`
- All Python dependencies declared in `pyproject.toml`, locked via `uv.lock`
- Never use `pip install` directly; always `uv add` or edit `pyproject.toml`
- Activate env via `uv run` or the flake devShell

### Dataset
- Primary: Korean Wikipedia HuggingFace dataset (`wikimedia/wikipedia`, `20231101.ko`)
- Secondary: NamuWiki dump (`heegyu/namuwiki-extracted` or raw XML dump)
- Preprocessing must be implemented as pure functions in `data/preprocess.py`
- Quality filtering required: Korean character ratio ≥ 0.5, min length 50 chars

### Model Spec
- Architecture: Decoder-only Transformer (GPT-style)
- Target size: ~100M parameters
- Positional encoding: RoPE
- Normalization: RMSNorm (pre-norm)
- Activation: SwiGLU
- Context length: 1024
- Implement via raw `torch.nn` — do NOT use `transformers.AutoModel`; use `transformers` only for tokenizer and dataset utilities

### Tokenizer
- Use `transformers.PreTrainedTokenizerFast` wrapping a trained `tokenizers` BPE model
- Vocab size: 32000
- Train tokenizer on the same dataset before model training
- Save to `artifacts/tokenizer/`

### Code Standards
Apply all rules from Section 2 above. Additional project-specific notes:
- Every function must have a docstring (purpose / inputs / outputs)
- Every module must have corresponding unit tests in `tests/`
- Run tests immediately after writing or modifying code
- `git push` after each completed feature

### Project Structure
```
.
├── flake.nix
├── pyproject.toml
├── config.py          # all hyperparameters as constants
├── data/
│   ├── preprocess.py  # pure fn: load, filter, tokenize
│   └── dataset.py     # torch Dataset wrapper
├── model/
│   ├── attention.py   # RoPE multi-head attention
│   ├── block.py       # TransformerBlock (RMSNorm + SwiGLU MLP)
│   └── lm.py          # full decoder LM, parameter count check
├── train/
│   ├── trainer.py     # training loop
│   └── checkpoint.py  # save/load logic
├── artifacts/
│   ├── tokenizer/
│   └── checkpoints/
└── tests/
    ├── test_preprocess.py
    ├── test_model.py
    └── test_trainer.py
```

### Hyperparameter Reference (config.py)
```python
VOCAB_SIZE      = 32_000
CONTEXT_LEN     = 1024
D_MODEL         = 768
N_HEADS         = 12
N_LAYERS        = 12
D_FFN           = D_MODEL * 4        # SwiGLU: split into two halves internally
DROPOUT         = 0.1
BATCH_SIZE      = 32
LR              = 3e-4
WARMUP_STEPS    = 2_000
MAX_STEPS       = 100_000
GRAD_CLIP       = 1.0
LOG_EVERY       = 100
SAVE_EVERY      = 1_000
```

### Parameter Count Verification
After building the model, assert the parameter count is within [90M, 110M]:
```python
n_params = sum(p.numel() for p in model.parameters())
assert 90_000_000 <= n_params <= 110_000_000, f"Unexpected param count: {n_params}"
```

### Training Notes
- Use `torch.cuda.amp` (bfloat16) for mixed precision
- Gradient accumulation if needed to hit effective batch size
- Log loss to stdout every `LOG_EVERY` steps; optionally wandb
- Dataset streaming preferred (`datasets` library `streaming=True`) to avoid full disk dump
