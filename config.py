"""
Project configuration and hyperparameters.
"""

# Tokenizer and Model constraints
VOCAB_SIZE = 32_000
CONTEXT_LEN = 1024
D_MODEL = 768
N_HEADS = 12
N_KV_HEADS = 4  # GQA: Number of Key/Value heads
N_LAYERS = 12
D_FFN = 2048  # SwiGLU: target ~100M total params with D_MODEL=768
DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 8
LR = 3e-4
WARMUP_STEPS = 2_000
MAX_STEPS = 100_000
GRAD_CLIP = 1.0

# Logging and Checkpointing
LOG_EVERY = 100
SAVE_EVERY = 1_000
MIN_QUALITY_SCORE = 0.6  # Threshold for advanced quality filtering

# Paths
TOKENIZER_DIR = "artifacts/tokenizer/"
CHECKPOINT_DIR = "artifacts/checkpoints/"
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.ko"
EN_DATASET_CONFIG = "20231101.en"
NAMUWIKI_DATASET = "heegyu/namuwiki-extracted"

# Mixing Ratios (Wiki KO : Wiki EN : NamuWiki)
KO_WIKI_RATIO = 0.4
EN_WIKI_RATIO = 0.2
NAMU_RATIO = 0.4
