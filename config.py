"""
Project configuration and hyperparameters.
"""

# Tokenizer and Model constraints
VOCAB_SIZE = 32_000
CONTEXT_LEN = 2048
D_MODEL = 768
N_HEADS = 12
N_KV_HEADS = 4  # GQA: Number of Key/Value heads
N_LAYERS = 12
D_FFN = 2048  # SwiGLU: target ~100M total params with D_MODEL=768
DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 4  # Micro-batch size
GRAD_ACC_STEPS = 32  # Effective batch size = 32 * 4 = 128
LR = 4e-4

WARMUP_STEPS = 200
MAX_STEPS = 4000
GRAD_CLIP = 1.0

# Logging and Checkpointing
LOG_EVERY = 20
SAVE_EVERY = 200
MIN_QUALITY_SCORE = 0.6  # Threshold for advanced quality filtering

# Paths
TOKENIZER_DIR = "artifacts/tokenizer/"
CHECKPOINT_DIR = "artifacts/checkpoints/"
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.ko"
EN_DATASET_CONFIG = "20231101.en"
NAMUWIKI_DATASET = "heegyu/namuwiki-extracted"
STREAMING = False

# Mixing Ratios (Wiki KO : Wiki EN : NamuWiki)
KO_WIKI_RATIO = 0.2
EN_WIKI_RATIO = 0.1
NAMU_RATIO = 0.7
