"""
Project configuration and hyperparameters for Korean ~100M LLM.
"""

# Tokenizer and Model constraints
VOCAB_SIZE = 32_000
CONTEXT_LEN = 2048
D_MODEL = 768
N_HEADS = 12
N_KV_HEADS = 4  # Grouped Query Attention
N_LAYERS = 12
D_FFN = 2048  # SwiGLU; tuned for ~100M total params
DROPOUT = 0.1

# Training hyperparameters (WSD Schedule)
BATCH_SIZE = 4  # Micro-batch size
GRAD_ACC_STEPS = 32  # Effective batch size = 128
LR = 4e-4
WARMUP_STEPS = 400
MAX_STEPS = 8000
GRAD_CLIP = 1.0

# Logging and Checkpointing
LOG_EVERY = 20
SAVE_EVERY = 200

# Data mixing ratios
KO_WIKI_RATIO = 0.2
NAMU_RATIO = 0.2
CULTURAX_RATIO = 0.4
TINYSTORIES_RATIO = 0.2

# Filtering and Preprocessing constants
SAMPLE_SIZE = 10_000
CUT_PERCENTILE = 30
DOC_MIN_LENGTH = 50
MIN_LINE_LEN = 20
LINE_SCORE_THRESHOLD = 0.2
SHUFFLE_BUFFER = 10_000
SEED = 42

# Paths and Datasets
TOKENIZER_DIR = "artifacts/tokenizer/"
CHECKPOINT_DIR = "artifacts/checkpoints/"
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_KO = "20231101.ko"
NAMUWIKI_DATASET = "heegyu/namuwiki-extracted"
CULTURAX_DATASET = "uonlp/CulturaX"
TINYSTORIES_DATASET = "g0ster/TinyStories-Korean"
