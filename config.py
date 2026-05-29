"""
Project configuration and hyperparameters for Korean ~100M LLM.
All constants are centralized here to avoid magic numbers.
"""

# --- Tokenizer and Model Architecture ---
VOCAB_SIZE = 32_000         # 토크나이저 어휘 사전 크기
CONTEXT_LEN = 2048          # 모델이 한 번에 처리하는 최대 토큰 길이
D_MODEL = 768               # 임베딩 및 어텐션 레이어의 차원 수
N_HEADS = 12                # 멀티헤드 어텐션의 헤드 수
N_KV_HEADS = 4              # GQA(Grouped Query Attention)를 위한 Key/Value 헤드 수
N_LAYERS = 12               # 트랜스포머 블록(레이어)의 개수
D_FFN = 2048                # Feed-Forward 네트워크의 중간 차원 크기
DROPOUT = 0.1               # 드롭아웃 확률
RMS_NORM_EPS = 1e-6         # RMSNorm 안정성을 위한 작은 상수값
ROPE_THETA = 10000.0        # RoPE 위치 임베딩의 주기 제어 상수

# --- Training Hyperparameters (WSD Schedule) ---
BATCH_SIZE = 4              # 장치당 마이크로 배치 크기
GRAD_ACC_STEPS = 32         # 그래디언트 누적 단계 (실질 배치 크기 = BATCH_SIZE * GRAD_ACC_STEPS = 128)
LR = 4e-4                   # 최대 학습률
WARMUP_STEPS = 400          # 학습 초기 학습률을 선형적으로 증가시키는 단계 수
MAX_STEPS = 8000            # 전체 학습 단계 수
GRAD_CLIP = 1.0             # 그래디언트 클리핑 임계값

# --- Inference and Generation ---
REPETITION_PENALTY = 1.3    # 생성 시 동일 토큰 반복을 억제하기 위한 패널티 가중치

# --- Logging and Checkpointing ---
LOG_EVERY = 20              # 학습 로그를 출력할 단계 주기
SAVE_EVERY = 200            # 체크포인트를 저장할 단계 주기

# --- Data Mixing Ratios (Interleaving) ---
KO_WIKI_RATIO = 0.2         # 한국어 위키백과 데이터 비율
NAMU_RATIO = 0.2            # 나무위키 데이터 비율
CULTURAX_RATIO = 0.4        # CulturaX(웹 크롤링) 데이터 비율
TINYSTORIES_RATIO = 0.2     # TinyStories(합성 데이터) 데이터 비율

# --- Filtering and Preprocessing ---
SAMPLE_SIZE = 10_000        # 동적 임계값 계산을 위한 소스별 샘플 문서 수
WIKI_CUT_RATIO = 0.3        # 위키백과 품질 하위 필터링 비율 (0.3 = 하위 30% 제거)
NAMU_CUT_RATIO = 0.3        # 나무위키 품질 하위 필터링 비율
CULTURAX_CUT_RATIO = 0.3    # CulturaX 품질 하위 필터링 비율
DOC_MIN_LENGTH = 50         # 필터링 후 유지할 최소 문서 길이 (문자 단위)
MIN_LINE_LEN = 20           # 개별 라인이 유지되기 위한 최소 길이
LINE_SCORE_THRESHOLD = 0.2  # 개별 라인이 유지되기 위한 최소 품질 점수
SHUFFLE_BUFFER = 10_000     # 데이터 믹싱 시 셔플링을 위한 버퍼 크기
SEED = 42                   # 데이터 처리 및 셔플링 재현성을 위한 시드값

# --- Quality Score Penalties (get_quality_score) ---
SENTENCE_UNIT = 50          # 문장 밀도 계산을 위한 기준 문자 수
NEWLINE_PENALTY = 3         # 줄바꿈 과다 시 점수 감점 가중치
DENSITY_THRESHOLD = 0.3     # 적정 문장 밀도 임계값
DENSITY_WEIGHT = 0.5        # 문장 밀도 부족 시 감점 가중치

# --- Tokenizer Training ---
TOKENIZER_TRAIN_LIMIT = 100_000  # 토크나이저 학습에 사용할 최대 문서 수
TOKENIZER_BATCH_SIZE = 1000      # 토크나이저 학습 시 배치 크기
EOS_TOKEN = "<|endoftext|>"      # 문서 종료를 나타내는 특수 토큰 문자열

# --- Paths and Datasets ---
TOKENIZER_DIR = "artifacts/tokenizer/"        # 토크나이저 저장 경로
CHECKPOINT_DIR = "artifacts/checkpoints/"    # 체크포인트 저장 경로
WIKI_DATASET = "wikimedia/wikipedia"           # HF 위키백과 데이터셋 이름
WIKI_KO = "20231101.ko"                        # 위키백과 한국어 버전 설정
NAMUWIKI_DATASET = "heegyu/namuwiki-extracted" # 나무위키 데이터셋 이름
CULTURAX_DATASET = "uonlp/CulturaX"            # CulturaX 데이터셋 이름
TINYSTORIES_DATASET = "g0ster/TinyStories-Korean" # TinyStories 데이터셋 이름
STREAMING = True                            # 데이터셋 로딩 시 스트리밍 모드 사용 여부 (True 시 메모리 절약)
