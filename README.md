# Korean 100M LLM from Scratch 🚀

이 프로젝트는 한국어 위키백과 데이터를 사용하여 약 1억 개(100M)의 파라미터를 가진 Decoder-only Transformer 언어 모델을 처음부터(from scratch) 구현한 프로젝트입니다. LLaMA 등 최신 LLM 아키텍처에서 사용하는 기술들을 반영하여 설계되었습니다.

## 🏗 핵심 아키텍처 및 기술 (Key Features)

### 1. Modern Transformer & Tokenization
*   **Byte-level BPE Tokenizer**: 외부 의존성 없이 데이터 기반으로 최적화된 토큰을 생성하는 Byte-level BPE를 사용합니다.
*   **Weight Tying**: 입력 임베딩과 출력 레이어의 가중치를 공유하여 파라미터 효율성을 극대화하고 깊은 모델 구조를 확보했습니다.
*   **RoPE, RMSNorm, SwiGLU, GQA**: 최신 LLM(Llama 3 등)의 핵심 아키텍처를 그대로 반영했습니다.

### 2. Model Specification
*   **Target Parameters**: ~100M (최종 약 100.1M)
*   **Layers**: 12
*   **D_Model**: 768
*   **Attention Heads**: 12 (Query), 4 (KV - GQA 적용)

### 3. Data Pipeline & Scaling
*   **Multi-Source Dataset**: Chinchilla Scaling Law(100M 모델 기준 약 20억 토큰)를 충족하기 위해 대규모 데이터를 확보했습니다.
    *   **KO Wikipedia** (40%): 정제된 백과사전 지식
    *   **EN Wikipedia** (20%): 추론 능력 및 영어 이해도 향상
    *   **NamuWiki** (40%): 방대한 한국어 구어체 및 최신 정보
*   **Streaming & Interleaving**: 대용량 데이터를 메모리 효율적으로 처리하기 위해 실시간 스트리밍 및 가중치 기반 인터리빙 방식을 사용합니다.
*   **Advanced Quality Scoring**: 한글/영문 비율, 개행 밀도, 문장 밀도 등을 종합하여 고품질 텍스트만 선별합니다.

## 📁 프로젝트 구조 (Project Structure)

```text
.
├── config.py          # 모든 하이퍼파라미터 및 상수 설정
├── data/
│   ├── preprocess.py  # 순수 함수 기반 데이터 정제 및 필터링
│   └── dataset.py     # PyTorch IterableDataset 및 토크나이저 로직
├── model/
│   ├── attention.py   # RoPE가 적용된 Multi-Head Attention
│   ├── block.py       # RMSNorm + SwiGLU + Attention 블록
│   └── lm.py          # 전체 언어 모델 조립 및 파라미터 체크
├── train/
│   ├── trainer.py     # 학습 루프 및 Mixed Precision 지원
│   └── checkpoint.py  # 체크포인트 저장 및 로드
├── tests/             # 모든 모듈에 대한 단위 테스트
├── main.py            # 전체 프로세스 실행 엔트리 포인트
├── flake.nix          # Nix 기반 개발 환경 설정 (libstdc++ 등 해결)
└── pyproject.toml     # uv 기반 의존성 관리
```

## 🛠 실행 방법 (Usage)

### 환경 준비
이 프로젝트는 `nix`와 `uv`를 사용하여 의존성을 관리합니다.

```bash
# 1. Nix 개발 환경 진입 (필요한 라이브러리 및 uv 자동 설정)
nix develop

# 2. 의존성 동기화
uv sync
```

### 테스트 실행
모든 모듈의 정상 작동 여부를 확인합니다.

```bash
nix develop --command uv run pytest
```

### 학습 시작
토크나이저 훈련부터 모델 학습까지 한 번에 실행합니다.

```bash
nix develop --command uv run python main.py
```

## 📜 코딩 원칙
이 프로젝트는 다음의 엄격한 코딩 표준을 준수합니다.
*   모든 함수는 **Single Responsibility Principle(SRP)**을 따릅니다.
*   함수 길이는 최대 **30라인** 이하로 유지합니다.
*   부수 효과가 없는 **순수 함수** 작성을 지향합니다.
*   모든 주요 로직은 **단위 테스트**를 포함합니다.
