# TODO — GEMINI.md 준수 미흡 사항

## 1. `config.py` — 누락 상수 (No Magic Numbers 규칙 위반)

아래 값들이 소스 코드 곳곳에 리터럴로 박혀 있다. 모두 `config.py`로 올려야 한다.

| 상수 이름 | 값 | 사용 위치 |
|---|---|---|
| `REPETITION_PENALTY` | `1.3` | `eval_model.py:22` |
| `TOKENIZER_TRAIN_LIMIT` | `100_000` | `data/dataset.py:93` |
| `TOKENIZER_BATCH_SIZE` | `1000` | `data/dataset.py:84` |
| `RMS_NORM_EPS` | `1e-6` | `model/block.py:16` |
| `ROPE_THETA` | `10000.0` | `model/attention.py:11` |
| `EOS_TOKEN` | `"<\|endoftext\|>"` | `data/preprocess.py:35`, `data/dataset.py:79` |

`get_quality_score` 내부에도 매직 넘버가 집중되어 있다 (`data/preprocess.py:68–70`):

```python
sentence_density = sentences / (total / 50)   # 50 → SENTENCE_UNIT
return lang_ratio - newline_ratio * 3 ...     # 3  → NEWLINE_PENALTY
         - max(0, 0.3 - sentence_density) * 0.5  # 0.3, 0.5 → 상수화
```

---

## 2. `model/lm.py` — 파라미터 카운트 어서션 누락

파라미터 카운트 어서션이 `main.py:48`에만 있고 모델 생성자 내부에 없다. GEMINI.md는 "Assert parameter count within [90M, 110M] after construction"을 명시하므로 `KoLLM.__init__` 끝에 추가해야 한다.

---

## 3. `data/dataset.py` — 동적 임계값 계산이 순수 함수로 분리되지 않음

GEMINI.md: "Implement as a pure fn returning `{source_name: threshold}`".  
현재는 `compute_source_threshold`가 단일 float을 반환하고, 호출·조합 로직이 `load_multi_source_dataset` 안에 인라인으로 섞여 있다.  
`compute_all_thresholds(datasets_dict, ...) -> dict[str, float]` 순수 함수로 분리 필요.

---

## 4. `train/checkpoint.py` — `load_checkpoint` optimizer 필수 파라미터

GEMINI.md: "optimizer load must be optional / skippable".  
현재 `load_checkpoint(model, optimizer, path)` 에서 `optimizer`가 필수(`checkpoint.py:34`). 추론/평가 시 optimizer 없이 호출 불가.

**수정**: `optimizer: Optional[Optimizer] = None` 으로 변경하고, None이면 `optimizer_state_dict` 로드를 건너뜀.

---

## 5. `eval_model.py` — 매직 넘버

`penalty = 1.3` (`eval_model.py:22`) → `config.REPETITION_PENALTY` 참조로 교체.

---

## 6. 테스트 미비 (Mandatory Testing 위반)

GEMINI.md: "Unit tests in `tests/` for every module".

| 모듈 | 테스트 파일 | 미비 항목 |
|---|---|---|
| `data/dataset.py` | 없음 | `compute_source_threshold`, `KoIterableDataset.__iter__`, `get_tokenizer` |
| `eval_model.py` | 없음 | `generate` 함수 (반복 패널티 적용 검증 포함) |
| `train/checkpoint.py` | `test_trainer.py` | `load_checkpoint` with `optimizer=None` 케이스 |

---

## 7. 함수 문서화 누락 (Function Documentation 위반)

GEMINI.md: "Every function must include a preceding comment describing its purpose, inputs, and outputs."

| 위치 | 함수 | 현황 |
|---|---|---|
| `data/dataset.py:72` | `train_bpe_tokenizer` | docstring 첫 줄만 있고 Inputs/Outputs 없음 |
| `data/dataset.py:137` | `get_tokenizer` | 단 한 줄 설명만, Inputs/Outputs 없음 |
| `model/block.py:22` | `RMSNorm._norm` | Inputs/Outputs 없음 |

---

## 우선순위 요약

| 우선순위 | 항목 |
|---|---|
| 높음 | §4 optimizer optional / §1 config 상수 추가 |
| 중간 | §2 파라미터 카운트 어서션 / §3 임계값 순수 함수 분리 / §5 eval 매직 넘버 |
| 낮음 | §6 테스트 추가 / §7 docstring 보완 |
