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

## Project: Korean ~100M LLM from Scratch

### Environment
- Runtime managed by `flake.nix` + `uv`
- All Python dependencies declared in `pyproject.toml`, locked via `uv.lock`
- Never use `pip install` directly; always `uv add` or edit `pyproject.toml`
- Activate env via `uv run` or the flake devShell

### Datasets & Mixing
Korean-only. English dropped entirely (wastes capacity on a 100M model).
Mixing ratios (interleaved):
- Korean Wikipedia (`wikimedia/wikipedia`, `20231101.ko`) — 0.2
- NamuWiki (`heegyu/namuwiki-extracted`) — 0.2
- CulturaX Korean (`uonlp/CulturaX`, `ko`) — 0.4
- TinyStories Korean (`g0ster/TinyStories-Korean`) — 0.2

Rationale: for a 100M model the bottleneck is Korean structure acquisition, not
knowledge capacity. TinyStories (simple, repetitive) helps basic grammar/word
order; CulturaX adds web-text diversity; wiki/namu add clean expository style.

### Preprocessing (`data/preprocess.py`)
All functions pure. Rename `clean_wikipedia_text` → `clean_text` (applies to all
sources). Update every caller in `dataset.py`.

`clean_text` responsibilities (compose small pure fns, ≤30 lines each):
- Remove reference markers `[1]`, `[2]`, …
- Replace URLs with a single space (integrate `replace_urls`; do NOT use a
  `[URL]` placeholder token — model would learn to emit it)
- Strip TinyStories `<|endoftext|>` delimiter (map to EOS at tokenization, see below)
- Normalize whitespace
- Line-level filtering: drop lines shorter than a MIN_LINE_LEN constant or with
  per-line quality score below a FIXED low threshold (~0.2). Purpose is removing
  buggy/noise lines only, so fixed threshold is fine.

`get_quality_score` changes:
- `lang_chars` counts KOREAN syllables + allowed punctuation ONLY. Remove the
  English-alphabet branch.
- KEEP the `num_ratio` penalty — it usefully filters number-dump pages (stat
  sites, tables). Normal prose only takes a small hit from dates/phone numbers.
- Keep newline-ratio and sentence-density terms as-is.

### Dynamic Per-Source Quality Threshold
Document-level threshold is estimated per source (distributions differ), NOT global.
- Sample ~10,000 docs per source, compute `get_quality_score`, take the Nth
  percentile (default 30th → drop bottom 30%) as that source's threshold.
- TinyStories is already clean/synthetic → NO document quality filtering applied.
- Implement as a pure fn returning `{source_name: threshold}`. Filter each source
  BEFORE `interleave_datasets`, so each stream uses its own threshold.
- Constants: SAMPLE_SIZE, DATA_CUT_RATIO, MIN_LINE_LEN, LINE_SCORE_THRESHOLD,
  DOC_MIN_LENGTH in `config.py` (no magic numbers).

### CulturaX Dedup
CulturaX carries a `url` field. Drop rows whose URL contains `ko.wikipedia.org`
or `namu.wiki` to avoid duplicating the wiki/namu sources. CulturaX is already
mC4+OSCAR-filtered, so no heavy additional filtering beyond this + the dynamic
threshold.

### Dataset Iteration (`data/dataset.py`)
- Keep the deque token-packing `IterableDataset`.
- Append the tokenizer EOS token id at the end of EACH document's tokens before
  extending the buffer. This marks document boundaries inside packed contexts so
  the model does not learn spurious cross-document continuations.
- Add `.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)` after interleave.

### Model Spec (`model/`)
- Decoder-only Transformer, target ~100M params, RoPE, RMSNorm (pre-norm), SwiGLU.
- GQA: `N_HEADS=12`, `N_KV_HEADS=4`.
- Weight tying between `tok_embeddings` and `output` head (saves ~24M params,
  significant at this scale).
- **Causal masking via `is_causal=True` in `scaled_dot_product_attention`.** Do
  NOT pass a hand-built boolean upper-triangular mask — the previous bug attended
  to FUTURE tokens (SDPA attends where mask is True), letting the model copy the
  current token and collapsing loss toward 0. Remove manual mask construction in
  `lm.py` and pass `mask=None` down.
- Assert parameter count within [90M, 110M] after construction.

### Training Schedule (WSD: Warmup–Stable–Decay)
Two-phase plan:
1. Phase 1 — Warmup then STABLE (constant) LR. Run long, watch the loss curve,
   stop manually at a good checkpoint. No cosine-to-zero decay in this phase.
2. Phase 2 (later) — Continued training from a chosen checkpoint with a short LR
   cooldown (cosine/linear) to a non-zero floor for final stabilization.

If a cosine schedule is used at all, the LR floor must be a non-zero `min_lr`
(~10% of peak), never decay to 0 — the tail steps at ~0 LR are wasted otherwise.

### Hyperparameters (`config.py`)
```python
VOCAB_SIZE      = 32_000
CONTEXT_LEN     = 2048
D_MODEL         = 768
N_HEADS         = 12
N_KV_HEADS      = 4
N_LAYERS        = 12
D_FFN           = 2048        # SwiGLU; tuned so total ~100M with D_MODEL=768
DROPOUT         = 0.1

BATCH_SIZE      = 4           # micro-batch (>=8 OOMs at ctx 2048 on 16GB)
GRAD_ACC_STEPS  = 32          # effective batch = 128
LR              = 4e-4        # scaled for effective batch 128
WARMUP_STEPS    = 400
MAX_STEPS       = 8000        # large; stop manually on plateau
GRAD_CLIP       = 1.0

LOG_EVERY       = 20
SAVE_EVERY      = 200

# data mixing
KO_WIKI_RATIO   = 0.2
NAMU_RATIO      = 0.2
CULTURAX_RATIO  = 0.4
TINYSTORIES_RATIO = 0.2

# filtering
SAMPLE_SIZE         = 10_000
WIKI_CUT_RATIO      = 0.3
NAMU_CUT_RATIO      = 0.3
CULTURAX_CUT_RATIO  = 0.3
DOC_MIN_LENGTH      = 50
MIN_LINE_LEN        = 20
LINE_SCORE_THRESHOLD = 0.2
SHUFFLE_BUFFER      = 10_000
SEED                = 42
```
Note: English dataset config and ratio are removed from `config.py`.

### Evaluation (`eval_model.py`)
- Load checkpoint via `model.load_state_dict(checkpoint["model_state_dict"])`
  (optimizer load must be optional / skippable).
- Generate AUTOREGRESSIVELY: feed back the predicted token each step, take
  `logits[:, -1, :]` for the next token, truncate context to last `CONTEXT_LEN`.
- Apply a repetition penalty (divide logits of already-generated token ids by a
  constant ~1.3) to curb the repeat-loop failure mode of small/under-trained
  models. Temperature optional; top-k/top-p are NOT helpful for repetition.
- Do not name the eval file `inspect.py` (shadows stdlib `inspect`).

### Code Standards
Apply all rules from Section 2. Project-specific reminders:
- Hyperparameters & filtering constants live in `config.py` (no magic numbers).
- Docstring (purpose / inputs / outputs) on every function.
- Unit tests in `tests/` for every module; run immediately after writing/editing.
- `git push` after each completed feature.
