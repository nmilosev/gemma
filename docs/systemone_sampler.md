# System One Classifier

The `gm.text.SystemOneSampler` provides ultra-low-latency, non-generative classification, intent routing, and decision triage for Gemma models.

Traditional LLM generation uses an autoregressive decoding loop where tokens are decoded sequentially ($O(T)$). In contrast, `SystemOneSampler` evaluates structured decisions via single-pass tree-attention prefill ($O(1)$) directly from vocabulary logits without generating any new tokens.

## Jev Decision Primitives

Modeled after Jev decision primitives, `SystemOneSampler` supports three zero-shot decision types:

| Primitive | Enum | Description | Output |
| :--- | :--- | :--- | :--- |
| **NOUL** | `gm.text.QuestionType.NOUL` | Calibrated binary decision | `True` or `False`, probabilities, confidence |
| **CHOICE** | `gm.text.QuestionType.CHOICE` | Categorical selection among $K$ options | Option text, index, probability distribution, confidence |
| **SCORE** | `gm.text.QuestionType.SCORE` | Ordinal rating scale (e.g. 1 to 5) | Discrete bucket, continuous expected value $\mathbb{E}[S]$, confidence |

### 1. NOUL (Binary Decision)

Evaluates whether a statement holds for a given context. Uses calibrated logits between affirmative (`True`) and negative (`False`) token subspaces:

```python
result = sampler.evaluate_noul(
    state="Customer received broken hardware and demands immediate refund.",
    question="Is the user requesting a monetary refund?",
)
print(result.value)       # True / False
print(result.confidence)  # Normalized Shannon confidence in [0.0, 1.0]
```

### 2. CHOICE (Categorical Routing)

Selects the best option among candidate categories (e.g., ticket routing, intent detection, topic classification):

```python
result = sampler.evaluate_choice(
    state="Payment gateway returning HTTP 504 timeouts across checkout endpoints.",
    question="Which engineering team is responsible for this alert?",
    options=[
        "Database Infrastructure",
        "Frontend Checkout",
        "Network Operations",
        "Security & Compliance",
    ],
)
print(result.value)           # e.g. "Network Operations"
print(result.selected_index)  # e.g. 2
print(result.probabilities)   # {'Database Infrastructure': 0.12, ...}
```

### 3. SCORE (Ordinal Rating)

Evaluates an integer scale (e.g., 1 to 5) and computes both the modal bucket and the continuous expectation $\mathbb{E}[S] = \sum_{s} s \cdot P(S = s)$:

```python
result = sampler.evaluate_score(
    state="Production database at 99% CPU load with blocked query locks.",
    question="Rate the incident severity on an ITIL scale of 1 to 5.",
    score_range=(1, 5),
)
print(result.value)           # e.g. 5
print(result.expected_value)  # e.g. 4.87
```

## Tree-Attention Prefill ($O(1)$)

When evaluating multiple decision questions on the same input context, running each question through an independent forward pass incurs redundant prefix computation and $N$ forward passes.

`SystemOneSampler` uses **tree-attention packing** (`gm.text.build_tree_attention_pack`):

1. **Shared State Prefix**: The common context is encoded at the root of the sequence.
2. **Question Branches**: All $N$ questions are appended into the same packed sequence.
3. **RoPE Position Restarts**: The position IDs of each branch restart immediately after the shared context prefix ($P = [0 \dots L_{\text{state}}-1, L_{\text{state}} \dots L_{\text{state}}+L_{q_1}-1, L_{\text{state}} \dots L_{\text{state}}+L_{q_2}-1]$), preventing sequential positional drift.
4. **2D Block-Diagonal Causal Masking**: The attention mask allows each branch to attend to the full shared state and to itself causally, but completely masks out other branches.

This guarantees:
* **Zero inter-question contamination**: Branch $B_j$ cannot observe branch $B_i$.
* **Bit-level numerical parity**: Branch logits match an isolated forward pass ($\Delta = 0$).
* **Single forward pass**: All $N$ decisions resolve in a single prefill pass.

```python
response = sampler.evaluate_systemone(
    state="User inquiry or system telemetry...",
    questions=[
        gm.text.QuestionSpec(id="urgent", text="Is urgent?", type=gm.text.QuestionType.NOUL),
        gm.text.QuestionSpec(id="team", text="Route to?", type=gm.text.QuestionType.CHOICE, options=["A", "B", "C"]),
        gm.text.QuestionSpec(id="rating", text="Severity 1-5", type=gm.text.QuestionType.SCORE, score_range=(1, 5)),
    ],
)
```

## Context-Free Null Calibration

LLMs often exhibit token frequency and positional priors (e.g. favoring "True" over "False" or "A" over "B" regardless of context).

`SystemOneSampler` eliminates this prior bias via context-free null calibration:

$$z_{\text{cal}} = z_{\text{real}} - z_{\text{null}}$$

Where $z_{\text{null}}$ represents the unnormalized logits evaluated against a null state (e.g. `"N/A"`).

You can precompute null priors once at application startup to achieve maximum serving throughput:

```python
sampler.precompute_null_priors(questions)
```

## Confidence Scoring

Confidence is computed using normalized Shannon entropy:

$$H(P) = -\sum_{i=1}^K P_i \log_2(P_i)$$
$$C = 1 - \frac{H(P)}{\log_2(K)}$$

* $C = 1.0$: Deterministic certainty ($P_k = 1$).
* $C = 0.0$: Maximum uncertainty / uniform distribution ($P_k = \frac{1}{K}$).

## Model Support

`SystemOneSampler` is compatible with all Gemma models:

### Gemma 3 (Recommended for Local Evaluation)

Gemma 3 1B (`gemma3-1b-it`) is compact (~1.5 GB weights) and can be loaded and evaluated locally on CPU or consumer GPU:

```python
from gemma import gm

model = gm.nn.Gemma3_1B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
tokenizer = gm.text.Gemma3Tokenizer()

sampler = gm.text.SystemOneSampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
)
```

See the complete runnable example in [`examples/systemone_gemma3.py`](https://github.com/google-deepmind/gemma/blob/main/examples/systemone_gemma3.py).

### Gemma 4

For Gemma 4 models (e.g., `gemma4-e2b-it`), use `Gemma4Tokenizer` and pass `text_only=True` to exclude media encoders during text classification:

```python
from gemma import gm

# Smallest Gemma 4 checkpoint (2B effective parameters)
model = gm.nn.Gemma4_E2B(text_only=True)
params = gm.ckpts.load_params(
    gm.ckpts.CheckpointPath.GEMMA4_E2B_IT,
    text_only=True,
)
tokenizer = gm.text.Gemma4Tokenizer()

sampler = gm.text.SystemOneSampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
)
```

