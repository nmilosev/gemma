# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SystemOneSampler: Zero-shot non-generative Jev decision sampler."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import enum
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


class QuestionType(enum.Enum):
  """Decision primitive types matching Jev specification."""

  NOUL = "noul"  # Binary / boolean decision
  CHOICE = "choice"  # Categorical selection
  SCORE = "score"  # Ordinal numeric rating


@dataclasses.dataclass(frozen=True)
class QuestionSpec:
  """Defines a structured Jev question to evaluate."""

  id: str
  text: str
  type: QuestionType
  options: Optional[List[str]] = None  # Required for CHOICE
  score_range: Tuple[int, int] = (1, 5)  # Used for SCORE (inclusive)
  trailing_whitespace: bool = True  # Whether prompt template ends with ' '

  def __post_init__(self):
    if self.type == QuestionType.CHOICE:
      if not self.options or len(self.options) < 2:
        raise ValueError(
            f"CHOICE question {self.id!r} requires at least 2 options, got"
            f" {self.options}"
        )
      if len(self.options) > 26:
        raise ValueError(
            f"CHOICE question {self.id!r} supports at most 26 options in letter"
            f" readout, got {len(self.options)}"
        )
      if len(set(self.options)) != len(self.options):
        raise ValueError(
            f"CHOICE question {self.id!r} options must be unique, got"
            f" {self.options}"
        )
    elif self.type == QuestionType.SCORE:
      min_v, max_v = self.score_range
      if max_v <= min_v:
        raise ValueError(
            f"SCORE question {self.id!r} requires max_v > min_v in"
            f" score_range=(min_v, max_v), got {self.score_range}"
        )


@dataclasses.dataclass(frozen=True)
class NoulResult:
  """Result for a binary (noul) decision."""

  id: str
  value: bool
  confidence: float  # Calibrated normalized Shannon confidence [0.0, 1.0]
  probabilities: Dict[str, float]  # {"true": p_true, "false": p_false}
  raw_entropy: float


@dataclasses.dataclass(frozen=True)
class ChoiceResult:
  """Result for a categorical (choice) decision."""

  id: str
  value: str  # Selected option string
  selected_index: str  # Option label (e.g., 'A', 'B')
  confidence: float
  probabilities: Dict[str, float]  # Option string -> probability
  raw_entropy: float


@dataclasses.dataclass(frozen=True)
class ScoreResult:
  """Result for an ordinal (score) decision."""

  id: str
  value: int  # Argmax bucket score
  expected_value: float  # Continuous expectation: Sum(k * P(k))
  confidence: float
  probabilities: Dict[int, float]  # Score int -> probability
  raw_entropy: float


DecisionResult = Union[NoulResult, ChoiceResult, ScoreResult]


@dataclasses.dataclass(frozen=True)
class SystemOneResponse:
  """Aggregated response matching Jev /v1/systemone contract."""

  decisions: Dict[str, DecisionResult]
  latency_ms: float


# --- Jev Prompt Formatting Patterns ---


def format_noul_prompt(
    state: str,
    question: str,
    trailing_whitespace: bool = True,
) -> Tuple[str, List[str]]:
  """Standardizes on True / False target tokens."""
  prompt = (
      f"Context:\n{state}\n\nQuestion: {question}\nAnswer with True or"
      " False.\nAnswer:"
  )
  if trailing_whitespace:
    prompt += " "
  return prompt, ["True", "False"]


def format_choice_prompt(
    state: str,
    question: str,
    options: Sequence[str],
    trailing_whitespace: bool = True,
) -> Tuple[str, List[str]]:
  """Maps up to 26 options to single-letter index tokens.

  Labels: ['A', 'B', 'C', ...]
  """
  option_lines = [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)]
  options_block = "\n".join(option_lines)
  labels = [chr(65 + i) for i in range(len(options))]
  prompt = (
      f"Context:\n{state}\n\nQuestion: {question}\n{options_block}\nAnswer:"
  )
  if trailing_whitespace:
    prompt += " "
  return prompt, labels


def format_score_prompt(
    state: str,
    question: str,
    min_v: int,
    max_v: int,
    trailing_whitespace: bool = True,
) -> Tuple[str, List[str]]:
  """Maps discrete numeric ranges directly to digit tokens.

  Labels: ['1', '2', '3', '4', '5']
  """
  labels = [str(v) for v in range(min_v, max_v + 1)]
  prompt = (
      f"Context:\n{state}\n\nQuestion: {question}\nRate from {min_v} to"
      f" {max_v}.\nAnswer:"
  )
  if trailing_whitespace:
    prompt += " "
  return prompt, labels


def _format_noul_branch(
    question: str,
    trailing_whitespace: bool = True,
) -> Tuple[str, List[str]]:
  branch = f"Question: {question}\nAnswer with True or False.\nAnswer:"
  if trailing_whitespace:
    branch += " "
  return branch, ["True", "False"]


def _format_choice_branch(
    question: str,
    options: Sequence[str],
    trailing_whitespace: bool = True,
) -> Tuple[str, List[str]]:
  option_lines = [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)]
  options_block = "\n".join(option_lines)
  labels = [chr(65 + i) for i in range(len(options))]
  branch = f"Question: {question}\n{options_block}\nAnswer:"
  if trailing_whitespace:
    branch += " "
  return branch, labels


def _format_score_branch(
    question: str,
    min_v: int,
    max_v: int,
    trailing_whitespace: bool = True,
) -> Tuple[str, List[str]]:
  labels = [str(v) for v in range(min_v, max_v + 1)]
  branch = f"Question: {question}\nRate from {min_v} to {max_v}.\nAnswer:"
  if trailing_whitespace:
    branch += " "
  return branch, labels


# --- Tokenizer and Token Mapping Utilities ---


def _encode(tokenizer: Any, text: str, add_bos: bool = False) -> List[int]:
  """Encodes text across various tokenizer implementations."""
  try:
    return list(tokenizer.encode(text, add_bos=add_bos))
  except TypeError:
    pass

  try:
    tokens = list(tokenizer.encode(text))
  except Exception:  # pylint: disable=broad-exception-caught
    if hasattr(tokenizer, "EncodeAsIds"):
      tokens = list(tokenizer.EncodeAsIds(text))
    else:
      raise

  if add_bos:
    bos = None
    if hasattr(tokenizer, "bos_id"):
      bos = (
          tokenizer.bos_id() if callable(tokenizer.bos_id) else tokenizer.bos_id
      )
    elif hasattr(tokenizer, "special_tokens") and hasattr(
        tokenizer.special_tokens, "BOS"
    ):
      bos = tokenizer.special_tokens.BOS
    if bos is not None:
      tokens = [bos] + tokens
  return tokens


def _resolve_candidate_tokens(
    tokenizer: Any,
    candidates: Sequence[str],
    prefix_ends_with_space: bool = True,
) -> List[int]:
  """Maps candidate strings strictly to single token IDs in the vocabulary."""
  ids: List[int] = []
  for cand in candidates:
    found: Optional[int] = None
    if cand.startswith(" "):
      variants = (cand,)
    elif prefix_ends_with_space:
      variants = (cand, " " + cand)
    else:
      variants = (" " + cand, cand)

    for variant in variants:
      toks = _encode(tokenizer, variant, add_bos=False)
      if len(toks) == 1:
        found = toks[0]
        break

    if found is None:
      all_encs = {v: _encode(tokenizer, v, add_bos=False) for v in variants}
      raise ValueError(
          f"Candidate label {cand!r} does not map to a single token in the"
          f" tokenizer vocabulary (tested variants and encodings: {all_encs})."
      )
    ids.append(found)

  if len(set(ids)) != len(ids):
    raise ValueError(
        f"Candidate token IDs collide: {dict(zip(candidates, ids))}"
    )
  return ids


# --- Calibration & Entropy Mathematics ---


def calibrate_and_score(
    real_logits: jnp.ndarray,
    null_logits: Optional[jnp.ndarray] = None,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, float, float]:
  """Computes calibrated probabilities, Shannon entropy, and confidence."""
  # 1. Null Context Debias
  if null_logits is not None:
    calibrated_logits = real_logits - null_logits
  else:
    calibrated_logits = real_logits

  # 2. Subspace Softmax with Temperature
  probs = jax.nn.softmax(calibrated_logits / temperature, axis=-1)

  # 3. Shannon Entropy: H(P) = -Sum(p * log2(p))
  clipped_probs = jnp.clip(probs, 1e-12, 1.0)
  entropy = -jnp.sum(probs * jnp.log2(clipped_probs))

  # 4. Normalized Jev Confidence: 1.0 - (H(P) / log2(M))
  num_options = probs.shape[-1]
  if num_options > 1:
    max_entropy = math.log2(num_options)
    confidence = jnp.clip(1.0 - (entropy / max_entropy), 0.0, 1.0)
  else:
    confidence = 1.0

  if isinstance(entropy, jax.core.Tracer):
    return probs, entropy, confidence
  return probs, float(entropy), float(confidence)


# --- Tree Attention Packing Logic ---


def build_tree_attention_pack(
    state_ids: Sequence[int],
    questions_ids: Sequence[Sequence[int]],
    max_seq_len: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Constructs flattened tree inputs, positions, masks, and terminal indices.

  Args:
    state_ids: Token IDs of the shared context.
    questions_ids: List of token IDs for each independent question branch.
    max_seq_len: Optional maximum sequence length guard.

  Returns:
    input_ids: Shape [1, total_len]
    positions: Shape [1, total_len]
    attention_mask: Shape [1, 1, total_len, total_len] (Flax boolean format)
    terminal_indices: Shape [num_questions]
  """
  s_len = len(state_ids)
  q_lens = [len(q) for q in questions_ids]
  total_len = s_len + sum(q_lens)

  if max_seq_len is not None and total_len > max_seq_len:
    raise ValueError(
        f"Total packed sequence length {total_len} exceeds max_seq_len"
        f" {max_seq_len}"
    )

  # 1. Flatten Input IDs
  flat_ids = list(state_ids)
  for q in questions_ids:
    flat_ids.extend(q)
  input_ids = jnp.array([flat_ids], dtype=jnp.int32)

  # 2. Position IDs with Branching (each question branch restarts at s_len)
  positions_list = list(range(s_len))
  for q_len in q_lens:
    positions_list.extend(range(s_len, s_len + q_len))
  positions = jnp.array([positions_list], dtype=jnp.int32)

  # Validation confirming RoPE position tensors match expected branch offsets
  assert len(positions_list) == total_len, "Position list length mismatch"
  curr_chk = s_len
  for q_len in q_lens:
    assert positions_list[curr_chk : curr_chk + q_len] == list(
        range(s_len, s_len + q_len)
    ), f"RoPE branch offset mismatch at offset {curr_chk}"
    curr_chk += q_len

  # 3. 2D Attention Mask (total_len x total_len)
  mask = np.zeros((total_len, total_len), dtype=bool)

  # Causal mask for shared state
  mask[:s_len, :s_len] = np.tril(np.ones((s_len, s_len), dtype=bool))

  curr_offset = s_len
  terminal_indices = []
  for q_len in q_lens:
    branch_end = curr_offset + q_len
    # Attend to shared state
    mask[curr_offset:branch_end, :s_len] = True
    # Causal attention within branch
    mask[curr_offset:branch_end, curr_offset:branch_end] = np.tril(
        np.ones((q_len, q_len), dtype=bool)
    )
    terminal_indices.append(branch_end - 1)
    curr_offset = branch_end

  attention_mask = jnp.array(mask[None, None, :, :])
  terminal_indices = jnp.array(terminal_indices, dtype=jnp.int32)

  return input_ids, positions, attention_mask, terminal_indices


# --- SystemOneSampler Class ---


class SystemOneSampler:
  """Zero-shot non-generative Jev System One sampler for Gemma."""

  def __init__(
      self,
      model: Any,  # gm.nn.Gemma
      params: Any,  # PyTree
      tokenizer: Any,  # gm.text.Tokenizer
      default_temperature: float = 1.0,
      cache_null_priors: bool = True,
  ):
    self.model = model
    self.params = params
    self.tokenizer = tokenizer
    self.default_temperature = default_temperature
    self.cache_null_priors = cache_null_priors
    self._null_cache: Dict[Tuple[int, ...], jnp.ndarray] = {}

  def evaluate_noul(
      self,
      state: str,
      question: str,
      question_id: str = "q_noul",
      temperature: Optional[float] = None,
      calibrate: bool = True,
  ) -> NoulResult:
    """Evaluates a single binary (noul) question against state."""
    spec = QuestionSpec(
        id=question_id,
        text=question,
        type=QuestionType.NOUL,
    )
    response = self.evaluate_systemone(
        state=state,
        questions=[spec],
        temperature=temperature,
        calibrate=calibrate,
    )
    result = response.decisions[question_id]
    assert isinstance(result, NoulResult)
    return result

  def evaluate_choice(
      self,
      state: str,
      question: str,
      options: List[str],
      question_id: str = "q_choice",
      temperature: Optional[float] = None,
      calibrate: bool = True,
  ) -> ChoiceResult:
    """Evaluates a single categorical (choice) question against state."""
    spec = QuestionSpec(
        id=question_id,
        text=question,
        type=QuestionType.CHOICE,
        options=options,
    )
    response = self.evaluate_systemone(
        state=state,
        questions=[spec],
        temperature=temperature,
        calibrate=calibrate,
    )
    result = response.decisions[question_id]
    assert isinstance(result, ChoiceResult)
    return result

  def evaluate_score(
      self,
      state: str,
      question: str,
      score_range: Tuple[int, int] = (1, 5),
      question_id: str = "q_score",
      temperature: Optional[float] = None,
      calibrate: bool = True,
  ) -> ScoreResult:
    """Evaluates an ordinal (score) question against state."""
    spec = QuestionSpec(
        id=question_id,
        text=question,
        type=QuestionType.SCORE,
        score_range=score_range,
    )
    response = self.evaluate_systemone(
        state=state,
        questions=[spec],
        temperature=temperature,
        calibrate=calibrate,
    )
    result = response.decisions[question_id]
    assert isinstance(result, ScoreResult)
    return result

  def precompute_null_priors(
      self,
      questions: Sequence[QuestionSpec],
      null_state: str = "N/A",
  ) -> Dict[Tuple[int, ...], jnp.ndarray]:
    """Precomputes and caches null context priors for questions."""
    state_prefix = f"Context:\n{null_state}\n\n"
    state_ids = _encode(self.tokenizer, state_prefix, add_bos=True)

    questions_ids = []
    cand_ids_list = []
    for q in questions:
      if q.type == QuestionType.NOUL:
        branch_text, labels = _format_noul_branch(
            q.text, trailing_whitespace=q.trailing_whitespace
        )
      elif q.type == QuestionType.CHOICE:
        assert q.options is not None
        branch_text, labels = _format_choice_branch(
            q.text, q.options, trailing_whitespace=q.trailing_whitespace
        )
      elif q.type == QuestionType.SCORE:
        branch_text, labels = _format_score_branch(
            q.text,
            q.score_range[0],
            q.score_range[1],
            trailing_whitespace=q.trailing_whitespace,
        )
      else:
        raise ValueError(f"Unknown question type: {q.type}")

      q_ids = _encode(self.tokenizer, branch_text, add_bos=False)
      cand_ids = _resolve_candidate_tokens(
          self.tokenizer, labels, prefix_ends_with_space=q.trailing_whitespace
      )
      questions_ids.append(q_ids)
      cand_ids_list.append(cand_ids)

    max_seq_len = getattr(self.model, "max_seq_len", None) or getattr(
        getattr(self.model, "config", None), "max_seq_len", None
    )
    input_ids, positions, attention_mask, terminal_indices = (
        build_tree_attention_pack(
            state_ids, questions_ids, max_seq_len=max_seq_len
        )
    )

    logits = self._forward(input_ids, positions, attention_mask)

    for i, (q_ids, cand_ids) in enumerate(zip(questions_ids, cand_ids_list)):
      t_idx = int(terminal_indices[i])
      cand_indices = jnp.array(cand_ids, dtype=jnp.int32)
      null_subspace = logits[0, t_idx, cand_indices]
      self._null_cache[tuple(cand_ids)] = null_subspace
      self._null_cache[tuple(q_ids)] = null_subspace

    return self._null_cache

  def _forward(
      self,
      input_ids: jnp.ndarray,
      positions: jnp.ndarray,
      attention_mask: jnp.ndarray,
  ) -> jnp.ndarray:
    """Executes a single forward pass without autoregressive loop."""
    # Flax Transformer in gm.nn expects 3D mask [B, L, L] matching tokens [B, L]
    if attention_mask.ndim == 4:
      model_mask = jnp.squeeze(attention_mask, axis=1)
    else:
      model_mask = attention_mask

    try:
      out = self.model.apply(
          {"params": self.params},
          input_ids,
          positions=positions,
          attention_mask=model_mask,
      )
    except TypeError:
      out = self.model.apply(
          {"params": self.params},
          input_ids,
          positions=positions,
          mask=attention_mask,
      )

    logits = out.logits if hasattr(out, "logits") else out
    return logits

  def evaluate_systemone(
      self,
      state: str,
      questions: List[QuestionSpec],
      temperature: Optional[float] = None,
      calibrate: bool = True,
  ) -> SystemOneResponse:
    """Executes a single tree-attention forward pass for N mixed questions."""
    start_time = time.perf_counter()
    temp = temperature if temperature is not None else self.default_temperature

    if not questions:
      return SystemOneResponse(decisions={}, latency_ms=0.0)

    # 1. Tokenize shared state
    state_prefix = f"Context:\n{state}\n\n"
    state_ids = _encode(self.tokenizer, state_prefix, add_bos=True)

    # 2. Tokenize question branches and resolve candidates
    questions_ids: List[List[int]] = []
    cand_labels_list: List[List[str]] = []
    cand_ids_list: List[List[int]] = []

    for q in questions:
      if q.type == QuestionType.NOUL:
        branch_text, labels = _format_noul_branch(
            q.text, trailing_whitespace=q.trailing_whitespace
        )
      elif q.type == QuestionType.CHOICE:
        if q.options is None:
          raise ValueError(f"CHOICE question {q.id} missing options.")
        branch_text, labels = _format_choice_branch(
            q.text, q.options, trailing_whitespace=q.trailing_whitespace
        )
      elif q.type == QuestionType.SCORE:
        branch_text, labels = _format_score_branch(
            q.text,
            q.score_range[0],
            q.score_range[1],
            trailing_whitespace=q.trailing_whitespace,
        )
      else:
        raise ValueError(f"Unsupported QuestionType: {q.type}")

      q_ids = _encode(self.tokenizer, branch_text, add_bos=False)
      cand_ids = _resolve_candidate_tokens(
          self.tokenizer, labels, prefix_ends_with_space=q.trailing_whitespace
      )

      questions_ids.append(q_ids)
      cand_labels_list.append(labels)
      cand_ids_list.append(cand_ids)

    # 3. Build tree attention pack
    max_seq_len = getattr(self.model, "max_seq_len", None) or getattr(
        getattr(self.model, "config", None), "max_seq_len", None
    )
    input_ids, positions, attention_mask, terminal_indices = (
        build_tree_attention_pack(
            state_ids, questions_ids, max_seq_len=max_seq_len
        )
    )

    # 4. Single forward pass
    logits = self._forward(input_ids, positions, attention_mask)

    # 5. Terminal logit slicing, calibration, and result building
    decisions: Dict[str, DecisionResult] = {}

    for i, q in enumerate(questions):
      t_idx = int(terminal_indices[i])
      cand_ids = cand_ids_list[i]
      cand_indices = jnp.array(cand_ids, dtype=jnp.int32)

      # Subspace logit slice at terminal token position
      subspace_logits = logits[0, t_idx, cand_indices]

      # Context-free null debias if enabled
      null_logits = None
      if calibrate:
        cand_key = tuple(cand_ids)
        q_key = tuple(questions_ids[i])
        if cand_key in self._null_cache:
          null_logits = self._null_cache[cand_key]
        elif q_key in self._null_cache:
          null_logits = self._null_cache[q_key]

      probs, entropy, confidence = calibrate_and_score(
          subspace_logits, null_logits=null_logits, temperature=temp
      )

      if q.type == QuestionType.NOUL:
        p_true = float(probs[0])
        p_false = float(probs[1])
        decisions[q.id] = NoulResult(
            id=q.id,
            value=bool(p_true >= p_false),
            confidence=confidence,
            probabilities={"true": p_true, "false": p_false},
            raw_entropy=entropy,
        )

      elif q.type == QuestionType.CHOICE:
        assert q.options is not None
        probs_dict = {opt: float(p) for opt, p in zip(q.options, probs)}
        argmax_idx = int(jnp.argmax(probs))
        decisions[q.id] = ChoiceResult(
            id=q.id,
            value=q.options[argmax_idx],
            selected_index=cand_labels_list[i][argmax_idx],
            confidence=confidence,
            probabilities=probs_dict,
            raw_entropy=entropy,
        )

      elif q.type == QuestionType.SCORE:
        min_v, max_v = q.score_range
        values = jnp.arange(min_v, max_v + 1, dtype=jnp.float32)
        probs_f32 = jnp.asarray(probs, dtype=jnp.float32)
        expected_value = float(jnp.sum(values * probs_f32))
        argmax_idx = int(jnp.argmax(probs))
        val = int(values[argmax_idx])
        probs_dict = {int(v): float(p) for v, p in zip(values, probs_f32)}
        decisions[q.id] = ScoreResult(
            id=q.id,
            value=val,
            expected_value=expected_value,
            confidence=float(confidence),
            probabilities=probs_dict,
            raw_entropy=float(entropy),
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return SystemOneResponse(decisions=decisions, latency_ms=latency_ms)
