"""
Generic HuggingFace chat LLM: one user message, full decoded response.

Used for models like DeepSeek-Prover-V2 that expect a single user prompt
and return a full response (e.g. proof plan + Lean 4 code). No internal
prompt formatting or answer extraction; the caller (e.g. SinglePassProver)
handles that.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

from invpro.models.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# Max tokens to stream when extracting first-tactic confidence (stop early for efficiency)
_FIRST_TACTIC_MAX_TOKENS = 256


class HFChatLLM(BaseLLM):
    """
    Load a HuggingFace causal LM and run chat-style generation.

    generate(prompt) treats prompt as the sole user message, applies the
    model's chat template, generates, and returns the raw decoded text.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 8192,
        model_name_or_path: Optional[str] = None,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        """
        Args:
            model: Pre-loaded AutoModelForCausalLM (or similar).
            tokenizer: Pre-loaded AutoTokenizer.
            max_new_tokens: Max tokens to generate (default 8192 for long proofs).
            model_name_or_path: Optional model id for cache keys.
            temperature: Sampling temperature (default 0.7). Higher = more random.
            top_p: If set, nucleus sampling (e.g. 0.9). Ignored if None.
            top_k: If set, top-k sampling. Ignored if None.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self._model_name_or_path = model_name_or_path or ""
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def get_model_cache_id(self) -> Optional[str]:
        """Stable hash for cache keys: backend class + model id (HF vs vLLM must not share cache)."""
        if not self._model_name_or_path:
            return None
        blob = f"HFChatLLM:{self._model_name_or_path}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def generate(self, prompt: str) -> str:
        """Run the model on a single user message and return the full decoded response."""
        chat = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.model.device)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            gen_kwargs["top_k"] = self.top_k
        with torch.inference_mode():
            out = self.model.generate(**gen_kwargs)
        # Decode only the generated part
        gen = out[:, inputs["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

    def generate_first_tactic_confidence(
        self,
        prompt: str,
        max_new_tokens: int = _FIRST_TACTIC_MAX_TOKENS,
        verbose: bool = False,
    ) -> float:
        """
        Stream generation token-by-token with greedy decoding. Return the average
        probability of tokens corresponding to the first non-empty line after the
        problem statement (first tactic).

        DeepSeek-Prover outputs the formal statement first, then the proof. We
        skip past the statement (find ':= by', then the first newline), skip empty
        lines, then collect tokens for the first tactic line until the next newline.
        """
        _log = (lambda s: print(f"[first_tactic] {s}")) if verbose else (lambda s: None)
        _t0 = time.perf_counter()
        chat = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        # State: 0=looking for ":= by", 1=looking for newline after stmt, 2=skipping empty lines, 3=collecting tactic
        state = 0
        buffer = ""
        tactic_probs: list[float] = []
        token_count = 0

        for _ in range(max_new_tokens):
            t_step = time.perf_counter()
            token_count += 1
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits.float(), dim=-1)
            next_id = logits.argmax(dim=-1, keepdim=True)
            prob = probs.gather(-1, next_id).squeeze(-1).item()
            t_step_done = time.perf_counter() - t_step

            if next_id.item() == self.tokenizer.eos_token_id:
                _log(f"token #{token_count}: EOS, done")
                break

            input_ids = torch.cat([input_ids, next_id], dim=1)
            token_text = self.tokenizer.decode(next_id[0], skip_special_tokens=False)
            token_repr = repr(token_text) if len(token_text) <= 40 else repr(token_text[:40]) + "..."
            buffer += token_text

            if state == 0:
                if ":= by" in buffer:
                    state = 1
                    _log(f"token #{token_count}: found ':= by', state 0->1")
            elif state == 1:
                if "\n" in token_text:
                    state = 2
                    _log(f"token #{token_count}: found newline after stmt, state 1->2")
            elif state == 2:
                stripped = token_text.strip()
                if stripped and stripped != "\n":
                    state = 3
                    tactic_probs.append(prob)
                    _log(f"token #{token_count}: first tactic token {token_repr} prob={prob:.4f}, state 2->3")
                    if "\n" in token_text:
                        _log(f"token #{token_count}: newline ends first tactic, done")
                        break
            elif state == 3:
                tactic_probs.append(prob)
                _log(f"token #{token_count}: tactic token {token_repr} prob={prob:.4f}")
                if "\n" in token_text:
                    _log(f"token #{token_count}: newline ends first tactic, done")
                    break

        t_total = time.perf_counter() - _t0
        if tactic_probs:
            avg = sum(tactic_probs) / len(tactic_probs)
            _log(f"done: {len(tactic_probs)} tactic tokens, avg_prob={avg:.4f}, total_time={t_total:.2f}s")
        else:
            _log(f"done: no tactic tokens collected, total_time={t_total:.2f}s")
        if not tactic_probs:
            return 0.0
        return sum(tactic_probs) / len(tactic_probs)

    def get_log_probs_for_continuation(
        self,
        user_content_prefix: str,
        continuation: str,
        verbose: bool = False,
    ) -> List[float]:
        """
        Return log P(continuation token | prefix) for each token in continuation.
        User message = user_content_prefix + continuation; uses chat template.
        One forward pass; returns list of log probabilities (one per continuation token).
        """
        _log = (lambda s: print(f"[log_probs_continuation] {s}")) if verbose else (lambda s: None)
        chat_prefix = [{"role": "user", "content": user_content_prefix}]
        chat_full = [{"role": "user", "content": user_content_prefix + continuation}]
        prefix_str = self.tokenizer.apply_chat_template(
            chat_prefix,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_str = self.tokenizer.apply_chat_template(
            chat_full,
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix_ids = self.tokenizer(
            prefix_str,
            return_tensors="pt",
        ).to(self.model.device).input_ids
        full_ids = self.tokenizer(
            full_str,
            return_tensors="pt",
        ).to(self.model.device).input_ids
        prefix_len = prefix_ids.shape[1]
        if full_ids.shape[1] <= prefix_len:
            _log(f"continuation empty or token boundary issue: prefix_len={prefix_len} full_len={full_ids.shape[1]}")
            return []
        # Count only tokens that span the continuation substring in full_str (exclude template suffix, e.g. assistant start)
        cont_start = full_str.rfind(continuation)
        if cont_start == -1:
            num_continuation_tokens = full_ids.shape[1] - prefix_len
            _log("continuation not found as substring in full_str, using all continuation positions")
        else:
            cont_end = cont_start + len(continuation)
            enc = self.tokenizer(full_str, return_offsets_mapping=True)
            offset_mapping = enc.get("offset_mapping")
            num_continuation_tokens = 0
            if offset_mapping:
                # Batch of 1: offset_mapping is list of (start,end) or list of lists
                offsets = offset_mapping[0] if isinstance(offset_mapping[0], list) else offset_mapping
                for i in range(prefix_len, len(offsets)):
                    start, end = offsets[i]
                    if start < cont_end and end > cont_start:
                        num_continuation_tokens += 1
            else:
                num_continuation_tokens = full_ids.shape[1] - prefix_len
            num_continuation_tokens = max(0, min(num_continuation_tokens, full_ids.shape[1] - prefix_len))
        with torch.inference_mode():
            outputs = self.model(input_ids=full_ids)
        logits = outputs.logits
        # logits[:, i, :] predicts token at position i+1
        log_probs: List[float] = []
        for i in range(num_continuation_tokens):
            pos = prefix_len - 1 + i
            token_id = full_ids[0, prefix_len + i].item()
            log_p = F.log_softmax(logits[0, pos, :].float(), dim=-1)[token_id].item()
            log_probs.append(log_p)
            if verbose:
                tok = self.tokenizer.decode([token_id], skip_special_tokens=False)
                _log(f"  cont token {i+1}/{num_continuation_tokens}: id={token_id} {repr(tok)[:50]} log_p={log_p:.4f}")
        _log(f"returned {len(log_probs)} log probs, mean={sum(log_probs)/len(log_probs) if log_probs else 0:.4f}")
        return log_probs

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        max_new_tokens: int = 8192,
        device_map: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> "HFChatLLM":
        """Load model and tokenizer from HuggingFace (or local path)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map or "auto",
            torch_dtype=torch_dtype or torch.bfloat16,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            model_name_or_path=model_name_or_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
