"""
vLLM-based chat LLM: drop-in replacement for HFChatLLM with faster inference.

Uses vLLM's batched inference engine for speed. Implements generate() and
generate_first_tactic_confidence() compatible with the BaseLLM interface.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import os
from typing import Any, Dict, List, Optional

from invpro.models.llm.base import BaseLLM

# Quiet vLLM and tqdm by default (progress bars and INFO logs break script output).
# Set VLLM_CONFIGURE_LOGGING=1 or TQDM_DISABLE=0 to re-enable.
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
os.environ.setdefault("TQDM_DISABLE", "1")

logger = logging.getLogger(__name__)

# Max tokens to generate when extracting first-tactic confidence (stop early for efficiency)
_FIRST_TACTIC_MAX_TOKENS = 256


def _vllm_gen_kwargs(llm) -> dict:
    """Pass use_tqdm=False when the vLLM generate() API supports it."""
    if hasattr(llm.generate, "__wrapped__") or "use_tqdm" in inspect.signature(llm.generate).parameters:
        return {"use_tqdm": False}
    return {}


def _logprob_from_prompt_logprob_dict(cell: Any, token_id: int) -> Optional[float]:
    """Single position from vLLM prompt_logprobs: dict[token_id] -> Logprob-like."""
    if cell is None or not isinstance(cell, dict):
        return None
    ent = cell.get(token_id)
    if ent is None:
        return None
    v = getattr(ent, "logprob", None)
    if v is None:
        v = getattr(ent, "log_prob", None)
    if v is None:
        return None
    return float(v)


class VLLMChatLLM(BaseLLM):
    """
    vLLM-backed chat LLM. Same interface as HFChatLLM but uses vLLM for faster inference.
    """

    def __init__(
        self,
        llm,  # vllm.LLM
        tokenizer,
        max_new_tokens: int = 8192,
        model_name_or_path: Optional[str] = None,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        use_chat_template: bool = True,
    ):
        """
        Args:
            llm: Pre-initialized vllm.LLM instance.
            tokenizer: Pre-loaded AutoTokenizer (for chat template).
            max_new_tokens: Max tokens to generate.
            model_name_or_path: Optional model id for cache keys.
            temperature: Sampling temperature.
            top_p: Nucleus sampling.
            top_k: Top-k sampling.
            use_chat_template: If False, treat prompt as plain completion text.
        """
        super().__init__()
        self._llm = llm
        self._tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self._model_name_or_path = model_name_or_path or ""
        self.temperature = temperature
        self.top_p = top_p if top_p is not None else 1.0
        self.top_k = top_k if top_k is not None else -1
        self.use_chat_template = bool(use_chat_template)

    def _prepare_prompt_text(self, prompt: str, *, add_generation_prompt: bool = True) -> str:
        """Build final prompt text for vLLM: chat-templated or raw completion."""
        if not self.use_chat_template:
            return prompt
        chat = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def get_model_cache_id(self) -> Optional[str]:
        """Stable hash for cache keys: backend class + model id (HF vs vLLM must not share cache)."""
        if not self._model_name_or_path:
            return None
        mode = "chat" if self.use_chat_template else "completion"
        blob = f"VLLMChatLLM:{mode}:{self._model_name_or_path}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def generate(self, prompt: str) -> str:
        """Run the model on a single user message and return the full decoded response."""
        from vllm import SamplingParams

        text = self._prepare_prompt_text(prompt, add_generation_prompt=True)
        sp = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if self.top_k and self.top_k > 0:
            sp.top_k = self.top_k
        outputs = self._llm.generate([text], sp, **_vllm_gen_kwargs(self._llm))
        out = outputs[0].outputs[0]
        return out.text

    def generate_with_metadata(self, prompt: str) -> Dict[str, Any]:
        """
        Run one generation and return text plus metadata from the raw model response.
        Metadata includes token count for the full raw response (before any parsing).
        """
        from vllm import SamplingParams

        text = self._prepare_prompt_text(prompt, add_generation_prompt=True)
        sp = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if self.top_k and self.top_k > 0:
            sp.top_k = self.top_k
        outputs = self._llm.generate([text], sp, **_vllm_gen_kwargs(self._llm))
        out = outputs[0].outputs[0]
        token_ids = getattr(out, "token_ids", None) or []
        return {
            "text": out.text,
            "num_generated_tokens": len(token_ids),
        }

    def generate_batch(self, prompt: str, batch_size: int) -> List[str]:
        """
        Run the model on the same prompt batch_size times in one vLLM batch; return
        list of decoded responses. Used by SinglePassProver.prove_batch for speed.
        """
        from vllm import SamplingParams

        text = self._prepare_prompt_text(prompt, add_generation_prompt=True)
        sp = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if self.top_k and self.top_k > 0:
            sp.top_k = self.top_k
        prompts = [text] * batch_size
        outputs = self._llm.generate(prompts, sp, **_vllm_gen_kwargs(self._llm))
        return [req_out.outputs[0].text for req_out in outputs]

    def generate_batch_with_metadata(self, prompt: str, batch_size: int) -> List[Dict[str, Any]]:
        """
        Batched generation with per-sample metadata from raw model outputs.
        Includes full response token count before proof extraction.
        """
        from vllm import SamplingParams

        text = self._prepare_prompt_text(prompt, add_generation_prompt=True)
        sp = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if self.top_k and self.top_k > 0:
            sp.top_k = self.top_k
        prompts = [text] * batch_size
        outputs = self._llm.generate(prompts, sp, **_vllm_gen_kwargs(self._llm))
        rows: List[Dict[str, Any]] = []
        for req_out in outputs:
            out = req_out.outputs[0]
            token_ids = getattr(out, "token_ids", None) or []
            rows.append({
                "text": out.text,
                "num_generated_tokens": len(token_ids),
            })
        return rows

    def generate_first_tactic_confidence(
        self,
        prompt: str,
        max_new_tokens: int = _FIRST_TACTIC_MAX_TOKENS,
        verbose: bool = False,
    ) -> float:
        """
        Generate with greedy decoding and logprobs; return mean prob of tokens
        in the first tactic line (same logic as HFChatLLM).
        """
        from vllm import SamplingParams

        _log = (lambda s: print(f"[first_tactic] {s}")) if verbose else (lambda s: None)
        text = self._prepare_prompt_text(prompt, add_generation_prompt=True)
        sampling = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            logprobs=1,
        )
        outputs = self._llm.generate([text], sampling, **_vllm_gen_kwargs(self._llm))
        req_out = outputs[0]
        comp = req_out.outputs[0]
        token_ids = comp.token_ids
        logprobs_list = comp.logprobs
        if not logprobs_list:
            _log("no logprobs returned")
            return 0.0

        tactic_probs: list[float] = []
        buffer = ""
        state = 0
        eos_id = self._tokenizer.eos_token_id

        for i, (tid, lp_dict) in enumerate(zip(token_ids, logprobs_list)):
            if tid == eos_id:
                _log(f"token #{i+1}: EOS, done")
                break
            if lp_dict is None:
                continue
            lp_entry = lp_dict.get(tid)
            if lp_entry is None:
                continue
            logprob = getattr(lp_entry, "logprob", None)
            if logprob is None:
                continue
            prob = math.exp(logprob)
            token_text = self._tokenizer.decode([tid], skip_special_tokens=False)
            buffer += token_text

            if state == 0:
                if ":= by" in buffer:
                    state = 1
                    _log(f"token #{i+1}: found ':= by', state 0->1")
            elif state == 1:
                if "\n" in token_text:
                    state = 2
                    _log(f"token #{i+1}: found newline after stmt, state 1->2")
            elif state == 2:
                stripped = token_text.strip()
                if stripped and stripped != "\n":
                    state = 3
                    tactic_probs.append(prob)
                    _log(f"token #{i+1}: first tactic token prob={prob:.4f}, state 2->3")
                    if "\n" in token_text:
                        _log(f"token #{i+1}: newline ends first tactic, done")
                        break
            elif state == 3:
                tactic_probs.append(prob)
                if "\n" in token_text:
                    _log(f"token #{i+1}: newline ends first tactic, done")
                    break

        if tactic_probs:
            avg = sum(tactic_probs) / len(tactic_probs)
            _log(f"done: {len(tactic_probs)} tactic tokens, avg_prob={avg:.4f}")
            return avg
        _log("done: no tactic tokens collected")
        return 0.0

    def get_log_probs_for_continuation(
        self,
        user_content_prefix: str,
        continuation: str,
        verbose: bool = False,
    ) -> List[float]:
        """
        Return log P(continuation token | prefix) for each scored theorem token, matching HFChatLLM.

        Uses vLLM ``prompt_logprobs`` on the chat-templated user message (prefix + continuation).
        Prefers ``prompt_logprobs=-1`` (full vocab per position) so log-probs match teacher-forced HF
        scoring; falls back to smaller top-k values if vLLM rejects -1 (``max_logprobs`` cap).
        """
        from vllm import SamplingParams

        _log = (lambda s: print(f"[log_probs_continuation] {s}")) if verbose else (lambda s: None)

        def _short(s: str, n: int = 100) -> str:
            t = s.replace("\n", "\\n")
            return t if len(t) <= n else t[: n - 3] + "..."

        if self.use_chat_template:
            chat_prefix = [{"role": "user", "content": user_content_prefix}]
            chat_full = [{"role": "user", "content": user_content_prefix + continuation}]
            prefix_str = self._tokenizer.apply_chat_template(
                chat_prefix,
                tokenize=False,
                add_generation_prompt=False,
            )
            full_str = self._tokenizer.apply_chat_template(
                chat_full,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            prefix_str = user_content_prefix
            full_str = user_content_prefix + continuation

        prefix_ids = self._tokenizer(prefix_str, return_tensors="pt").input_ids
        full_ids = self._tokenizer(full_str, return_tensors="pt").input_ids
        prefix_len = int(prefix_ids.shape[1])
        full_len = int(full_ids.shape[1])
        if verbose:
            _log(
                "backend=VLLMChatLLM | "
                f"prefix_chars={len(user_content_prefix)} cont_chars={len(continuation)} | "
                f"prefix_tail={_short(user_content_prefix[-120:])!r} | "
                f"theorem_head={_short(continuation[:120])!r}"
            )
            _log(f"token lens: prefix_len={prefix_len} full_len={full_len} (chat-templated user message)")
        if full_len <= prefix_len:
            _log(f"empty continuation: prefix_len={prefix_len} full_len={full_len}")
            return []

        cont_start = full_str.rfind(continuation)
        if cont_start == -1:
            num_continuation_tokens = full_len - prefix_len
            _log("continuation not found as substring; using all tokens after prefix")
        else:
            cont_end = cont_start + len(continuation)
            enc = self._tokenizer(full_str, return_offsets_mapping=True)
            offset_mapping = enc.get("offset_mapping")
            num_continuation_tokens = 0
            if offset_mapping:
                offsets = offset_mapping[0] if isinstance(offset_mapping[0], list) else offset_mapping
                for i in range(prefix_len, len(offsets)):
                    start, end = offsets[i]
                    if start < cont_end and end > cont_start:
                        num_continuation_tokens += 1
            else:
                num_continuation_tokens = full_len - prefix_len
            num_continuation_tokens = max(0, min(num_continuation_tokens, full_len - prefix_len))

        if verbose:
            _log(
                f"scoring {num_continuation_tokens} theorem token position(s) "
                f"(cont_substr_found={cont_start != -1})"
            )

        if num_continuation_tokens <= 0:
            _log("no continuation tokens to score (num_continuation_tokens=0)")
            return []

        full_ids_list = full_ids[0].tolist()

        gen_sig = inspect.signature(self._llm.generate)
        gen_kw = _vllm_gen_kwargs(self._llm)
        # Prefer token ids so positions align exactly with HF tokenization.
        use_prompt_token_ids = "prompt_token_ids" in gen_sig.parameters
        if verbose:
            _log(f"vLLM.generate: use_prompt_token_ids={use_prompt_token_ids} gen_kw={list(gen_kw.keys())}")

        # max_tokens>=1: some vLLM versions reject 0; one greedy token is discarded for scoring.
        prompt_logprob_candidates = (-1, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
        last_err: Optional[BaseException] = None
        outputs = None
        plp_used: Optional[int] = None
        for plp in prompt_logprob_candidates:
            try:
                sp = SamplingParams(
                    max_tokens=1,
                    temperature=0.0,
                    prompt_logprobs=plp,
                )
                if use_prompt_token_ids:
                    kwargs_call = dict(gen_kw)
                    kwargs_call["sampling_params"] = sp
                    kwargs_call["prompt_token_ids"] = [full_ids_list]
                    if "prompts" in gen_sig.parameters:
                        kwargs_call["prompts"] = None
                    outputs = self._llm.generate(**kwargs_call)
                else:
                    outputs = self._llm.generate([full_str], sp, **gen_kw)
                plp_used = plp
                break
            except Exception as e:
                last_err = e
                _log(f"prompt_logprobs={plp} failed: {e}")
                continue
        if outputs is None:
            logger.warning(
                "vLLM theorem-surprise scoring failed to build SamplingParams: %s", last_err
            )
            return []

        req = outputs[0]
        pls = getattr(req, "prompt_logprobs", None)
        pids = getattr(req, "prompt_token_ids", None)
        if verbose:
            _log(
                f"vLLM returned: prompt_logprobs_setting={plp_used!r} | "
                f"len(prompt_logprobs)={len(pls) if pls else 0} | "
                f"len(prompt_token_ids)={len(pids) if pids is not None else 'n/a'}"
            )
        if not pls:
            _log("no prompt_logprobs in output")
            return []

        if pids is not None and len(pids) != len(full_ids_list):
            _log(f"len(prompt_token_ids)={len(pids)} != len(tokenizer)={len(full_ids_list)}")
        if len(pls) < full_len:
            _log(f"len(prompt_logprobs)={len(pls)} < full_len={full_len}")

        log_probs: List[float] = []
        for i in range(num_continuation_tokens):
            idx = prefix_len + i
            if idx >= len(full_ids_list) or idx >= len(pls):
                break
            token_id = full_ids_list[idx]
            if pids is not None and idx < len(pids) and pids[idx] != token_id:
                _log(f"token id mismatch at {idx}: ours={token_id} vllm={pids[idx]}")
                token_id = pids[idx]
            cell = pls[idx]
            lp = _logprob_from_prompt_logprob_dict(cell, token_id)
            if lp is None:
                _log(f"missing logprob for token {token_id} at index {idx} (try prompt_logprobs=-1)")
                return []
            log_probs.append(lp)
            if verbose:
                tok = self._tokenizer.decode([token_id], skip_special_tokens=False)
                _log(
                    f"  cont token {i + 1}/{num_continuation_tokens}: "
                    f"id={token_id} {repr(tok)[:50]} log_p={lp:.4f}"
                )

        _log(
            f"returned {len(log_probs)} log probs, mean={sum(log_probs) / len(log_probs) if log_probs else 0:.4f}"
        )
        return log_probs

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        max_new_tokens: int = 8192,
        max_model_len: Optional[int] = 8192,
        max_num_seqs: int = 8,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        use_chat_template: bool = True,
        dtype: str = "bfloat16",
        **kwargs,
    ) -> "VLLMChatLLM":
        """Load model via vLLM and tokenizer from HuggingFace."""
        from transformers import AutoTokenizer
        from vllm import LLM

        # Suppress vLLM INFO logs (model load, inference progress, etc.)
        logging.getLogger("vllm").setLevel(logging.WARNING)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        llm_kwargs: dict = dict(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        llm = LLM(**llm_kwargs, **kwargs)
        return cls(
            llm=llm,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            model_name_or_path=model_name_or_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_chat_template=use_chat_template,
        )
