import hashlib
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from rwens.models.llm.base import BaseLLM
from rwens.models.llm.conf.types import QwenCoderS2TConfig
from rwens.prompt.system import QWEN_CODER_SYSTEM_PROMPT
from rwens.prompt.tactic_prediction import BaseStateToTacPromptFormatter
from rwens.prompt.base import PromptFormatter

logger = logging.getLogger(__name__)

class QwenCoderS2T(BaseLLM):
    """
    QwenCoder model for state-to-tactic prediction. Should work for different
    model sizes, which may come helpful during training.
    
    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        prompt_formatter: The prompt formatter to use.
        system_prompt: The system prompt to use.
    """
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt_formatter: PromptFormatter,
        system_prompt: str = QWEN_CODER_SYSTEM_PROMPT,
        base_model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.prompt_formatter = prompt_formatter
        self._base_model_name = base_model_name or ""
        self._checkpoint_path = checkpoint_path

    def get_model_cache_id(self) -> Optional[str]:
        """Hash of backend class, base model and checkpoint path for cache keys."""
        if not self._base_model_name and not self._checkpoint_path:
            return None
        blob = (
            "QwenCoderS2T\n"
            + self._base_model_name
            + "\n"
            + (self._checkpoint_path or "")
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def format_prompt(self, state: str) -> str:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_formatter.format(state)}
        ]
    
    def generate(self, state: str) -> str:
        messages = self.format_prompt(state)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self.prompt_formatter.extract_answer(response)

    def generate_greedy_with_confidence(
        self, state: str, max_new_tokens: int = 512, verbose: bool = False
    ) -> Tuple[str, List[float]]:
        """
        Generate next-tactic with greedy decoding (argmax per token) and return
        the extracted tactic plus the list of per-token probabilities of the chosen tokens.
        Aggregation (e.g. mean or log-mean) is left to the caller (e.g. energy heuristic).

        When verbose=True, prints per-token id, decoded token, prob and log_p (same style
        as theorem_surprise log_probs_continuation) for debugging.
        """
        _log = (lambda s: print(f"[confidence_continuation] {s}")) if verbose else (lambda s: None)
        messages = self.format_prompt(state)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_ids = model_inputs.input_ids
        prompt_len = input_ids.shape[1]
        probs_list: List[float] = []
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits.float(), dim=-1)
            next_token = logits.argmax(dim=-1, keepdim=True)
            prob_chosen = probs.gather(-1, next_token).squeeze(-1).item()
            probs_list.append(prob_chosen)
            if verbose:
                tid = next_token.item()
                decoded = self.tokenizer.decode([tid])
                log_p = math.log(max(prob_chosen, 1e-10))
                _log(
                    f"  cont token {len(probs_list)}: id={tid} {repr(decoded)} "
                    f"prob={prob_chosen:.4f} log_p={log_p:.4f}"
                )
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                if verbose:
                    _log(f"  EOS at token {len(probs_list)}, stopping")
                break
        if verbose and probs_list:
            mean_p = sum(probs_list) / len(probs_list)
            _log(f"returned {len(probs_list)} token probs, mean(prob)={mean_p:.4f}")
        generated_ids = input_ids[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tactic = self.prompt_formatter.extract_answer(response)
        if verbose:
            _log(f"extracted tactic (first 200 chars): {repr((tactic or '')[:200])}")
        return tactic, probs_list

    def generate_greedy_with_confidence_batch(
        self, states: List[str], max_new_tokens: int = 32, pad_to_multiple_of: int = 8
    ) -> List[Tuple[str, List[float]]]:
        """
        Batched greedy generation. Returns list of (tactic, per-token_probabilities)
        in same order as states. Aggregation is left to the caller (e.g. energy heuristic).
        """
        if not states:
            return []
        batch_size = len(states)
        texts = []
        for state in states:
            messages = self.format_prompt(state)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)
        tokenizer_kw = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self.model.config, "max_position_embeddings", 32768),
            return_attention_mask=True,
        )
        if pad_to_multiple_of and pad_to_multiple_of > 0:
            tokenizer_kw["pad_to_multiple_of"] = pad_to_multiple_of
        enc = self.tokenizer(texts, **tokenizer_kw).to(self.model.device)
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask.clone()
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        probs_per_seq = [[] for _ in range(batch_size)]
        done = [False] * batch_size
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits.float(), dim=-1)
            next_tokens = logits.argmax(dim=-1)
            for i in range(batch_size):
                if not done[i]:
                    probs_per_seq[i].append(probs[i, next_tokens[i]].item())
                    if next_tokens[i].item() == eos_id:
                        done[i] = True
            if all(done):
                break
            next_tokens = next_tokens.unsqueeze(1)
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device),
            ], dim=1)

        results = []
        for i in range(batch_size):
            start = prompt_lengths[i]
            generated_ids = input_ids[i, start:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            tactic = self.prompt_formatter.extract_answer(response)
            results.append((tactic, probs_per_seq[i]))
        return results

    @classmethod
    def from_config(
        cls,
        config: QwenCoderS2TConfig,
        prompt_formatter: PromptFormatter,
    ):
        """
        Create a QwenCoderS2T instance from a configuration object.
        
        If config.checkpoint is provided, loads from checkpoint (supports LoRA adapters
        and full checkpoints). Otherwise, loads the base model from HuggingFace.
        
        Args:
            config: QwenCoderS2TConfig configuration object
            prompt_formatter: The prompt formatter to use
            
        Returns:
            QwenCoderS2T instance
        """
        # Load tokenizer from base model
        logger.info(f"Loading tokenizer from {config.base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        checkpoint_path_str = getattr(config, "checkpoint", None)
        checkpoint_path = Path(checkpoint_path_str) if checkpoint_path_str else None
        if checkpoint_path is not None:
            # Load from checkpoint
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            
            # Configure quantization if using 4-bit
            quantization_config = None
            if config.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            
            # Load base model
            logger.info(f"Loading base model from {config.base_model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if not config.use_4bit else None,
                trust_remote_code=True,
            )
            
            # Check if this is a LoRA checkpoint or full model checkpoint
            adapter_config = checkpoint_path / "adapter_config.json"
            
            if adapter_config.exists():
                logger.info(f"Detected LoRA checkpoint. Loading adapters from {checkpoint_path}...")
                model = PeftModel.from_pretrained(model, str(checkpoint_path))
                logger.info("Merging LoRA weights for inference...")
                model = model.merge_and_unload()  # Merge LoRA weights for inference
            else:
                # Check if it's a standard HuggingFace checkpoint
                config_json = checkpoint_path / "config.json"
                if config_json.exists():
                    logger.info(f"Loading full model checkpoint from {checkpoint_path}...")
                    # Replace the base model with the checkpoint
                    model = AutoModelForCausalLM.from_pretrained(
                        str(checkpoint_path),
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.bfloat16 if not config.use_4bit else None,
                        trust_remote_code=True,
                    )
                else:
                    raise ValueError(
                        f"Checkpoint path {checkpoint_path} does not appear to be a valid "
                        "LoRA adapter or full model checkpoint. Missing adapter_config.json or config.json"
                    )
            
            model.eval()
            logger.info("Model loaded successfully!")
            return cls(
                model,
                tokenizer,
                prompt_formatter,
                base_model_name=config.base_model_name,
                checkpoint_path=str(checkpoint_path),
            )
        else:
            # Load base model directly from HuggingFace
            logger.info(f"Initializing {cls.__name__} with model {config.base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            return cls(
                model,
                tokenizer,
                prompt_formatter,
                base_model_name=config.base_model_name,
                checkpoint_path=None,
            )


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    prompt_formatter = BaseStateToTacPromptFormatter()
    config = QwenCoderS2TConfig(base_model_name=model_name)
    model = QwenCoderS2T.from_config(config, prompt_formatter)
    state = "x y z : ℚ\nhx : x ≠ -1\nhy : y ≠ -2\nhz : z ≠ -3\nh : 2015 / (x + 1) + 2015 / (y + 2) + 2015 / (z + 3) = 2014\n⊢ x + 1 ≠ 0"
    response = model.generate(state)
    print(response)