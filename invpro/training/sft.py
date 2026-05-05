"""
Supervised Fine-Tuning (SFT) training script for Qwen-Coder models.

This script implements standard HuggingFace SFTTrainer-style training
for state-to-tactic prediction using Qwen-Coder models.
"""

import logging
from pathlib import Path
from typing import Dict

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from peft import LoraConfig, TaskType

from invpro.prompt.tactic_prediction import BaseStateToTacPromptFormatter
from invpro.prompt.system import QWEN_CODER_SYSTEM_PROMPT
from invpro.training.conf import SFTConfig, load_sft_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_conversational_example(
    formatter: BaseStateToTacPromptFormatter,
    state: str,
    next_tactic: str,
    system_prompt: str = QWEN_CODER_SYSTEM_PROMPT,
) -> Dict:
    """
    Format a training example in conversational format for TRL's assistant_only_loss.
    
    Uses the prompt/completion format where:
    - "prompt" contains system and user messages
    - "completion" contains the assistant's response
    
    Args:
        formatter: The prompt formatter to use
        state: The proof state
        next_tactic: The correct next tactic (golden answer)
        system_prompt: System prompt to use
        
    Returns:
        Dictionary with "prompt" and "completion" fields in conversational format:
        {
            "prompt": [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
            "completion": [{"role": "assistant", "content": ...}]
        }
    """
    # Format the prompt (state to tactic question)
    prompt_content = formatter.format(state=state)
    
    # Format the answer with the tactic
    formatted_answer = formatter.format_answer(next_tactic)
    
    # Create conversational format as requested:
    # Prompt includes system and user messages (what the user provides)
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_content},
    ]
    
    # Completion includes only the assistant's response (what the model should generate)
    completion_messages = [
        {"role": "assistant", "content": formatted_answer},
    ]

    return {
        "prompt": prompt_messages,
        "completion": completion_messages,
    }


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    formatter: BaseStateToTacPromptFormatter,
    system_prompt: str = QWEN_CODER_SYSTEM_PROMPT,
) -> Dataset:
    """
    Preprocess dataset for SFT training in conversational format.
    
    Args:
        dataset: Dataset with 'state' and 'next_tactic' columns
        tokenizer: Tokenizer to use (kept for compatibility, but not used in formatting)
        formatter: Prompt formatter
        system_prompt: System prompt
        
    Returns:
        Preprocessed dataset with 'prompt' and 'completion' columns in conversational format
    """
    def process_example(example: Dict) -> Dict:
        """Process a single example into conversational format."""
        state = example["state"]
        next_tactic = example["next_tactic"]
        
        # Format in conversational format with separate prompt and completion
        conversational = format_conversational_example(
            formatter=formatter,
            state=state,
            next_tactic=next_tactic,
            system_prompt=system_prompt,
        )
        
        return conversational
    
    # Apply preprocessing
    # Remove all original columns (state and next_tactic) since we've converted them
    processed_dataset = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset to conversational format",
    )
    
    return processed_dataset


def train(config: SFTConfig):
    """
    Train a Qwen-Coder model using SFTTrainer.
    
    Args:
        config: SFTConfig object containing all training configuration
    """
    # Set random seed
    torch.manual_seed(config.seed)
    
    logger.info(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure quantization if using 4-bit
    quantization_config = None
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for compute dtype
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    # Load model
    # Use bfloat16 for consistent dtype matching with training config
    model_dtype = torch.bfloat16 if not config.use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    
    # Set tokenizer on model for SFTTrainer
    # In some TRL versions, SFTTrainer uses model.tokenizer if available
    model.tokenizer = tokenizer
    
    # Apply LoRA if requested
    peft_config = None
    if config.use_lora:
        logger.info("Configuring LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
    
    # Load datasets
    logger.info(f"Loading training dataset from {config.train_dataset_path}")
    train_dataset = load_from_disk(config.train_dataset_path)
    
    logger.info(f"Loading validation dataset from {config.val_dataset_path}")
    val_dataset = load_from_disk(config.val_dataset_path)
    
    # Initialize prompt formatter
    formatter = BaseStateToTacPromptFormatter()
    
    # Determine system prompt (use config or default)
    system_prompt = config.system_prompt if config.system_prompt is not None else QWEN_CODER_SYSTEM_PROMPT
    
    # Preprocess datasets
    logger.info("Preprocessing training dataset...")
    train_dataset = preprocess_dataset(
        train_dataset,
        tokenizer=tokenizer,
        formatter=formatter,
        system_prompt=system_prompt,
    )
    
    logger.info("Preprocessing validation dataset...")
    val_dataset = preprocess_dataset(
        val_dataset,
        tokenizer=tokenizer,
        formatter=formatter,
        system_prompt=system_prompt,
    )
    
    # Convert our SFTConfig to TRL's SFTConfig
    # This ensures compatibility with SFTTrainer which accepts TRL's SFTConfig
    logger.info("Converting config to TRL SFTConfig...")
    training_args = config.to_trl_sft_config()
    logger.info(f"Using TRL SFTConfig for training")
    
    # Initialize trainer
    logger.info("Initializing SFTTrainer...")
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
    }
    
    # Add optional parameters
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    
    trainer = SFTTrainer(**trainer_kwargs)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info("Training completed!")


def main():
    """Main entry point for SFT training script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Qwen-Coder model for state-to-tactic prediction using SFT."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config_path = Path(args.config)
    config = load_sft_config(config_path)
    
    # Start training
    train(config)


if __name__ == "__main__":
    main()
