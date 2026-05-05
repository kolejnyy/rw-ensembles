"""
SFT (Supervised Fine-Tuning) configuration utilities.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from invpro.training.conf.utils import (
    deep_merge,
    get_nested_value,
    load_yaml_config,
    resolve_path,
    validate_config,
)

from trl.trainer.sft_config import SFTConfig as TRLSFTConfig

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """
    Configuration for Supervised Fine-Tuning training.
    
    Attributes:
        model_name: Name or path of the base model
        train_dataset_path: Path to training dataset directory
        val_dataset_path: Path to validation dataset directory
        output_dir: Directory to save checkpoints and final model
        use_lora: Whether to use LoRA for efficient fine-tuning
        use_4bit: Whether to use 4-bit quantization
        max_seq_length: Maximum sequence length
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        warmup_steps: Number of warmup steps
        save_steps: Steps between checkpoints
        eval_steps: Steps between evaluations
        logging_steps: Steps between logging
        seed: Random seed
        system_prompt: System prompt to use (optional)
        eos_token: EOS token to use for training (should match chat template, e.g., "<|im_end|>" for Qwen)
    """
    model_name: str
    train_dataset_path: str
    val_dataset_path: str
    output_dir: str
    use_lora: bool = True
    use_4bit: bool = True
    max_seq_length: int = 2048
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    seed: int = 42
    system_prompt: Optional[str] = None
    eos_token: Optional[str] = None  # If None, will use tokenizer's default eos_token
    
    def __post_init__(self):
        """Set default values for optional fields."""
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    def to_trl_sft_config(self):
        """
        Convert this SFTConfig to TRL's SFTConfig instance.
        
        Returns:
            TRL's SFTConfig object (from trl.trainer.sft_config) with all relevant parameters from this config
        """

        return TRLSFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            bf16=True,  # Use bfloat16 instead of fp16 for better compatibility with modern models
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=self.warmup_steps,
            report_to="none",  # Disable TensorBoard logging
            seed=self.seed,
            dataloader_pin_memory=False,
            shuffle_dataset=True,
            # Conversational format: dataset has "prompt" and "completion" fields
            # TRL will automatically handle this format with completion_only_loss=True
            packing=False,
            max_length=self.max_seq_length,
            eos_token=self.eos_token,
        )


def load_sft_config(
    config_path: str | Path,
    base_config_path: Optional[str | Path] = None,
    config_base_dir: Optional[str | Path] = None,
) -> SFTConfig:
    """
    Load and parse SFT training configuration from a YAML file.
    
    First loads base_sft.yaml (required - contains all default parameters),
    then loads the specific config and merges them, with the specific config
    overriding base values. All default parameters must be defined in base_sft.yaml,
    not hardcoded in Python.
    
    Args:
        config_path: Path to the YAML configuration file (overrides base)
        base_config_path: Path to base config file (default: auto-detected base_sft.yaml)
        config_base_dir: Base directory for resolving relative paths in config (default: config file directory)
        
    Returns:
        SFTConfig object with parsed configuration
        
    Raises:
        FileNotFoundError: If base_sft.yaml is not found
        ValueError: If required configuration keys are missing
    """
    config_path = Path(config_path)
    
    # Find and load base config (required - contains all default parameters)
    if base_config_path is None:
        # Try to find base config in the same directory as the config file
        default_base = config_path.parent / "base_sft.yaml"
        if default_base.exists():
            base_config_path = default_base
        else:
            # Try configs/training/base_sft.yaml relative to project root
            # Path(__file__) is invpro/training/conf/sft.py
            # parents[0] = conf/, parents[1] = training/, parents[2] = invpro/ (package), parents[3] = invpro/ (project root)
            project_root = Path(__file__).resolve().parents[3]  # Go up from invpro/training/conf/sft.py to project root
            default_base = project_root / "configs" / "training" / "base_sft.yaml"
            if default_base.exists():
                base_config_path = default_base
    
    if base_config_path is None:
        # Calculate paths for error message (same path calculation)
        project_root = Path(__file__).resolve().parents[3]
        raise FileNotFoundError(
            f"Base config file (base_sft.yaml) not found. Expected one of:\n"
            f"  - {config_path.parent / 'base_sft.yaml'}\n"
            f"  - {project_root / 'configs' / 'training' / 'base_sft.yaml'}\n"
            f"Base config is required as it contains all default parameters."
        )
    
    base_config_path = Path(base_config_path)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    
    # Load base config (contains all default parameters)
    logger.info(f"Loading base config from {base_config_path}")
    base_config_dict = load_yaml_config(base_config_path)
    
    # Load specific config (contains overrides)
    logger.info(f"Loading config from {config_path}")
    specific_config_dict = load_yaml_config(config_path)
    
    # Merge configs (specific overrides base)
    config_dict = deep_merge(base_config_dict, specific_config_dict)
    
    # Set base directory for resolving relative paths (default to project root)
    # Paths in YAML are relative to project root, not config file directory
    if config_base_dir is None:
        # Find project root by walking up from base_config_path
        # Project root contains "configs" directory or "pyproject.toml"
        current = base_config_path.parent
        project_root = None
        
        # Walk up to find project root (look for configs/ directory or pyproject.toml)
        while current != current.parent:  # Stop at filesystem root
            # Check if this is the project root
            if (current / "configs").is_dir() or (current / "pyproject.toml").exists():
                project_root = current
                break
            current = current.parent
        
        if project_root is None:
            # Fallback: if not found, assume project root is parent of configs directory
            # If base_config_path is at configs/training/base_sft.yaml:
            # - parent = configs/training/
            # - parent.parent = configs/
            # - parent.parent.parent = project root
            project_root = base_config_path.parent.parent.parent
        
        config_base_dir = project_root
        logger.info(f"Resolving relative paths from project root: {config_base_dir}")
    
    # Validate required keys (all must be present in merged config)
    required_keys = [
        "model.name",
        "data.train_dataset_path",
        "data.val_dataset_path",
        "output.output_dir",
        "training.use_lora",
        "training.use_4bit",
        "training.max_seq_length",
        "training.batch_size",
        "training.gradient_accumulation_steps",
        "training.num_epochs",
        "training.learning_rate",
        "training.warmup_steps",
        "training.save_steps",
        "training.eval_steps",
        "training.logging_steps",
        "training.seed",
        "lora.r",
        "lora.alpha",
        "lora.dropout",
        "lora.target_modules",
    ]
    validate_config(config_dict, required_keys)
    
    # Extract all values directly from merged config and create SFTConfig
    # All defaults come from base_sft.yaml
    return SFTConfig(
        # Model configuration
        model_name=get_nested_value(config_dict, "model.name", required=True),
        # Data paths - resolve relative paths
        train_dataset_path=str(resolve_path(
            get_nested_value(config_dict, "data.train_dataset_path", required=True),
            config_base_dir
        )),
        val_dataset_path=str(resolve_path(
            get_nested_value(config_dict, "data.val_dataset_path", required=True),
            config_base_dir
        )),
        # Output path - resolve relative paths
        output_dir=str(resolve_path(
            get_nested_value(config_dict, "output.output_dir", required=True),
            config_base_dir
        )),
        # Training configuration - all values from merged config_dict (from base_sft.yaml)
        use_lora=get_nested_value(config_dict, "training.use_lora", required=True),
        use_4bit=get_nested_value(config_dict, "training.use_4bit", required=True),
        max_seq_length=get_nested_value(config_dict, "training.max_seq_length", required=True),
        batch_size=get_nested_value(config_dict, "training.batch_size", required=True),
        gradient_accumulation_steps=get_nested_value(config_dict, "training.gradient_accumulation_steps", required=True),
        num_epochs=get_nested_value(config_dict, "training.num_epochs", required=True),
        learning_rate=get_nested_value(config_dict, "training.learning_rate", required=True),
        warmup_steps=get_nested_value(config_dict, "training.warmup_steps", required=True),
        save_steps=get_nested_value(config_dict, "training.save_steps", required=True),
        eval_steps=get_nested_value(config_dict, "training.eval_steps", required=True),
        logging_steps=get_nested_value(config_dict, "training.logging_steps", required=True),
        seed=get_nested_value(config_dict, "training.seed", required=True),
        # LoRA configuration - all values from merged config_dict (from base_sft.yaml)
        lora_r=get_nested_value(config_dict, "lora.r", required=True),
        lora_alpha=get_nested_value(config_dict, "lora.alpha", required=True),
        lora_dropout=get_nested_value(config_dict, "lora.dropout", required=True),
        lora_target_modules=get_nested_value(config_dict, "lora.target_modules", required=True),
        # Prompt configuration (optional - can be None)
        system_prompt=get_nested_value(config_dict, "prompt.system_prompt", default=None),
        # EOS token configuration (optional - if None, uses tokenizer default)
        eos_token=get_nested_value(config_dict, "prompt.eos_token", default=None),
    )
