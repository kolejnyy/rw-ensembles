"""
General configuration processing utilities.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def validate_config(config: Dict[str, Any], required_keys: list[str]) -> None:
    """
    Validate that required keys are present in the configuration.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key paths (e.g., ["model", "name"] for nested keys)
        
    Raises:
        ValueError: If any required key is missing
    """
    missing_keys = []
    
    for key_path in required_keys:
        keys = key_path.split(".")
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                missing_keys.append(key_path)
                break
            current = current[key]
    
    if missing_keys:
        raise ValueError(
            f"Missing required configuration keys: {', '.join(missing_keys)}"
        )


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None, required: bool = False) -> Any:
    """
    Get a value from a nested dictionary using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the key (e.g., "model.name")
        default: Default value to return if key is not found (only used if required=False)
        required: If True, raise ValueError if key is not found
        
    Returns:
        Value at the key path, or default if not found (and required=False)
        
    Raises:
        ValueError: If required=True and key is not found
    """
    keys = key_path.split(".")
    current = config
    
    for key in keys:
        if not isinstance(current, dict):
            if required:
                raise ValueError(f"Required configuration key not found: {key_path}")
            return default
        if key not in current:
            if required:
                raise ValueError(f"Required configuration key not found: {key_path}")
            return default
        current = current[key]
    
    return current


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """
    Resolve a path, optionally relative to a base directory.
    
    Args:
        path: Path to resolve (can be relative or absolute)
        base_dir: Base directory for relative paths (default: current working directory)
        
    Returns:
        Resolved absolute Path
    """
    path = Path(path)
    
    if path.is_absolute():
        return path.resolve()
    
    if base_dir is not None:
        base_dir = Path(base_dir).resolve()
        return (base_dir / path).resolve()
    
    return path.resolve()


def deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override_dict taking precedence.
    
    Nested dictionaries are merged recursively, while other values (including lists)
    are completely replaced.
    
    Args:
        base_dict: Base dictionary (defaults)
        override_dict: Override dictionary (specific values)
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override the value (or add if key doesn't exist)
            # This includes lists, which are completely replaced
            result[key] = value
    
    return result
