"""
Factory functions for creating canonicalization modules from configuration.
"""

from typing import Dict, Any, Optional

from rwens.canonicalization.base import CanonicalizationModule
from rwens.canonicalization.conf.types import (
    VariableRenamerConfig,
    IdentityModuleConfig,
    SimpModuleConfig,
    RewritingCanonicalizationConfig,
    CanonicalizationConfig,
)

_CONFIG_CLASSES = {
    "VariableRenamer": VariableRenamerConfig,
    "IdentityModule": IdentityModuleConfig,
    "SimpModule": SimpModuleConfig,
    "RewritingCanonicalization": RewritingCanonicalizationConfig,
}


def dict_to_config(config_dict: Dict[str, Any]) -> CanonicalizationConfig:
    """
    Convert a dictionary (from YAML) to a canonicalization config object.

    Args:
        config_dict: Dictionary with 'class' and 'parameters' keys
            Example: {
                "class": "VariableRenamer",
                "parameters": {
                    "project_root": ".",
                    "timeout_seconds": 90.0
                }
            }

    Returns:
        Configuration object instance
    """
    class_name = config_dict.get("class")
    parameters = config_dict.get("parameters", {})

    if class_name is None:
        raise ValueError("Config dictionary must have a 'class' key")

    if class_name == "VariableRenamer":
        return VariableRenamerConfig(**parameters)
    if class_name == "IdentityModule":
        return IdentityModuleConfig(**parameters)
    if class_name == "SimpModule":
        return SimpModuleConfig(**parameters)
    if class_name == "RewritingCanonicalization":
        return RewritingCanonicalizationConfig(**parameters)
    else:
        raise ValueError(
            f"Unknown canonicalization class: {class_name}. "
            f"Supported: {list(_CONFIG_CLASSES.keys())}"
        )


def canonicalization_from_config(
    config: CanonicalizationConfig,
    initial_imports: Optional[str] = None,
    llm: Optional[Any] = None,
) -> CanonicalizationModule:
    """
    Create a canonicalization module instance from config.

    Args:
        config: Configuration object (e.g., VariableRenamerConfig)
        initial_imports: Optional initial imports for the Lean file.
            Defaults to "import Mathlib\\n" if not provided.
        llm: Optional LLM instance required for RewritingCanonicalization.

    Returns:
        CanonicalizationModule instance
    """
    from rwens.canonicalization.renaming import VariableRenamer
    from rwens.canonicalization.identity import IdentityModule
    from rwens.canonicalization.simp import SimpModule
    from rwens.canonicalization.rewriting import RewritingCanonicalizationModule

    config_class_name = None
    for name, cls in _CONFIG_CLASSES.items():
        if isinstance(config, cls):
            config_class_name = name
            break

    if config_class_name == "VariableRenamer":
        return VariableRenamer(
            project_root=config.project_root,
            initial_imports=initial_imports or "import Mathlib\n",
            timeout_seconds=config.timeout_seconds,
        )
    if config_class_name == "IdentityModule":
        return IdentityModule(
            project_root=config.project_root,
            initial_imports=initial_imports or "import Mathlib\n",
            timeout_seconds=config.timeout_seconds,
        )
    if config_class_name == "SimpModule":
        return SimpModule(
            project_root=config.project_root,
            initial_imports=initial_imports or "import Mathlib\n",
            timeout_seconds=config.timeout_seconds,
        )
    if config_class_name == "RewritingCanonicalization":
        if llm is None:
            raise ValueError(
                "RewritingCanonicalization requires an LLM instance; "
                "pass llm= to canonicalization_from_config"
            )
        from rwens.canonicalization.rewrites import (
            cache_config_tag_from_energy,
            get_energy_heuristic,
            get_reranking_heuristic,
        )
        reranking_heuristic = get_reranking_heuristic(
            getattr(config, "reranking", None) or {}
        )
        energy_cfg = getattr(config, "energy", None)
        inject_energy_type = None
        if energy_cfg is None or (
            isinstance(energy_cfg, dict) and energy_cfg.get("type") == "none"
        ):
            energy_heuristic = None
        elif isinstance(energy_cfg, dict) and energy_cfg.get("type") == "theorem_surprise":
            energy_heuristic = None
            inject_energy_type = "theorem_surprise"
        else:
            energy_heuristic = get_energy_heuristic(
                energy_cfg or {}, project_root=config.project_root
            )
        cache_config_tag = cache_config_tag_from_energy(
            energy_cfg if isinstance(energy_cfg, dict) else None
        )
        sampling = getattr(config, "sampling", None) or {}
        max_per_step = sampling.get("max_per_step", 10)
        depth = sampling.get("depth", 2)
        only_simplifying_rewrites = sampling.get(
            "only_simplifying_rewrites",
            getattr(config, "only_simplifying_rewrites", False),
        )
        use_explicit_comm = sampling.get(
            "use_explicit_comm",
            getattr(config, "use_explicit_comm", False),
        )
        use_combined = sampling.get(
            "use_combined",
            getattr(config, "use_combined", False),
        )
        num_combined = sampling.get(
            "num_combined",
            getattr(config, "num_combined", 20),
        )
        reverse_order = sampling.get(
            "reverse_order",
            getattr(config, "reverse_order", False),
        )
        return RewritingCanonicalizationModule(
            project_root=config.project_root,
            llm=llm,
            initial_imports=initial_imports or "import Mathlib\n",
            timeout_seconds=config.timeout_seconds,
            top_rewrites=config.top_rewrites,
            max_per_step=max_per_step,
            depth=depth,
            reverse_order=reverse_order,
            filter_rewrite_namespaces=getattr(
                config, "filter_rewrite_namespaces", None
            ),
            namespace_blacklist=getattr(config, "namespace_blacklist", None),
            only_simplifying_rewrites=only_simplifying_rewrites,
            use_explicit_comm=use_explicit_comm,
            use_combined=use_combined,
            num_combined=num_combined,
            reranking_heuristic=reranking_heuristic,
            energy_heuristic=energy_heuristic,
            inject_energy_type=inject_energy_type,
            cache_config_tag=cache_config_tag,
        )
    else:
        raise ValueError(
            f"Unknown config type: {config_class_name}. "
            f"Supported: {list(_CONFIG_CLASSES.keys())}"
        )
