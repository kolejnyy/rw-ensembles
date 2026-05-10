"""Build rewriting modules from YAML-style dict config."""

from typing import Any, Dict, Optional

from rwens.rewriting.base import CanonicalizationModule
from rwens.rewriting.conf.types import CanonicalizationConfig, RewritingCanonicalizationConfig


def dict_to_config(config_dict: Dict[str, Any]) -> CanonicalizationConfig:
    """
    Convert a dictionary (from YAML) to a rewriting config object.

    Expects ``class: RewritingCanonicalization`` (legacy alias ``RewritingCanonicalizationConfig``).
    """
    class_name = config_dict.get("class")
    parameters = config_dict.get("parameters", {})

    if class_name is None:
        raise ValueError("Config dictionary must have a 'class' key")

    if class_name == "RewritingCanonicalization":
        return RewritingCanonicalizationConfig(**parameters)
    raise ValueError(
        f"Unknown rewriting config class: {class_name!r}. Supported: ['RewritingCanonicalization']"
    )


def canonicalization_from_config(
    config: CanonicalizationConfig,
    initial_imports: Optional[str] = None,
    llm: Optional[Any] = None,
) -> CanonicalizationModule:
    """
    Create a rewriting module instance from config.

    ``RewritingCanonicalization`` requires ``llm=``.
    """
    from rwens.rewriting.module import RewritingCanonicalizationModule
    from rwens.rewriting.rewrites import (
        cache_config_tag_from_energy,
        get_energy_heuristic,
        get_reranking_heuristic,
    )

    if not isinstance(config, RewritingCanonicalizationConfig):
        raise ValueError(f"Unsupported config type: {type(config)!r}")

    if llm is None:
        raise ValueError(
            "RewritingCanonicalization requires an LLM instance; pass llm= to canonicalization_from_config"
        )

    reranking_heuristic = get_reranking_heuristic(getattr(config, "reranking", None) or {})
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
        filter_rewrite_namespaces=getattr(config, "filter_rewrite_namespaces", None),
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
