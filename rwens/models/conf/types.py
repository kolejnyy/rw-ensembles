"""
Configuration classes for formal provers.

This module defines classes that represent the configuration for different
types of formal provers.
"""

from pathlib import Path
from typing import Any, Union, Optional

from rwens.models.llm.conf.types import LLMConfig
from rwens.canonicalization.conf.types import CanonicalizationConfig

GOAL_TIMEOUT_SECONDS = 90.0


class NaiveStepByStepProverConfig:
    """Configuration for NaiveStepByStepProver.
    
    Args:
        llm_config: Configuration for the LLM model to use
        project_root: Root of the Lean project (where lakefile.lean lives)
        max_iterations: Maximum number of tactic steps before stopping (default: 100)
        timeout_seconds: Timeout for get_goal calls (default: GOAL_TIMEOUT_SECONDS)
    """
    llm_config: LLMConfig
    project_root: str
    max_iterations: int = 100
    timeout_seconds: float = GOAL_TIMEOUT_SECONDS
    
    def __init__(
        self,
        llm_config: LLMConfig,
        project_root: str,
        max_iterations: int = 100,
        timeout_seconds: float = GOAL_TIMEOUT_SECONDS,
        **kwargs: Any
    ) -> None:
        self.llm_config = llm_config
        self.project_root = project_root
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds


class CanonicalProverConfig:
    """Configuration for CanonicalProver.

    Args:
        llm_config: Configuration for the LLM model (trained on augmented states)
        canonicalization_config: Configuration for the canonicalization module
        full_aug: If True, augment state at each step; if False, only at start
        max_iterations: Maximum tactic steps (default: 100)
    """

    llm_config: LLMConfig
    canonicalization_config: CanonicalizationConfig
    full_aug: bool = False
    max_iterations: int = 100

    def __init__(
        self,
        llm_config: LLMConfig,
        canonicalization_config: CanonicalizationConfig,
        full_aug: bool = False,
        max_iterations: int = 100,
        **kwargs: Any
    ) -> None:
        self.llm_config = llm_config
        self.canonicalization_config = canonicalization_config
        self.full_aug = full_aug
        self.max_iterations = max_iterations

    @property
    def project_root(self) -> str:
        """Project root from canonicalization config."""
        return self.canonicalization_config.project_root


class CanonicalLLMProverConfig:
    """Configuration for CanonicalLLMProver (rewriting + single-pass).

    Args:
        rewriting_config: CanonicalizationConfig for RewritingCanonicalizationModule.
        single_pass_prover_config: Config for the single-pass prover (e.g. SinglePassProver).
        Energy type (e.g. theorem_surprise) is set under rewriting_config.parameters.energy.type.
    """
    rewriting_config: "CanonicalizationConfig"
    single_pass_prover_config: "SinglePassProverConfig"

    def __init__(
        self,
        rewriting_config: "CanonicalizationConfig",
        single_pass_prover_config: "SinglePassProverConfig",
        **kwargs: Any
    ) -> None:
        self.rewriting_config = rewriting_config
        self.single_pass_prover_config = single_pass_prover_config

    @property
    def project_root(self) -> str:
        return self.rewriting_config.project_root


class EnsembleLLMProverConfig:
    """Configuration for EnsembleLLMProver (multiple augmentations + single-pass)."""

    rewriting_config: "CanonicalizationConfig"
    single_pass_prover_config: "SinglePassProverConfig"
    ensemble_size: int = 4
    shuffle_seed: Optional[int] = None
    state_to_statement_mode: str = "naive"
    gpt_state_to_statement_model: str = "gpt-5.4-mini"
    gpt_state_api_key_env: str = "OPENAI_API_KEY"
    gpt_state_max_output_tokens: int = 300

    def __init__(
        self,
        rewriting_config: "CanonicalizationConfig",
        single_pass_prover_config: "SinglePassProverConfig",
        ensemble_size: int = 4,
        shuffle_seed: Optional[int] = None,
        state_to_statement_mode: str = "naive",
        gpt_state_to_statement_model: str = "gpt-5.4-mini",
        gpt_state_api_key_env: str = "OPENAI_API_KEY",
        gpt_state_max_output_tokens: int = 300,
        **kwargs: Any,
    ) -> None:
        self.rewriting_config = rewriting_config
        self.single_pass_prover_config = single_pass_prover_config
        self.ensemble_size = ensemble_size
        self.shuffle_seed = shuffle_seed
        self.state_to_statement_mode = state_to_statement_mode
        self.gpt_state_to_statement_model = gpt_state_to_statement_model
        self.gpt_state_api_key_env = gpt_state_api_key_env
        self.gpt_state_max_output_tokens = gpt_state_max_output_tokens

    @property
    def project_root(self) -> str:
        return self.rewriting_config.project_root


class SinglePassProverConfig:
    """Configuration for SinglePassProver (LLM generates full proof in one pass; Lean verifies).

    Args:
        llm_config: Configuration for the LLM (e.g. DeepSeek-Prover-V2)
        prompt_formatter_config: Config for the prompt formatter (e.g. DeepSeekProverPromptFormatter)
        project_root: Root of the Lean project
        timeout_seconds: Timeout for Lean diagnostics (default: GOAL_TIMEOUT_SECONDS)
        initial_imports: Preamble kept loaded by the verifier (default: "import Mathlib\\n")
    """
    llm_config: Any
    prompt_formatter_config: Any
    project_root: str
    timeout_seconds: float = GOAL_TIMEOUT_SECONDS
    initial_imports: Optional[str] = None

    def __init__(
        self,
        llm_config: Any,
        prompt_formatter_config: Any,
        project_root: str,
        timeout_seconds: float = GOAL_TIMEOUT_SECONDS,
        initial_imports: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self.llm_config = llm_config
        self.prompt_formatter_config = prompt_formatter_config
        self.project_root = project_root
        self.timeout_seconds = timeout_seconds
        self.initial_imports = initial_imports


# Create a single type for all prover configuration classes
ProverConfig = Union[
    NaiveStepByStepProverConfig,
    CanonicalProverConfig,
    CanonicalLLMProverConfig,
    EnsembleLLMProverConfig,
    SinglePassProverConfig,
]
