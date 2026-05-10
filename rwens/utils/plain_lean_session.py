"""Minimal LSP-backed Lean session with no rewriting augmentation."""

from __future__ import annotations

from typing import Optional, Tuple

from rwens.rewriting.base import SimpleCanonicalizationModule
from rwens.utils.applier import GOAL_TIMEOUT_SECONDS, StateFetchAbort, TacticApplier


class PlainLeanSession(SimpleCanonicalizationModule):
    """
    Temp Lean file + LSP client where augmented state equals the current goal state.
    Used for round-trip checks and dataset tooling that do not apply rewrite augmentation.
    """

    def __init__(
        self,
        project_root: str,
        initial_imports: str = "import Mathlib\n",
        timeout_seconds: float = GOAL_TIMEOUT_SECONDS,
        *,
        temp_suffix: str = "plain",
    ) -> None:
        super().__init__(
            project_root=project_root,
            initial_imports=initial_imports,
            timeout_seconds=timeout_seconds,
            temp_suffix=temp_suffix,
        )

    def get_states(
        self,
        keep_augmentation: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        if self._applier is None or self._cached_state is None:
            self._applier = self._applier or TacticApplier(
                self.sfc, timeout_seconds=self.timeout_seconds
            )
            self._cached_state = self._applier._build_hypothesis_stack_from_code(
                reset_stack=True
            )
        state = self._cached_state
        if state is None:
            raise StateFetchAbort("state fetch timed out", is_timeout=True)
        return (state, state)
