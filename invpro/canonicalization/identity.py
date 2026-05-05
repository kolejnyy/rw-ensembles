"""
Identity canonicalization module.

Implements CanonicalizationModule with no transformations: augmented state
equals original state.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from invpro.canonicalization.base import SimpleCanonicalizationModule
from invpro.utils.applier import StateFetchAbort, GOAL_TIMEOUT_SECONDS, TacticApplier


class IdentityModule(SimpleCanonicalizationModule):
    """
    Canonicalization module that applies no transformations.

    Implements all CanonicalizationModule methods; augmented_state always equals
    current_state.
    """

    def __init__(
        self,
        project_root: str,
        initial_imports: str = "import Mathlib\n",
        timeout_seconds: float = GOAL_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            project_root=project_root,
            initial_imports=initial_imports,
            timeout_seconds=timeout_seconds,
            temp_suffix="identity",
        )

    def get_states(
        self,
        keep_augmentation: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (current_state, augmented_state).

        No transformations are applied: augmented_state equals current_state.
        """
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
