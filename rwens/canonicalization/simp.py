"""
Simp canonicalization module.

Applies "try simp at *" to simplify all hypotheses and the goal, then recovers
the augmented state. Structure similar to VariableRenamer but without renaming
maps: only this single tactic is applied for augmentation.
"""

from __future__ import annotations

from typing import Optional, Tuple

from rwens.canonicalization.base import SimpleCanonicalizationModule
from rwens.logger import get_logger
from rwens.utils.applier import TacticApplier, StateFetchAbort, GOAL_TIMEOUT_SECONDS

logger = get_logger(__name__)

_SIMP_TACTIC = "try simp at *"


class SimpModule(SimpleCanonicalizationModule):
    """
    Canonicalization module that augments the proof state by applying
    "try simp at *" and returning the resulting state.

    Current state = goal at EOF. Augmented state = goal after temporarily
    appending "try simp at *"; the tactic line is then removed unless
    keep_augmentation is True.
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
            temp_suffix="simp",
        )

    def get_states(
        self,
        keep_augmentation: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (current_state, augmented_state).

        Augmented state is the goal after applying "try simp at *" at the
        current proof position. The tactic is appended temporarily and
        removed unless keep_augmentation is True.
        """
        if self._applier is None or self._cached_state is None:
            self._applier = self._applier or TacticApplier(
                self.sfc, timeout_seconds=self.timeout_seconds
            )
            self._cached_state = self._applier._build_hypothesis_stack_from_code(
                reset_stack=True
            )
        current_state = self._cached_state
        if current_state is None:
            raise StateFetchAbort("state fetch timed out", is_timeout=True)

        try:
            current_file_content = self.sfc.get_file_content()
        except Exception as e:
            logger.warning("File client operation failed in get_states, recreating: %s", e)
            self._recreate_file_client()
            current_file_content = self.sfc.get_file_content()
        insert_start = self._eof_position(current_file_content)

        try:
            self._append_to_file("  " + _SIMP_TACTIC + "\n")
            aug_state = self._applier._build_hypothesis_stack_from_code(
                reset_stack=True
            )
            if aug_state is None:
                raise StateFetchAbort(
                    "state fetch timed out after simp", is_timeout=True
                )
            result = (current_state, aug_state)
        except StateFetchAbort:
            raise
        finally:
            if keep_augmentation:
                self._content = self.sfc.get_file_content()
                self._applier = TacticApplier(
                    self.sfc, timeout_seconds=self.timeout_seconds
                )
                self._cached_state = self._applier._build_hypothesis_stack_from_code(
                    reset_stack=True
                )
            else:
                try:
                    cur = self.sfc.get_file_content()
                except Exception as e:
                    logger.warning(
                        "File client operation failed in get_states cleanup, recreating: %s",
                        e,
                    )
                    self._recreate_file_client()
                    cur = self.sfc.get_file_content()
                cur_end = self._eof_position(cur)
                if cur_end != insert_start:
                    self._delete_range(insert_start, cur_end)
                    self._content = self.sfc.get_file_content()
                    self._cached_state = self._applier._build_hypothesis_stack_from_code(
                        reset_stack=True
                    )
        return result
