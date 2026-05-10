"""
Base class for formal theorem provers.

This module defines the abstract base class for all formal theorem provers
in this project.
"""

from abc import ABC, abstractmethod
from typing import Optional


class FormalProver(ABC):
    """
    Abstract base class for formal theorem provers.
    
    All formal provers in this project should inherit from this class
    and implement the `generate` method.
    """
    
    @abstractmethod
    def generate(self, problem_statement: str) -> Optional[str]:
        """
        Generate a proof for the given problem statement.
        
        Args:
            problem_statement: The formal problem statement (e.g., Lean4 code)
            
        Returns:
            The generated proof as a string, or None if the proof generation failed
        """
        pass

    @abstractmethod
    def prove(self, problem_statement: str) -> dict:
        """
        Prove the given problem statement.
        
        Args:
            problem_statement: The formal problem statement (e.g., Lean4 code)
            
        Returns:
            A dictionary with the following keys:
            - success: bool, whether the proof was completed
            - final_code: str, the final code with all tactics applied
            - steps (optional): e.g. list of dicts with 'tactic', 'state_before', 'state_after'
            - error (optional): Optional[str], error message if failed
        """
        pass