import re
from rwens.rewriting.utils import extract_all_declarations

def extract_state_text(result) -> str:
    """
    Extract state text from Lean LSP goal result.
    
    Args:
        result: The result from leanclient's get_goal method
        
    Returns:
        Formatted state text
    """
    if result is None:
        return ""

    goals = result.get("goals")
    rendered = result.get("rendered", "")

    if isinstance(goals, list) and len(goals) == 0:
        return rendered

    if isinstance(goals, list) and len(goals) > 0:
        # Join multiple goals with blank lines between them.
        return "\n\n".join(goals)

    # Fallback: prefer rendered if present.
    return rendered
