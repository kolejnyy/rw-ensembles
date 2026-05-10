"""
Invariant Formal Provers - A research project on proof-invariant formal provers in Lean4.
"""

__version__ = "0.1.0"

# Fix encoding issues on Windows: ensure UTF-8 is used by default for file operations
# This prevents UnicodeDecodeError when leanclient reads files with non-ASCII characters
import os
import builtins

# Set environment variable for Python I/O encoding
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Monkey-patch built-in open() to default to UTF-8 encoding on Windows
# This fixes UnicodeDecodeError in leanclient when reading files without explicit encoding
_original_open = builtins.open

def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, 
                newline=None, closefd=True, opener=None):
    """Patched open() that defaults to UTF-8 encoding on Windows for text mode."""
    # Only patch text mode reads without explicit encoding (when leanclient reads files)
    if 'b' not in mode and encoding is None:
        encoding = 'utf-8'
        # Use 'replace' to handle any encoding issues gracefully
        if errors is None:
            errors = 'replace'
    return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

builtins.open = _utf8_open
