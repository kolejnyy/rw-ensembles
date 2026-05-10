"""
Standard cache layout: all cache files live under project_root/.cache with subfolders.

  .cache/
    rw/   - rewriting: rwcnc rewrites, get_states results, augmented problems
    conf/ - confidence/energy: first-tactic, theorem_surprise, step-by-step confidence

Previously used .rwcache and .confcache at project root; those are no longer written.
You can remove them or leave them; new data goes only to .cache/rw and .cache/conf.
"""

from pathlib import Path
from typing import Union

CACHE_DIR_NAME = ".cache"
SUBDIR_RW = "rw"
SUBDIR_CONF = "conf"


def get_cache_root(project_root: Union[str, Path]) -> Path:
    """Root directory for all caches: project_root/.cache."""
    return Path(project_root).resolve() / CACHE_DIR_NAME


def get_rw_cache_dir(project_root: Union[str, Path]) -> Path:
    """Rewriting cache: .cache/rw (rewrites, get_states, augmented problems)."""
    return get_cache_root(project_root) / SUBDIR_RW


def get_conf_cache_dir(project_root: Union[str, Path]) -> Path:
    """Confidence/energy cache: .cache/conf (first-tactic, theorem_surprise, etc.)."""
    return get_cache_root(project_root) / SUBDIR_CONF
