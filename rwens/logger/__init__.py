"""
rwens logger: file-based logging for use in workers and subprocesses.

Stdout/stderr are often redirected or captured (e.g. worker subprocesses).
This logger writes to a configurable file so debug output (e.g. hypothesis stack)
is visible regardless of process structure.

Usage:
  from rwens.logger import get_logger, set_log_file

  set_log_file(Path("rwens.log"))  # optional; defaults to RWENS_LOG_FILE or rwens.log in cwd
  log = get_logger(__name__)
  log.info("hypothesis_stack: %s", stack)
"""

from rwens.logger.logger import get_logger, set_log_file

__all__ = ["get_logger", "set_log_file"]
