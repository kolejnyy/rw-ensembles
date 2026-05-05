"""
invpro logger: file-based logging for use in workers and subprocesses.

Stdout/stderr are often redirected or captured (e.g. worker subprocesses).
This logger writes to a configurable file so debug output (e.g. hypothesis stack)
is visible regardless of process structure.

Usage:
  from invpro.logger import get_logger, set_log_file

  set_log_file(Path("invpro.log"))  # optional; defaults to INVPRO_LOG_FILE or invpro.log in cwd
  log = get_logger(__name__)
  log.info("hypothesis_stack: %s", stack)
"""

from invpro.logger.logger import get_logger, set_log_file

__all__ = ["get_logger", "set_log_file"]
