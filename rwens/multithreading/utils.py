import threading
from typing import Optional

def run_with_timeout(fn, timeout: float) -> tuple[Optional[any], bool, Optional[BaseException]]:
    """
    Run a callable in a background thread with a hard timeout.
    
    Args:
        fn: Zero-argument callable to execute
        timeout: Maximum time to wait in seconds
        
    Returns:
        (result, timed_out, error):
        - result: the callable's return value (or None on timeout / error)
        - timed_out: True if the timeout was exceeded
        - error: any exception raised by fn, or None
    """
    result_holder = [None]
    error_holder = [None]

    def _runner():
        try:
            result_holder[0] = fn()
        except BaseException as e:
            error_holder[0] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        return None, True, None

    return result_holder[0], False, error_holder[0]
