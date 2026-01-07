import os
import threading

_LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'system_busy.lock')
_internal_lock = threading.Lock()

# Original implementation used a simple boolean flag but did not work properly with multiple processes
# New implementation uses a lock file to ensure proper synchronization between processes
# i.e. if one process sets the flag to True, other processes will be able to see that it is True and
# wait for it to be cleared before proceeding
def set_system_busy(busy: bool):
    """Sets the system busy flag using a lock file."""
    with _internal_lock:    # Ensure only one thread can access the lock file at a time
        if busy:
            try:
                # Create file if it updates to busy
                with open(_LOCK_FILE, 'w') as f:
                    f.write("busy")
            except Exception:
                pass
        else:
            try:
                # Remove file if it updates to not busy
                if os.path.exists(_LOCK_FILE):
                    os.remove(_LOCK_FILE)
            except Exception:
                pass

def get_system_busy() -> bool:
    """Returns True if the system busy lock file exists."""
    return os.path.exists(_LOCK_FILE)
