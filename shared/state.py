"""Global status dicts and threading locks for cross-thread state sharing."""

import threading

MAX_LOG_ENTRIES = 200


def _make_status():
    """Create a fresh status dict. Main page and revision page each get one."""
    return {
        "running": False,
        "phase": "idle",
        "total": 0,
        "completed": 0,
        "errors": 0,
        "log": [],
        "images": [],
        "done": False,
        "output_dir": "",
        "episode_dir": "",
        "speakers": [],
        "sources": [],
        "round_num": 0,
        "ideas": [],
        "idea_groups": {},
        "cost": 0.0,
        "session_cost": 0.0,
        "last_api_call": "",
        "last_border_api_call": "",
        "desc_calls": 0,
        "desc_input_chars": 0,
        "desc_output_chars": 0,
    }


def append_log(st, lock, message):
    """Thread-safe log append with cap to prevent memory growth."""
    with lock:
        st["log"].append(message)
        if len(st["log"]) > MAX_LOG_ENTRIES:
            st["log"] = st["log"][-MAX_LOG_ENTRIES:]


# Separate status dicts so main page and revision page can run simultaneously
main_status = _make_status()
main_status_lock = threading.Lock()
revision_status = _make_status()
revision_status_lock = threading.Lock()

# Legacy aliases for code that still references the old globals (descriptions page, etc.)
status = main_status
status_lock = main_status_lock

# Face capture has its own minimal status
fc_status = {"running": False, "log": [], "done": False, "output_dir": "", "result_dirs": {}}
