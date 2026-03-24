"""Global status dicts and threading locks for cross-thread state sharing.

NOTE: In-memory sessions require a single gunicorn worker (WEB_CONCURRENCY=1).
Multiple workers would each have their own sessions dict, breaking session continuity.
"""

import os
import threading
import time

MAX_LOG_ENTRIES = 200

if int(os.environ.get("WEB_CONCURRENCY", "1")) > 1:
    print("[THUMB] WARNING: In-memory sessions require WEB_CONCURRENCY=1. "
          "Multiple workers will break session state.")


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


# ---------------------------------------------------------------------------
# Session registry: each browser tab gets its own status dict + lock
# ---------------------------------------------------------------------------
sessions = {}           # {session_id: {"status": dict, "lock": Lock, "last_access": float}}
sessions_lock = threading.Lock()


def get_session(session_id):
    """Get or create a session's status dict and lock. Sweeps expired sessions."""
    with sessions_lock:
        now = time.time()
        # Sweep sessions idle > 2 hours that aren't running
        expired = [k for k, v in sessions.items()
                   if now - v["last_access"] > 7200 and not v["status"]["running"]]
        for k in expired:
            print(f"[THUMB] cleanup | expired session {k}")
            del sessions[k]
        # Get or create
        if session_id not in sessions:
            sessions[session_id] = {
                "status": _make_status(),
                "lock": threading.Lock(),
                "last_access": now,
            }
            print(f"[THUMB] new session | id={session_id}")
        sess = sessions[session_id]
        sess["last_access"] = now
        return sess["status"], sess["lock"]
