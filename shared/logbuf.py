"""Process-wide in-memory log capture.

Tees sys.stdout / sys.stderr and the root logger into one bounded ring buffer so
the /logs tab can show the full backend — every print("[THUMB]…")/[REVISION]…
line, every traceback, every request line emitted by app.py's after_request hook,
across all sessions and worker threads. On Cloud Run the same stdout/stderr is
also captured by Cloud Logging (the durable copy); this buffer is the zero-latency
live view that needs no auth dance / no GCP API.
"""

import collections
import datetime
import logging
import sys
import threading

_MAXLEN = 20000
_buf = collections.deque(maxlen=_MAXLEN)
_lock = threading.Lock()
_seq = 0
_installed = False
_tl = threading.local()  # re-entrancy guard for the tee


def _add(stream, text, level=None):
    global _seq
    text = text.rstrip("\r\n")
    if not text:
        return
    with _lock:
        _seq += 1
        _buf.append({
            "seq": _seq,
            "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "stream": stream,        # "out" | "err" | "log"
            "level": level,          # logging level name, or None
            "text": text,
        })


class _Tee:
    """File-like wrapper: writes through to the real stream AND line-buffers into _buf."""

    def __init__(self, real, name):
        self._real = real
        self._name = name
        self._partial = ""

    def write(self, s):
        try:
            self._real.write(s)
        except Exception:
            pass
        if getattr(_tl, "busy", False):
            return s and len(s) or 0
        try:
            _tl.busy = True
            self._partial += s
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                _add(self._name, line)
        except Exception:
            pass
        finally:
            _tl.busy = False
        return len(s) if s else 0

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._real.isatty()
        except Exception:
            return False

    def __getattr__(self, attr):
        return getattr(self._real, attr)


class _BufHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            try:
                msg = record.getMessage()
            except Exception:
                return
        for line in msg.split("\n"):
            _add("log", line, level=record.levelname)


def install():
    """Idempotently tee stdout/stderr and attach a buffer handler to the root logger."""
    global _installed
    if _installed:
        return
    _installed = True
    sys.stdout = _Tee(sys.stdout, "out")
    sys.stderr = _Tee(sys.stderr, "err")
    h = _BufHandler()
    h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(h)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    print("[LOGS] in-app log capture installed")


def snapshot(since_seq=0, limit=5000):
    """Return {lines, seq, dropped} — lines with seq > since_seq (capped at `limit`).

    `dropped` = True if the ring buffer rolled past where the client last was, so
    the client should reset its view.
    """
    with _lock:
        oldest = _buf[0]["seq"] if _buf else 0
        latest = _buf[-1]["seq"] if _buf else since_seq
        items = [e for e in _buf if e["seq"] > since_seq]
    dropped = since_seq > 0 and oldest > since_seq + 1
    if len(items) > limit:
        items = items[-limit:]
        dropped = True
    return {"lines": items, "seq": latest, "dropped": dropped, "total": len(_buf)}


def as_text():
    """The whole buffer as one plain-text blob (for the Download button)."""
    with _lock:
        rows = list(_buf)
    out = []
    for e in rows:
        tag = e["stream"] + (("/" + e["level"]) if e["level"] else "")
        out.append(f"{e['ts']} [{tag}] {e['text']}")
    return "\n".join(out) + "\n"
