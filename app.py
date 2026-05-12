#!/usr/bin/env python3
"""
Doom Debates Thumbnail Generator v2 — Flask Edition

Generates YouTube thumbnail candidates via Google Gemini image generation,
with a browser UI for idea generation, source image gathering, and iteration.

Usage:
    python app.py

Opens http://127.0.0.1:9200 in your browser.
"""

import logging
import os
import time
import traceback
import webbrowser

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from config import (
    PORT, APP_PASS, APP_USER, BRAVE_API_KEY, GIT_VERSION,
    GEMINI_MODEL, TEXT_MODEL, DESCRIPTION_MODEL,
    CLAUDE_DESCRIPTION_MODEL, GPT_DESCRIPTION_MODEL,
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    THUMBNAILS_DIR, FC_CAPTURES_DIR,
)
from shared import logbuf

_req_log = logging.getLogger("doomthumb.request")

# High-frequency heartbeat/asset endpoints: don't emit a curated request line for
# their routine 200s (would drown the buffer) — but still log them when they fail.
# The raw gunicorn/werkzeug access log still records every one of them.
_QUIET_PATHS = {"/status", "/logs/data", "/last_api_call", "/last_border_api_call", "/image"}


def create_app():
    # Tee stdout/stderr + the root logger into the in-memory ring buffer that
    # backs the /logs tab. Must happen before blueprints/threads start so it
    # catches everything from boot on. Idempotent.
    logbuf.install()

    app = Flask(__name__)

    # Ensure output directories exist
    os.makedirs(THUMBNAILS_DIR, exist_ok=True)
    os.makedirs(FC_CAPTURES_DIR, exist_ok=True)

    # Import and register blueprints
    from thumbnails.routes import thumbnails_bp
    from revision.routes import revision_bp
    from descriptions.routes import descriptions_bp
    from face_capture.routes import face_capture_bp
    from logs.routes import logs_bp

    app.register_blueprint(thumbnails_bp)
    app.register_blueprint(revision_bp)
    app.register_blueprint(descriptions_bp)
    app.register_blueprint(face_capture_bp)
    app.register_blueprint(logs_bp)

    # ----- Request / failure logging: one line per response, every endpoint -----
    @app.before_request
    def _req_start():
        g._t0 = time.time()

    @app.after_request
    def _req_log_response(resp):
        try:
            dt_ms = (time.time() - getattr(g, "_t0", time.time())) * 1000.0
            sid = request.args.get("session_id") or "-"
            # Surface logical failures: many endpoints return {"error": ...} with HTTP 200.
            extra = ""
            ct = (resp.content_type or "")
            if (ct.startswith("application/json")
                    and not resp.direct_passthrough
                    and resp.status_code < 400):
                try:
                    data = resp.get_json(silent=True)
                    if isinstance(data, dict) and data.get("error"):
                        extra = f"  ERROR={str(data['error'])[:240]}"
                except Exception:
                    pass
            failed = resp.status_code >= 400 or bool(extra)
            if request.path in _QUIET_PATHS and not failed:
                return resp  # routine heartbeat — skip the curated line (access log still has it)
            line = (f"{request.method} {request.path} -> {resp.status_code} "
                    f"({dt_ms:.0f}ms) session={sid}{extra}")
            if failed:
                _req_log.warning(line)
            else:
                _req_log.info(line)
        except Exception:
            pass
        return resp

    @app.errorhandler(Exception)
    def _req_log_exception(e):
        if isinstance(e, HTTPException):
            return e  # 404/405/etc — rendered normally, logged by after_request
        try:
            _req_log.error(
                f"UNHANDLED {request.method} {request.path}: {e}\n{traceback.format_exc()}"
            )
        except Exception:
            pass
        return jsonify({"error": f"Internal error: {str(e)[:300]}"}), 500

    # Health check — no auth required (used by Render deploy monitoring)
    @app.route("/health")
    def health():
        return jsonify({"ok": True, "version": GIT_VERSION})

    return app


def main():
    # Gemini File API refs are now lazy-loaded on first generation request
    # (see shared/gemini_client.ensure_gemini_ready)

    print(f"Doom Debates Thumbnail Generator v2")
    print(f"Image Model: {GEMINI_MODEL}")
    print(f"Text Model: {TEXT_MODEL}")
    print(f"Description Model (Gemini): {DESCRIPTION_MODEL}")
    print(f"Description Model (Claude): {CLAUDE_DESCRIPTION_MODEL} {'[enabled]' if ANTHROPIC_API_KEY else '[disabled: no ANTHROPIC_API_KEY]'}")
    print(f"Description Model (GPT): {GPT_DESCRIPTION_MODEL} {'[enabled]' if OPENAI_API_KEY else '[disabled: no OPENAI_API_KEY]'}")
    print(f"Output: {THUMBNAILS_DIR}")
    print(f"Brave Search: {'enabled' if BRAVE_API_KEY else 'disabled (no BRAVE_API_KEY)'}")
    print(f"Server: http://0.0.0.0:{PORT}")
    if APP_PASS:
        print(f"Auth: enabled (user={APP_USER})")
    print(f"Gemini refs: will upload lazily on first generation request")
    print()

    app = create_app()

    if os.environ.get("NO_BROWSER") != "1":
        webbrowser.open(f"http://127.0.0.1:{PORT}")

    # Dev server — production uses gunicorn via Procfile
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
