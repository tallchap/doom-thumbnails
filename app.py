#!/usr/bin/env python3
"""
Doom Debates Thumbnail Generator v2 — Flask Edition

Generates YouTube thumbnail candidates via Google Gemini image generation,
with a browser UI for idea generation, source image gathering, and iteration.

Usage:
    python app.py

Opens http://127.0.0.1:9200 in your browser.
"""

import os
import webbrowser

from flask import Flask, jsonify

from config import (
    PORT, APP_PASS, APP_USER, BRAVE_API_KEY, GIT_VERSION,
    GEMINI_MODEL, TEXT_MODEL, DESCRIPTION_MODEL,
    CLAUDE_DESCRIPTION_MODEL, GPT_DESCRIPTION_MODEL,
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    THUMBNAILS_DIR, FC_CAPTURES_DIR,
)


def create_app():
    app = Flask(__name__)

    # Ensure output directories exist
    os.makedirs(THUMBNAILS_DIR, exist_ok=True)
    os.makedirs(FC_CAPTURES_DIR, exist_ok=True)

    # Import and register blueprints
    from thumbnails.routes import thumbnails_bp
    from revision.routes import revision_bp
    from descriptions.routes import descriptions_bp
    from face_capture.routes import face_capture_bp

    app.register_blueprint(thumbnails_bp)
    app.register_blueprint(revision_bp)
    app.register_blueprint(descriptions_bp)
    app.register_blueprint(face_capture_bp)

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
