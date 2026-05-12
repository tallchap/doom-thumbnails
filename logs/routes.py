"""Flask blueprint for the in-app backend Logs tab (/logs).

Serves the process-wide ring buffer from shared.logbuf — every print/[THUMB]/
[REVISION] line, every traceback, every request line from app.py's after_request
hook, across all sessions and worker threads. Live view; durable history lives in
Cloud Logging (Cloud Run retains it).
"""

import time

from flask import Blueprint, Response, jsonify, render_template, request

from auth import require_auth
from config import GIT_VERSION
from shared import logbuf

logs_bp = Blueprint("logs", __name__, template_folder="templates")


@logs_bp.route("/logs")
@require_auth
def logs_index():
    return render_template("logs.html", git_version=GIT_VERSION)


@logs_bp.route("/logs/data")
@require_auth
def logs_data():
    try:
        since = int(request.args.get("since", "0"))
    except (TypeError, ValueError):
        since = 0
    # Short long-poll: hold open up to ~10s while nothing is new so the tail
    # updates promptly without the client hammering; return immediately once
    # there are new lines. (Client chains one request at a time, like /status.)
    snap = logbuf.snapshot(since)
    deadline = time.time() + 10.0
    while not snap["lines"] and not snap["dropped"] and time.time() < deadline:
        time.sleep(0.4)
        snap = logbuf.snapshot(since)
    return jsonify(snap)


@logs_bp.route("/logs/download")
@require_auth
def logs_download():
    return Response(
        logbuf.as_text(),
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=doom-thumbnails-log.txt"},
    )
