"""Basic HTTP auth decorator for Flask routes."""

from functools import wraps
from flask import request, Response
from config import APP_USER, APP_PASS


def require_auth(f):
    """Decorator that enforces Basic HTTP auth (skipped if APP_PASS is empty)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not APP_PASS:
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or auth.username != APP_USER or auth.password != APP_PASS:
            return Response(
                "Authentication required", 401,
                {"WWW-Authenticate": 'Basic realm="Doom Debates Thumbnail Generator"'},
            )
        return f(*args, **kwargs)
    return decorated
