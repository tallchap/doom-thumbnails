"""Common helpers: API call recording, JSON serialization, multipart parsing."""

import datetime
import json
import urllib.parse

from shared.state import status, status_lock


def _serialize_for_debug(obj):
    """Best-effort serializer for prompt/API debug views."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)) or obj is None:
        return json.dumps(obj)
    if isinstance(obj, dict):
        try:
            return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)
    if isinstance(obj, (list, tuple)):
        parts = []
        for i, item in enumerate(obj, 1):
            parts.append(f"--- CONTENT[{i}] ---")
            parts.append(_serialize_for_debug(item))
        return "\n".join(parts)

    # Google File API / SDK objects (best effort)
    fields = {}
    for key in ("name", "display_name", "uri", "mime_type", "size_bytes"):
        if hasattr(obj, key):
            fields[key] = getattr(obj, key)
    if fields:
        return json.dumps({"file_ref": fields}, indent=2, ensure_ascii=False, default=str)

    return repr(obj)


def _record_api_call(model, contents, phase="", key="last_api_call", target_status=None, target_lock=None):
    """Store last outbound API call payload for UI inspection."""
    ts = datetime.datetime.now().isoformat()
    payload = _serialize_for_debug(contents)
    text = (
        f"timestamp: {ts}\n"
        f"phase: {phase or 'n/a'}\n"
        f"model: {model}\n"
        f"\n===== CONTENTS =====\n{payload}\n"
    )
    _st = target_status if target_status is not None else status
    _lk = target_lock if target_lock is not None else status_lock
    with _lk:
        _st[key] = text


def parse_multipart(headers, body):
    """Parse multipart/form-data body into fields and files."""
    content_type = headers.get("Content-Type", "")
    if "boundary=" not in content_type:
        return {}, {}

    boundary = content_type.split("boundary=")[1].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]
    boundary = boundary.encode()

    fields = {}
    files = {}

    parts = body.split(b"--" + boundary)
    for part in parts:
        part = part.strip()
        if not part or part == b"--":
            continue

        if b"\r\n\r\n" in part:
            header_block, content = part.split(b"\r\n\r\n", 1)
        elif b"\n\n" in part:
            header_block, content = part.split(b"\n\n", 1)
        else:
            continue

        if content.endswith(b"\r\n"):
            content = content[:-2]

        header_str = header_block.decode("utf-8", errors="replace")
        name = None
        filename = None
        for line in header_str.split("\n"):
            line = line.strip()
            if line.lower().startswith("content-disposition:"):
                if 'name="' in line:
                    name = line.split('name="')[1].split('"')[0]
                if 'filename="' in line:
                    filename = line.split('filename="')[1].split('"')[0]

        if name is None:
            continue

        if filename:
            if filename:
                files.setdefault(name, []).append(content)
        else:
            fields[name] = content.decode("utf-8", errors="replace")

    return fields, files


def parse_form_or_multipart(request):
    """Parse Flask request body as either multipart/form-data or url-encoded.
    Returns (fields_dict, files_dict)."""
    content_type = request.content_type or ""
    if "multipart/form-data" in content_type:
        # Use the raw multipart parser for compatibility with existing code
        body = request.get_data()
        return parse_multipart(dict(request.headers), body)
    else:
        body = request.get_data(as_text=True)
        fields = dict(urllib.parse.parse_qsl(body))
        return fields, {}
