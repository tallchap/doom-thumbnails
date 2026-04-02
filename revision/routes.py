"""Flask blueprint for the revision page (/revision)."""

import datetime
import io
import json
import os

from flask import Blueprint, jsonify, render_template, request
from PIL import Image

from auth import require_auth
from config import GIT_VERSION, THUMBNAILS_DIR
from shared.gemini_client import get_client, upload_files_from_bytes
from shared.helpers import parse_form_or_multipart
from shared.state import get_session
from thumbnails.generator import build_revision_prompts, run_generation
from thumbnails.prompts import BORDER_PASS_PROMPT

revision_bp = Blueprint("revision", __name__, template_folder="templates")


def _get_revision_session():
    """Extract session_id from query string and return (status_dict, lock).
    Must NOT access request.form — that consumes the body stream and breaks
    the custom multipart parser used by parse_form_or_multipart."""
    session_id = request.args.get("session_id") or "default"
    return get_session(session_id)


@revision_bp.route("/revision")
@require_auth
def revision_index():
    return render_template("revision.html", git_version=GIT_VERSION)


@revision_bp.route("/revise_upload", methods=["POST"])
@require_auth
def revise_upload():
    _st, _lk = _get_revision_session()
    with _lk:
        is_running = _st["running"]
    if is_running:
        return jsonify({"error": "Generation already in progress"})

    fields, files = parse_form_or_multipart(request)
    prompt = fields.get("prompt", "").strip()
    base_files = files.get("base_thumbnail", [])
    base_path = fields.get("base_path", "").strip()
    try:
        count = int(fields.get("count", "10"))
    except Exception:
        count = 10
    count = max(1, min(50, count))

    if not prompt:
        return jsonify({"error": "Revision prompt is required"})
    if not base_files and not base_path:
        return jsonify({"error": "Upload one base thumbnail image (or choose a prior output as follow-up base)"})

    base_img = None
    if base_files:
        try:
            base_img = Image.open(io.BytesIO(base_files[0])).convert("RGB")
        except Exception:
            return jsonify({"error": "Could not parse uploaded thumbnail image"})
    else:
        try:
            real = os.path.realpath(base_path)
            thumbs_real = os.path.realpath(THUMBNAILS_DIR)
            if not real.startswith(thumbs_real):
                return jsonify({"error": f"Follow-up base path outside thumbnails dir: {base_path}"})
            if not os.path.isfile(real):
                return jsonify({"error": f"Follow-up base image not found on disk: {base_path}"})
            base_img = Image.open(real).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Could not load follow-up base image: {str(e)[:200]}"})

    client = get_client()
    attachment_refs = upload_files_from_bytes(client, files.get("revision_images", []), "revision_img")

    episode_dir = os.path.join(THUMBNAILS_DIR, f"revision-page-{datetime.date.today().isoformat()}")
    with _lk:
        _st["round_num"] = _st.get("round_num", 0) + 1
        round_num = _st["round_num"]
        revision_idea_idx = 0
        _st["ideas"] = [f"Revision page: {prompt[:120]}"]
        _st["episode_dir"] = episode_dir

    round_dir = os.path.join(episode_dir, f"round{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    custom_context = fields.get("revision_context_prompt", "").strip() or None
    prompts = build_revision_prompts(
        [base_img],
        _st.get("speakers", []),
        prompt,
        count_per=count,
        idea_idx=revision_idea_idx,
        attachment_refs=attachment_refs if attachment_refs else None,
        context_prompt=custom_context,
    )

    _st["add_border"] = fields.get("add_border") == "1"
    _st["border_prompt"] = fields.get("border_prompt", "").strip() or BORDER_PASS_PROMPT
    run_generation(client, prompts, round_dir, "revision_page", target_status=_st, target_lock=_lk)
    with _lk:
        base_src = "uploaded file" if base_files else "follow-up result"
        _st["log"].append(f"Revision base: {base_src}")
        _st["log"].append(f"Prompt: {prompt[:180]}")
        _st["log"].append(f"Extra refs attached: {len(files.get('revision_images', []))}")
        _st["log"].append(f"Requested attempts: {count}")

    return jsonify({"ok": True, "output_dir": round_dir, "count": len(prompts)})
