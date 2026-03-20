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
from shared.state import revision_status, revision_status_lock
from thumbnails.generator import build_revision_prompts, run_generation
from thumbnails.prompts import BORDER_PASS_PROMPT

revision_bp = Blueprint("revision", __name__, template_folder="templates")


@revision_bp.route("/revision")
@require_auth
def revision_index():
    return render_template("revision.html", git_version=GIT_VERSION)


@revision_bp.route("/revise_upload", methods=["POST"])
@require_auth
def revise_upload():
    with revision_status_lock:
        is_running = revision_status["running"]
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
    with revision_status_lock:
        revision_status["round_num"] = revision_status.get("round_num", 0) + 1
        round_num = revision_status["round_num"]
        revision_idea_idx = 0
        revision_status["ideas"] = [f"Revision page: {prompt[:120]}"]
        revision_status["episode_dir"] = episode_dir

    round_dir = os.path.join(episode_dir, f"round{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    custom_context = fields.get("revision_context_prompt", "").strip() or None
    prompts = build_revision_prompts(
        [base_img],
        revision_status.get("speakers", []),
        prompt,
        count_per=count,
        idea_idx=revision_idea_idx,
        attachment_refs=attachment_refs if attachment_refs else None,
        context_prompt=custom_context,
    )

    revision_status["add_border"] = fields.get("add_border") == "1"
    revision_status["border_prompt"] = fields.get("border_prompt", "").strip() or BORDER_PASS_PROMPT
    run_generation(client, prompts, round_dir, "revision_page", target_status=revision_status, target_lock=revision_status_lock)
    with revision_status_lock:
        base_src = "uploaded file" if base_files else "follow-up result"
        revision_status["log"].append(f"Revision base: {base_src}")
        revision_status["log"].append(f"Prompt: {prompt[:180]}")
        revision_status["log"].append(f"Extra refs attached: {len(files.get('revision_images', []))}")
        revision_status["log"].append(f"Requested attempts: {count}")

    return jsonify({"ok": True, "output_dir": round_dir, "count": len(prompts)})
