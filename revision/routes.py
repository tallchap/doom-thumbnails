"""Flask blueprint for the revision page (/revision)."""

import datetime
import io
import json
import os
import threading

from flask import Blueprint, jsonify, render_template, request
from PIL import Image

from auth import require_auth
from config import GIT_VERSION, THUMBNAILS_DIR
from shared.gemini_client import get_primary_backend, get_all_backends, upload_files_from_bytes
from shared.drive_client import upload_episode_folder
from shared.helpers import parse_form_or_multipart, reset_generation_state, letter_for_index, safe_session_id
from shared.state import get_session
from thumbnails.generator import (
    apply_border_pillow, build_revision_prompts, run_generation,
    prepare_face_change, run_gpt_inpaint,
)


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


@revision_bp.route("/detect_faces", methods=["POST"])
@require_auth
def detect_faces():
    """Detect faces in an uploaded image using OpenCV."""
    try:
        import cv2
        import numpy as np
        fields, files = parse_form_or_multipart(request)
        img_bytes = files.get("image", [None])[0]
        if not img_bytes:
            return jsonify({"faces": []})
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        result = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
        return jsonify({"faces": result})
    except Exception as e:
        return jsonify({"faces": [], "error": str(e)[:200]})


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

    engine = (fields.get("engine", "gemini") or "gemini").strip().lower()
    if engine not in ("gemini", "openai"):
        return jsonify({"error": f"Unknown engine: {engine}"})
    if engine == "openai":
        from config import OPENAI_API_KEY
        if not OPENAI_API_KEY:
            return jsonify({"error": "OPENAI_API_KEY is not set on the server — the OpenAI engine is disabled."})

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

    # Face change preprocessing (optional)
    face_prompt = fields.get("face_change_prompt", "").strip()
    face_ref_files = files.get("face_ref_image", [])
    face_mask_files = files.get("face_mask_image", [])
    if face_prompt and not face_ref_files and not face_mask_files:
        return jsonify({"error": "Face change prompt was set but no new-face reference image "
                                 "and no painted mask were provided — attach a face reference "
                                 "(or paint a mask), or clear the face change prompt."})
    face_change_active = face_prompt and (face_ref_files or face_mask_files)
    face_changed_bases = []

    if face_change_active:
        from thumbnails.generator import prepare_face_change, async_face_changes
        import asyncio
        import time as _time

        base_buf = io.BytesIO()
        base_img.save(base_buf, "PNG")
        base_bytes = base_buf.getvalue()
        mask_data = face_mask_files[0] if face_mask_files else None
        ref_data = face_ref_files[0] if face_ref_files else None

        _st["log"].append(f"Face change: preprocessing image + mask...")
        thumb_bytes, mask_bytes, ref_bytes = prepare_face_change(base_bytes, face_prompt, ref_data, mask_data=mask_data)

        _fc_start = _time.time()
        _st["log"].append(f"Face change: firing {count} async OpenAI call(s) via asyncio.gather...")

        results = asyncio.run(async_face_changes(count, thumb_bytes, mask_bytes, ref_bytes, face_prompt))
        _fc_elapsed = _time.time() - _fc_start

        ok_count = 0
        for r in results:
            if isinstance(r, Exception):
                _st["log"].append(f"Face change failed: {str(r)[:150]}")
            else:
                idx, face_result = r
                face_changed_bases.append(Image.open(io.BytesIO(face_result)).convert("RGB"))
                ok_count += 1
        _st["log"].append(f"All {ok_count}/{count} face changes done in {_fc_elapsed:.1f}s → starting Gemini immediately")

    # Scope the output dir to this session so two browsers can't overwrite each other's files
    session_id = safe_session_id(request.args.get("session_id") or "default")
    episode_dir = os.path.join(THUMBNAILS_DIR, f"revision-page-{datetime.date.today().isoformat()}-{session_id}")
    with _lk:
        _st["round_num"] = _st.get("round_num", 0) + 1
        round_num = _st["round_num"]
        revision_idea_idx = 0
        _st["ideas"] = [f"Revision page [{engine}]: {prompt[:120]}"]
        _st["episode_dir"] = episode_dir

    round_dir = os.path.join(episode_dir, f"round{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    custom_context = fields.get("revision_context_prompt", "").strip() or None
    _st["add_border"] = fields.get("add_border") == "1"
    _st["logo_corner"] = fields.get("logo_corner", "bottom-left").strip() or "bottom-left"

    if engine == "openai":
        from thumbnails.generator import run_openai_revision
        bases = face_changed_bases if face_changed_bases else [base_img]
        base_bytes_list = []
        for im in bases:
            buf = io.BytesIO()
            im.save(buf, "PNG")
            base_bytes_list.append(buf.getvalue())
        # Face-changed variants get 1 revision each, matching the Gemini count_per=1 convention
        run_count = len(bases) if face_changed_bases else count
        run_openai_revision(
            base_bytes_list, prompt, run_count, round_dir,
            attachment_bytes_list=files.get("revision_images", []),
            context_prompt=custom_context,
            target_status=_st, target_lock=_lk,
        )
        with _lk:
            base_src = "uploaded file" if base_files else "follow-up result"
            _st["log"].append(f"Revision base: {base_src}")
            _st["log"].append(f"Prompt: {prompt[:180]}")
            _st["log"].append(f"Extra refs attached: {len(files.get('revision_images', []))}")
            _st["log"].append(f"Requested attempts: {run_count}")
        return jsonify({"ok": True, "output_dir": round_dir, "count": run_count})

    backends = get_all_backends()
    primary = backends[0]
    attachment_refs_by_backend = upload_files_from_bytes(files.get("revision_images", []), "revision_img")

    stored_speakers = _st.get("speakers")
    if isinstance(stored_speakers, dict):
        speaker_refs_by_backend = stored_speakers
    else:
        speaker_refs_by_backend = {"primary": list(stored_speakers or []), "secondary": []}

    if face_changed_bases:
        # Each face-changed variant gets 1 Gemini revision
        prompts = build_revision_prompts(
            backends,
            face_changed_bases,
            speaker_refs_by_backend,
            prompt,
            count_per=1,
            idea_idx=revision_idea_idx,
            attachment_refs_by_backend=attachment_refs_by_backend if any(attachment_refs_by_backend.values()) else None,
            context_prompt=custom_context,
        )
    else:
        prompts = build_revision_prompts(
            backends,
            [base_img],
            speaker_refs_by_backend,
            prompt,
            count_per=count,
            idea_idx=revision_idea_idx,
            attachment_refs_by_backend=attachment_refs_by_backend if any(attachment_refs_by_backend.values()) else None,
            context_prompt=custom_context,
        )

    run_generation(
        primary, prompts, round_dir, "revision_page",
        target_status=_st, target_lock=_lk,
        revision_mode=True,
    )
    with _lk:
        base_src = "uploaded file" if base_files else "follow-up result"
        _st["log"].append("Engine: Gemini")
        _st["log"].append(f"Revision base: {base_src}")
        _st["log"].append(f"Prompt: {prompt[:180]}")
        _st["log"].append(f"Extra refs attached: {len(files.get('revision_images', []))}")
        _st["log"].append(f"Requested attempts: {count}")

    return jsonify({"ok": True, "output_dir": round_dir, "count": len(prompts)})


@revision_bp.route("/border_only", methods=["POST"])
@require_auth
def border_only():
    """Apply border composite to an uploaded image — no AI, instant, $0."""
    fields, files = parse_form_or_multipart(request)
    base_files = files.get("base_thumbnail", [])
    base_path = fields.get("base_path", "").strip()
    logo_corner = fields.get("logo_corner", "bottom-left").strip() or "bottom-left"

    if not base_files and not base_path:
        return jsonify({"error": "Upload a thumbnail image"})

    try:
        if base_files:
            img = Image.open(io.BytesIO(base_files[0])).convert("RGB")
        else:
            real = os.path.realpath(base_path)
            thumbs_real = os.path.realpath(THUMBNAILS_DIR)
            if not real.startswith(thumbs_real):
                return jsonify({"error": "Path outside thumbnails dir"})
            if not os.path.isfile(real):
                return jsonify({"error": "Image not found on disk"})
            img = Image.open(real).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not load image: {str(e)[:200]}"})

    buf = io.BytesIO()
    img.save(buf, "PNG")
    bordered = apply_border_pillow(buf.getvalue(), logo_corner)

    session_id = safe_session_id(request.args.get("session_id") or "default")
    episode_dir = os.path.join(THUMBNAILS_DIR, f"revision-page-{datetime.date.today().isoformat()}-{session_id}")
    os.makedirs(episode_dir, exist_ok=True)
    out_path = os.path.join(episode_dir, f"border_only_{datetime.datetime.now().strftime('%H%M%S')}.png")
    with open(out_path, "wb") as f:
        f.write(bordered)

    _st, _lk = _get_revision_session()
    with _lk:
        idx = len(_st.get("images", [])) + 1
        _st.setdefault("images", []).append({"idx": idx, "path": out_path})
        _st["log"].append(f"Border-only applied (logo: {logo_corner})")

    return jsonify({"ok": True, "path": out_path, "idx": idx})


@revision_bp.route("/gpt_inpaint", methods=["POST"])
@require_auth
def gpt_inpaint():
    """Standalone GPT (gpt-image-2) inpainting for the Face Swap and Object Edits tabs.

    mode=face_swap : swap a face using a face reference and/or painted mask. If no
                     mask is painted, prepare_face_change auto-detects the face.
    mode=object    : revise the painted region per the prompt. A painted mask is
                     required; an optional reference image guides the result.

    Pure OpenAI — no Gemini pass. Always runs `count` calls in parallel.
    """
    from config import OPENAI_API_KEY

    _st, _lk = _get_revision_session()
    with _lk:
        if _st["running"]:
            return jsonify({"error": "Generation already in progress"})

    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY is not set on the server — GPT image edits are disabled."})

    fields, files = parse_form_or_multipart(request)
    mode = fields.get("mode", "object").strip()
    prompt = fields.get("prompt", "").strip()
    base_files = files.get("base_thumbnail", [])
    base_path = fields.get("base_path", "").strip()
    mask_files = files.get("mask_image", [])
    ref_files = files.get("ref_image", [])
    try:
        count = int(fields.get("count", "4"))
    except Exception:
        count = 4
    count = max(1, min(10, count))

    label = "Face swap" if mode == "face_swap" else "Object edit"

    if not prompt:
        return jsonify({"error": f"{label}: a prompt describing the change is required"})
    if not base_files and not base_path:
        return jsonify({"error": f"{label}: attach a base thumbnail first (Step 1)"})

    mask_data = mask_files[0] if mask_files else None
    ref_data = ref_files[0] if ref_files else None

    if mode == "object" and not mask_data:
        return jsonify({"error": "Object edit: paint a region on the canvas first, then Generate."})
    if mode == "face_swap" and not mask_data and not ref_data:
        return jsonify({"error": "Face swap: attach a new-face reference and/or paint a mask."})

    # Load the base image bytes (uploaded file or a prior result as follow-up base).
    try:
        if base_files:
            base_bytes = base_files[0]
            Image.open(io.BytesIO(base_bytes)).convert("RGB")  # validate
        else:
            real = os.path.realpath(base_path)
            thumbs_real = os.path.realpath(THUMBNAILS_DIR)
            if not real.startswith(thumbs_real):
                return jsonify({"error": f"Base path outside thumbnails dir: {base_path}"})
            if not os.path.isfile(real):
                return jsonify({"error": f"Base image not found on disk: {base_path}"})
            with open(real, "rb") as f:
                base_bytes = f.read()
    except Exception as e:
        return jsonify({"error": f"Could not load base image: {str(e)[:200]}"})

    try:
        thumb_bytes, mask_bytes, ref_bytes = prepare_face_change(
            base_bytes, prompt, ref_data, mask_data=mask_data
        )
    except Exception as e:
        return jsonify({"error": f"{label}: preprocessing failed — {str(e)[:200]}"})

    session_id = safe_session_id(request.args.get("session_id") or "default")
    episode_dir = os.path.join(THUMBNAILS_DIR, f"revision-page-{datetime.date.today().isoformat()}-{session_id}")
    with _lk:
        _st["round_num"] = _st.get("round_num", 0) + 1
        round_num = _st["round_num"]
        _st["ideas"] = [f"{label}: {prompt[:120]}"]
        _st["episode_dir"] = episode_dir
    round_dir = os.path.join(episode_dir, f"round{round_num}")

    run_gpt_inpaint(
        thumb_bytes, mask_bytes, ref_bytes, prompt, count, round_dir,
        target_status=_st, target_lock=_lk, label=label, mode=mode,
    )

    return jsonify({"ok": True, "output_dir": round_dir, "count": count})


@revision_bp.route("/revision/save_and_clear", methods=["POST"])
@require_auth
def save_and_clear():
    """Upload current revision results to Drive and clear generated state.

    Keeps the user's base image, prompt text, reference attachments, and
    count/border settings intact on the client side — we only touch server
    session state here.
    """
    _st, _lk = _get_revision_session()
    with _lk:
        if _st.get("running"):
            return jsonify({"error": "Cannot save — generation in progress"})
        images = list(_st.get("images", []))
        ideas = list(_st.get("ideas", []))

    if not images:
        return jsonify({"error": "Nothing to save — no results in session"})

    base_name = f"revision-{datetime.date.today().isoformat()}"

    # All revision results live under idea_idx=0 by convention. Sort by idx and
    # assign letters a, b, c, ..., z, aa, ab, ...
    imgs_sorted = sorted(
        [img for img in images if os.path.isfile(img.get("path", ""))],
        key=lambda x: x.get("idx", 0),
    )
    upload_list = []
    for letter_idx, img in enumerate(imgs_sorted):
        letter = letter_for_index(letter_idx)
        filename = f"idea1{letter}.png"
        upload_list.append((filename, img["path"]))

    if not upload_list:
        return jsonify({"error": "All result files are missing from disk"})

    lines = [
        f"Revision folder: {base_name}",
        f"Saved:           {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "Revision prompt(s):",
    ]
    for i, idea_text in enumerate(ideas, 1):
        lines.append(f"{i}. {idea_text}")
    text = "\n".join(lines) + "\n"

    session_id = request.args.get("session_id") or "default"

    # Reset session state synchronously so a follow-up /revise_upload can't race
    # the (slow) Drive upload — the upload uses the captured base_name/text/upload_list.
    reset_generation_state(_st, _lk)

    def _bg():
        try:
            result = upload_episode_folder(base_name, text, upload_list)
            status = f"✓ Saved to Drive: {result['folder_name']} ({result['file_count']} files) — {result['folder_url']}"
            print(f"[REVISION] save_and_clear | session={session_id} | folder={result['folder_name']} | files={result['file_count']}")
        except Exception as e:
            status = f"✗ Drive save failed: {str(e)[:200]}"
            print(f"[REVISION] save_and_clear FAILED | session={session_id} | {e}")
        with _lk:
            _st["log"].append(status)

    threading.Thread(target=_bg, daemon=True).start()
    return jsonify({"ok": True})
