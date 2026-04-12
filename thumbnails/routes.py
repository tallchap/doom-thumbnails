"""Flask blueprint for the main thumbnail generation page (/)."""

import datetime
import json
import os
import re
import shutil
import subprocess
import time

from flask import Blueprint, jsonify, render_template, request, send_file, Response
from PIL import Image

from auth import require_auth
from config import (
    APP_MODE, BRAVE_API_KEY, GIT_VERSION, THUMBNAILS_DIR,
)
from shared.gemini_client import get_primary_backend, get_all_backends, upload_files_from_bytes
from shared.drive_client import upload_episode_folder
from shared.helpers import parse_form_or_multipart, reset_generation_state, letter_for_index
from shared.state import get_session
from thumbnails.brave_search import search_images_brave, download_image_bytes
from thumbnails.generator import (
    generate_ideas, generate_search_queries,
    build_idea_prompts, build_variation_prompts, build_revision_prompts,
    run_generation, save_metadata,
)
from thumbnails.prompts import BORDER_PASS_PROMPT, REVISION_PROMPT, REVISION_CONTEXT_PROMPT

thumbnails_bp = Blueprint("thumbnails", __name__, template_folder="templates")


def _get_session():
    """Extract session_id from request and return (status_dict, lock)."""
    session_id = request.args.get("session_id") or request.form.get("session_id") or "default"
    return get_session(session_id)


def _normalize_backend_refs(refs):
    """Accept either a per-backend dict (new) or a bare list (legacy/None) and
    return a per-backend dict {'primary': [...], 'secondary': [...]}.

    Legacy bare lists are treated as primary-only since they were uploaded before
    multi-backend support and have no secondary handles.
    """
    if isinstance(refs, dict):
        return {k: list(v) for k, v in refs.items()}
    return {"primary": list(refs or []), "secondary": []}


@thumbnails_bp.route("/")
@require_auth
def index():
    if APP_MODE == "descriptions":
        from descriptions.routes import descriptions_index
        return descriptions_index()
    return render_template("thumbnails.html", git_version=GIT_VERSION)


@thumbnails_bp.route("/status")
@require_auth
def get_status():
    _st, _lk = _get_session()

    # Long-poll: hold the request open while nothing changes, up to ~25s.
    # This keeps continuous HTTP activity during generation so Cloud Run
    # allocates CPU to the background generator thread. (On always-on
    # hosts like Render, this has no ill effect — polls just arrive less
    # frequently.) Breaks immediately when any progress field changes.
    def _snapshot():
        with _lk:
            return (
                _st.get("completed", 0),
                _st.get("errors", 0),
                _st.get("phase", "idle"),
                _st.get("done", False),
                _st.get("running", False),
                len(_st.get("images", [])),
                len(_st.get("ideas", [])),
            )

    initial = _snapshot()
    deadline = time.time() + 25.0
    while time.time() < deadline:
        if _snapshot() != initial:
            break
        time.sleep(0.4)

    try:
        with _lk:
            safe = {
                "running": _st["running"],
                "phase": _st["phase"],
                "total": _st["total"],
                "completed": _st["completed"],
                "errors": _st["errors"],
                "log": list(_st["log"]),
                "images": list(_st["images"]),
                "done": _st["done"],
                "output_dir": _st["output_dir"],
                "episode_dir": _st["episode_dir"],
                "round_num": _st["round_num"],
                "ideas": list(_st.get("ideas", [])),
                "idea_groups": {k: list(v) for k, v in _st["idea_groups"].items()},
                "cost": _st["cost"],
                "session_cost": _st.get("session_cost", 0.0),
                "desc_calls": _st.get("desc_calls", 0),
                "desc_input_chars": _st.get("desc_input_chars", 0),
                "desc_output_chars": _st.get("desc_output_chars", 0),
            }
    except Exception:
        safe = {
            "running": _st.get("running", False),
            "done": _st.get("done", False),
            "completed": _st.get("completed", 0),
            "total": _st.get("total", 0),
            "errors": _st.get("errors", 0),
            "log": [], "images": [], "idea_groups": {},
            "phase": _st.get("phase", "idle"),
            "output_dir": "", "episode_dir": "",
            "round_num": 0, "ideas": [],
            "cost": 0, "session_cost": 0,
            "desc_calls": 0, "desc_input_chars": 0, "desc_output_chars": 0,
        }
    return jsonify(safe)


@thumbnails_bp.route("/last_api_call")
@require_auth
def last_api_call():
    _st, _lk = _get_session()
    with _lk:
        payload = _st.get("last_api_call", "")
    return jsonify({"ok": True, "text": payload})


@thumbnails_bp.route("/last_border_api_call")
@require_auth
def last_border_api_call():
    _st, _lk = _get_session()
    with _lk:
        payload = _st.get("last_border_api_call", "")
    return jsonify({"ok": True, "text": payload})


@thumbnails_bp.route("/border_prompt")
@require_auth
def border_prompt():
    return jsonify({"ok": True, "text": BORDER_PASS_PROMPT})


@thumbnails_bp.route("/revision_context_prompt")
@require_auth
def revision_context_prompt():
    return jsonify({"ok": True, "text": REVISION_CONTEXT_PROMPT})


@thumbnails_bp.route("/revision_prompt_template")
@require_auth
def revision_prompt_template():
    primary = get_primary_backend()
    return jsonify({"ok": True, "text": REVISION_PROMPT, "brand_count": len(primary.brand_files), "border_count": len(primary.border_ref_files)})


@thumbnails_bp.route("/image")
@require_auth
def serve_image():
    filepath = request.args.get("path", "")
    if not filepath or not os.path.isfile(filepath):
        return "Not found", 404
    real = os.path.realpath(filepath)
    if not real.startswith(os.path.realpath(THUMBNAILS_DIR)):
        return "Forbidden", 403
    ext = os.path.splitext(filepath)[1].lower()
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".webp": "image/webp",
    }.get(ext, "image/png")
    return send_file(filepath, mimetype=mime)


@thumbnails_bp.route("/download")
@require_auth
def serve_download():
    filepath = request.args.get("path", "")
    if not filepath or not os.path.isfile(filepath):
        return "Not found", 404
    real = os.path.realpath(filepath)
    if not real.startswith(os.path.realpath(THUMBNAILS_DIR)):
        return "Forbidden", 403
    return send_file(filepath, as_attachment=True)


@thumbnails_bp.route("/gather_images", methods=["POST"])
@require_auth
def gather_images():
    fields, _ = parse_form_or_multipart(request)
    title = fields.get("title", "").strip()
    custom_prompt = fields.get("custom_prompt", "").strip()

    if not title:
        return jsonify({"error": "Episode title is required"})
    if not BRAVE_API_KEY:
        return jsonify({"error": "BRAVE_API_KEY not configured. Upload source images manually."})

    try:
        backend = get_primary_backend()
        queries = generate_search_queries(backend.client, title, custom_prompt)
        images = search_images_brave(queries)
        return jsonify({"ok": True, "queries": queries, "images": images})
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)[:200]}"})


@thumbnails_bp.route("/generate_ideas", methods=["POST"])
@require_auth
def generate_ideas_route():
    fields, _ = parse_form_or_multipart(request)
    title = fields.get("title", "").strip()
    custom_prompt = fields.get("custom_prompt", "").strip()
    transcript_text = fields.get("transcript", "").strip()
    additional = fields.get("additional_instructions", "").strip()

    if not title:
        return jsonify({"error": "Episode title is required"})

    try:
        backend = get_primary_backend()
        ideas_list = generate_ideas(backend.client, title, custom_prompt, transcript_text, additional)
        _st, _lk = _get_session()
        with _lk:
            _st["ideas"] = ideas_list
        return jsonify({"ok": True, "ideas": ideas_list})
    except Exception as e:
        return jsonify({"error": f"Idea generation failed: {str(e)[:200]}"})


@thumbnails_bp.route("/more_ideas", methods=["POST"])
@require_auth
def more_ideas():
    fields, _ = parse_form_or_multipart(request)
    title = fields.get("title", "").strip()
    custom_prompt = fields.get("custom_prompt", "").strip()
    transcript_text = fields.get("transcript", "").strip()
    additional = fields.get("additional_instructions", "").strip()
    existing_json = fields.get("existing_ideas", "[]")

    if not title:
        return jsonify({"error": "Episode title is required"})

    try:
        existing = json.loads(existing_json)
    except json.JSONDecodeError:
        existing = []

    extra_instruction = ""
    if existing:
        summaries = [e[:80] for e in existing[:20]]
        extra_instruction = "\n\nAVOID duplicating these existing ideas:\n" + "\n".join(f"- {s}" for s in summaries)

    try:
        backend = get_primary_backend()
        combined_additional = (additional + extra_instruction) if additional else extra_instruction
        new_ideas = generate_ideas(backend.client, title, custom_prompt, transcript_text, combined_additional)
        return jsonify({"ok": True, "ideas": new_ideas})
    except Exception as e:
        return jsonify({"error": f"Idea generation failed: {str(e)[:200]}"})


@thumbnails_bp.route("/generate_from_ideas", methods=["POST"])
@require_auth
def generate_from_ideas():
    _st, _lk = _get_session()
    with _lk:
        is_running = _st["running"]
    if is_running:
        return jsonify({"error": "Generation already in progress for this session"})

    fields, files = parse_form_or_multipart(request)
    title = fields.get("title", "").strip()
    ideas_json = fields.get("ideas", "[]")
    custom_prompt = fields.get("custom_prompt", "").strip()
    additional = fields.get("additional_instructions", "").strip()

    try:
        ideas_list = json.loads(ideas_json)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid ideas JSON"})

    if not ideas_list:
        return jsonify({"error": "No ideas provided"})

    backends = get_all_backends()
    primary = backends[0]
    speaker_refs_by_backend = upload_files_from_bytes(files.get("speakers", []), "speaker")
    source_refs_by_backend = upload_files_from_bytes(files.get("sources", []), "source")

    # Persist speaker/source refs across re-runs: reuse previous if none uploaded
    if any(speaker_refs_by_backend.values()):
        print(f"[THUMB] generate_from_ideas | new speaker_refs={len(speaker_refs_by_backend.get('primary', []))}")
    else:
        speaker_refs_by_backend = _normalize_backend_refs(_st.get("speakers"))
        print(f"[THUMB] generate_from_ideas | reusing stored speaker_refs={len(speaker_refs_by_backend.get('primary', []))}")
    if not any(source_refs_by_backend.values()):
        source_refs_by_backend = _normalize_backend_refs(_st.get("sources"))
        print(f"[THUMB] generate_from_ideas | reusing stored source_refs={len(source_refs_by_backend.get('primary', []))}")

    source_urls_json = fields.get("source_urls", "[]")
    try:
        source_urls = json.loads(source_urls_json)
    except json.JSONDecodeError:
        source_urls = []

    # Parallel download + upload for web-gathered source images. Each worker
    # fetches the URL and uploads the bytes to the Gemini File API on every
    # backend. Sequential version used to take ~15s for 10 images; parallel
    # cuts this to ~3-5s.
    from concurrent.futures import ThreadPoolExecutor

    def _fetch_and_upload(url):
        b = download_image_bytes(url)
        if not b:
            return None
        return upload_files_from_bytes([b], "web_source")

    urls_to_fetch = source_urls[:10]
    if urls_to_fetch:
        with ThreadPoolExecutor(max_workers=10) as pool:
            for extra in pool.map(_fetch_and_upload, urls_to_fetch):
                if not extra:
                    continue
                for name, refs in extra.items():
                    source_refs_by_backend.setdefault(name, []).extend(refs)

    slug = re.sub(r"[^a-z0-9]+", "-", title[:40].lower()).strip("-") or "episode"
    date = datetime.date.today().isoformat()
    episode_dir = os.path.join(THUMBNAILS_DIR, f"{slug}-{date}")
    round_dir = os.path.join(episode_dir, "round1")
    os.makedirs(round_dir, exist_ok=True)

    num_speakers = len(speaker_refs_by_backend.get("primary", []))
    num_sources = len(source_refs_by_backend.get("primary", []))
    info_dict = {
        "title": title,
        "custom_prompt": custom_prompt,
        "num_ideas": len(ideas_list),
        "num_speaker_photos": num_speakers,
        "num_source_images": num_sources,
    }
    with open(os.path.join(episode_dir, "episode.json"), "w") as f:
        json.dump(info_dict, f, indent=2)

    prompts = build_idea_prompts(backends, ideas_list, speaker_refs_by_backend, source_refs_by_backend, custom_prompt, additional, variations_per=3)
    save_metadata(round_dir, info_dict, len(prompts), "round1")

    _st["episode_dir"] = episode_dir
    _st["speakers"] = speaker_refs_by_backend
    _st["sources"] = source_refs_by_backend
    _st["ideas"] = ideas_list
    _st["round_num"] = 1
    _st["add_border"] = fields.get("add_border") == "1"

    session_id = request.args.get("session_id") or request.form.get("session_id") or "default"
    print(f"[THUMB] generate_from_ideas | session={session_id} | ideas={len(ideas_list)} | speakers={num_speakers} | sources={num_sources} | prompts={len(prompts)} | dir={round_dir}")
    run_generation(primary, prompts, round_dir, "round1", target_status=_st, target_lock=_lk)

    return jsonify({"ok": True, "output_dir": round_dir, "count": len(prompts)})


@thumbnails_bp.route("/riff_idea", methods=["POST"])
@require_auth
def riff_idea():
    _st, _lk = _get_session()
    with _lk:
        is_running = _st["running"]
    if is_running:
        return jsonify({"error": "Generation already in progress for this session"})

    fields, files = parse_form_or_multipart(request)
    idea_text = fields.get("idea_text", "").strip()
    idea_idx = int(fields.get("idea_idx", "-1"))
    custom_prompt = fields.get("custom_prompt", "").strip()
    additional = fields.get("additional_instructions", "").strip()
    riff_prompt = fields.get("riff_prompt", "").strip()

    if not idea_text:
        return jsonify({"error": "No idea text provided"})

    if riff_prompt:
        additional = (additional + "\n\nRIFF INSTRUCTIONS: " + riff_prompt) if additional else "RIFF INSTRUCTIONS: " + riff_prompt

    backends = get_all_backends()
    primary = backends[0]
    speaker_refs_by_backend = _normalize_backend_refs(_st.get("speakers"))
    source_refs_by_backend = _normalize_backend_refs(_st.get("sources"))
    riff_image_refs_by_backend = upload_files_from_bytes(files.get("riff_images", []), "riff_img")
    for name, refs in riff_image_refs_by_backend.items():
        source_refs_by_backend.setdefault(name, []).extend(refs)

    episode_dir = _st.get("episode_dir", "")
    if not episode_dir:
        episode_dir = os.path.join(THUMBNAILS_DIR, f"riff-{datetime.date.today().isoformat()}")

    with _lk:
        _st["round_num"] = _st.get("round_num", 1) + 1
        round_num = _st["round_num"]
        existing_max = max((img["idea_idx"] for img in _st["images"]), default=-1)
        ideas_max = len(_st.get("ideas", [])) - 1
        riff_idea_idx = max(existing_max, ideas_max) + 1
        riff_label = f"(Riff on Idea {idea_idx + 1}) {idea_text}"
        ideas_list = _st.get("ideas", [])
        while len(ideas_list) <= riff_idea_idx:
            ideas_list.append("")
        ideas_list[riff_idea_idx] = riff_label
        _st["ideas"] = ideas_list

    session_id = request.args.get("session_id") or request.form.get("session_id") or "default"
    print(f"[THUMB] riff_idea | session={session_id} | idea_idx={idea_idx} | riff_idea_idx={riff_idea_idx} | ideas_len={len(_st.get('ideas', []))} | images_len={len(_st.get('images', []))} | text={idea_text[:60]}")

    round_dir = os.path.join(episode_dir, f"round{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    riff_count = int(fields.get("riff_count", "3"))
    if riff_count < 1:
        riff_count = 3
    if riff_count > 50:
        riff_count = 50
    prompts_raw = build_idea_prompts(backends, [idea_text], speaker_refs_by_backend, source_refs_by_backend, custom_prompt, additional, variations_per=riff_count)
    prompts = [(riff_idea_idx, var, contents) for (_, var, contents) in prompts_raw]

    _st["add_border"] = fields.get("add_border") == "1"
    run_generation(primary, prompts, round_dir, "riff", target_status=_st, target_lock=_lk)
    return jsonify({
        "ok": True, "output_dir": round_dir, "count": len(prompts),
        "riff_idea_idx": riff_idea_idx, "riff_label": idea_text,
    })


@thumbnails_bp.route("/revise", methods=["POST"])
@require_auth
def revise_post():
    _st, _lk = _get_session()
    with _lk:
        is_running = _st["running"]
    if is_running:
        return jsonify({"error": "Generation already in progress for this session"})

    fields, files = parse_form_or_multipart(request)
    indices_raw = fields.get("indices", "")
    custom_prompt = fields.get("prompt", "").strip()
    source_idea_indices_json = fields.get("source_idea_indices", "[]")

    indices = [int(x) for x in indices_raw.split(",") if x.strip().isdigit()]
    if not indices:
        return jsonify({"error": "No images selected"})
    if not custom_prompt:
        return jsonify({"error": "Revision prompt is required"})

    selected_images = []
    source_idea_idxs = set()
    for img_info in _st["images"]:
        if img_info["idx"] in indices and os.path.isfile(img_info["path"]):
            selected_images.append(Image.open(img_info["path"]))
            if img_info.get("idea_idx", -1) >= 0:
                source_idea_idxs.add(img_info["idea_idx"])

    if not selected_images:
        if not _st["images"]:
            return jsonify({"error": "No thumbnails in server memory. The server was likely restarted — please regenerate thumbnails first."})
        else:
            return jsonify({"error": "Could not load selected images (indices " + indices_raw + " not found among " + str(len(_st['images'])) + " known images). Try regenerating."})

    backends = get_all_backends()
    primary = backends[0]
    attachment_refs_by_backend = upload_files_from_bytes(files.get("revision_images", []), "revision_img")
    speaker_refs_by_backend = _normalize_backend_refs(_st.get("speakers"))
    episode_dir = _st.get("episode_dir", "")

    try:
        src_indices = json.loads(source_idea_indices_json)
    except (json.JSONDecodeError, ValueError):
        src_indices = list(source_idea_idxs)

    ideas_list = _st.get("ideas", [])
    source_parts = []
    for si in src_indices:
        si = int(si)
        if 0 <= si < len(ideas_list) and ideas_list[si]:
            source_parts.append(f"Idea {si + 1}: {ideas_list[si]}")
        else:
            source_parts.append(f"Idea {si + 1}")

    prompt_summary = custom_prompt[:120] + ("..." if len(custom_prompt) > 120 else "")
    if source_parts:
        revision_label = f"{prompt_summary} [REVISION OF {', '.join(source_parts)}]"
    else:
        revision_label = f"(Revision) {prompt_summary}"

    with _lk:
        _st["round_num"] = _st.get("round_num", 1) + 1
        round_num = _st["round_num"]
        existing_max = max((img["idea_idx"] for img in _st["images"]), default=-1)
        ideas_max = len(_st.get("ideas", [])) - 1
        revision_idea_idx = max(existing_max, ideas_max) + 1
        while len(ideas_list) <= revision_idea_idx:
            ideas_list.append("")
        ideas_list[revision_idea_idx] = revision_label
        _st["ideas"] = ideas_list

    session_id = request.args.get("session_id") or request.form.get("session_id") or "default"
    print(f"[THUMB] revise_post | session={session_id} | indices={indices} | revision_idea_idx={revision_idea_idx} | ideas_len={len(ideas_list)} | images_len={len(_st.get('images', []))}")

    round_dir = os.path.join(episode_dir, f"round{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    prompts = build_revision_prompts(
        backends, selected_images, speaker_refs_by_backend, custom_prompt,
        count_per=3, idea_idx=revision_idea_idx,
        attachment_refs_by_backend=attachment_refs_by_backend if any(attachment_refs_by_backend.values()) else None,
    )

    _st["add_border"] = fields.get("add_border") == "1"
    run_generation(primary, prompts, round_dir, "revision", target_status=_st, target_lock=_lk)

    return jsonify({
        "ok": True, "output_dir": round_dir, "count": len(prompts),
        "revision_idea_idx": revision_idea_idx, "revision_label": revision_label,
    })


@thumbnails_bp.route("/vary")
@require_auth
def vary():
    _st, _lk = _get_session()
    with _lk:
        is_running = _st["running"]
    if is_running:
        return jsonify({"error": "Generation already in progress for this session"})

    indices_raw = request.args.get("indices", "")
    indices = [int(x) for x in indices_raw.split(",") if x.strip().isdigit()]
    if not indices:
        return jsonify({"error": "No images selected"})

    selected_images = []
    for img_info in _st["images"]:
        if img_info["idx"] in indices and os.path.isfile(img_info["path"]):
            selected_images.append(Image.open(img_info["path"]))

    if not selected_images:
        if not _st["images"]:
            return jsonify({"error": "No thumbnails in server memory. The server was likely restarted — please regenerate thumbnails first."})
        else:
            return jsonify({"error": "Could not load selected images (indices " + indices_raw + " not found among " + str(len(_st['images'])) + " known images). Try regenerating."})

    speaker_refs_by_backend = _normalize_backend_refs(_st.get("speakers"))
    episode_dir = _st.get("episode_dir", "")
    with _lk:
        _st["round_num"] = _st.get("round_num", 1) + 1
        round_num = _st["round_num"]
    round_dir = os.path.join(episode_dir, f"round{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    backends = get_all_backends()
    primary = backends[0]
    prompts = build_variation_prompts(backends, selected_images, speaker_refs_by_backend, count_per=3)
    run_generation(primary, prompts, round_dir, "variation", target_status=_st, target_lock=_lk)

    return jsonify({"ok": True, "output_dir": round_dir, "count": len(prompts)})


@thumbnails_bp.route("/save_finals")
@require_auth
def save_finals():
    _st, _lk = _get_session()
    indices_raw = request.args.get("indices", "")
    indices = [int(x) for x in indices_raw.split(",") if x.strip().isdigit()]
    if not indices:
        return jsonify({"error": "No images selected"})

    episode_dir = _st.get("episode_dir", "")
    if not episode_dir:
        return jsonify({"error": "No episode directory found"})

    finals_dir = os.path.join(episode_dir, "finals")
    os.makedirs(finals_dir, exist_ok=True)

    count = 0
    for img_info in _st["images"]:
        if img_info["idx"] in indices and os.path.isfile(img_info["path"]):
            count += 1
            dest = os.path.join(finals_dir, f"final_{count}.png")
            shutil.copy2(img_info["path"], dest)

    return jsonify({"ok": True, "count": count, "finals_dir": finals_dir})


@thumbnails_bp.route("/save_and_clear", methods=["POST"])
@require_auth
def save_and_clear():
    """Upload the current session's thumbnails to Google Drive and clear generated state.

    Keeps user inputs (title, speakers, sources) so the user can immediately generate
    a new set of ideas/thumbnails without re-uploading anything.
    """
    _st, _lk = _get_session()
    with _lk:
        if _st.get("running"):
            return jsonify({"error": "Cannot save — generation in progress"})
        images = list(_st.get("images", []))
        ideas = list(_st.get("ideas", []))
        episode_dir = _st.get("episode_dir", "")

    if not images:
        return jsonify({"error": "Nothing to save — no thumbnails in session"})

    # Folder name: reuse the episode_dir basename (already {slug}-{date}) or fallback
    if episode_dir:
        base_name = os.path.basename(episode_dir.rstrip("/"))
    else:
        base_name = f"thumbnails-{datetime.date.today().isoformat()}"

    # Group images by idea_idx, sort each group by idx, assign letters
    by_idea = {}
    for img in images:
        if not os.path.isfile(img.get("path", "")):
            continue
        by_idea.setdefault(img.get("idea_idx", -1), []).append(img)

    upload_list = []
    for idea_idx in sorted(by_idea.keys()):
        imgs_sorted = sorted(by_idea[idea_idx], key=lambda x: x.get("idx", 0))
        for letter_idx, img in enumerate(imgs_sorted):
            letter = letter_for_index(letter_idx)
            if idea_idx < 0:
                # Variations without an explicit idea idx (e.g. /vary output)
                display = "var"
            else:
                display = str(idea_idx + 1)
            filename = f"idea{display}{letter}.png"
            upload_list.append((filename, img["path"]))

    if not upload_list:
        return jsonify({"error": "All thumbnail files are missing from disk"})

    # Build ideas.txt content
    lines = [
        f"Episode folder: {base_name}",
        f"Saved:          {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "Ideas:",
    ]
    for i, idea_text in enumerate(ideas, 1):
        lines.append(f"{i}. {idea_text}")
    text = "\n".join(lines) + "\n"

    # Upload
    try:
        result = upload_episode_folder(base_name, text, upload_list)
    except Exception as e:
        return jsonify({"error": f"Drive upload failed: {str(e)[:250]}"})

    # Clear generated state; keep speakers/sources/user inputs
    reset_generation_state(_st, _lk)

    session_id = request.args.get("session_id") or request.form.get("session_id") or "default"
    print(f"[THUMB] save_and_clear | session={session_id} | folder={result['folder_name']} | files={result['file_count']}")

    return jsonify({
        "ok": True,
        "folder_name": result["folder_name"],
        "folder_url": result["folder_url"],
        "file_count": result["file_count"],
    })


@thumbnails_bp.route("/open_folder")
@require_auth
def open_folder():
    _st, _lk = _get_session()
    if os.environ.get("NO_BROWSER") == "1":
        return jsonify({"ok": True, "note": "Folder open disabled on server"})
    episode_dir = _st.get("episode_dir", THUMBNAILS_DIR)
    if os.path.isdir(episode_dir):
        subprocess.Popen(["open", episode_dir])
    return jsonify({"ok": True})


@thumbnails_bp.route("/cancel", methods=["POST"])
@require_auth
def cancel():
    _st, _lk = _get_session()
    session_id = request.args.get("session_id", "default")
    with _lk:
        if _st.get("running"):
            _st["cancel_requested"] = True
            _st["log"].append("Cancel requested — waiting for in-flight API calls to finish...")
            print(f"[THUMB] cancel | session={session_id}")
            return jsonify({"ok": True, "message": "Cancel requested"})
        else:
            return jsonify({"ok": True, "message": "Nothing running"})
