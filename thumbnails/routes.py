"""Flask blueprint for the main thumbnail generation page (/)."""

import datetime
import json
import os
import re
import shutil
import subprocess

from flask import Blueprint, jsonify, render_template, request, send_file, Response
from PIL import Image

from auth import require_auth
from config import (
    APP_MODE, BRAVE_API_KEY, GIT_VERSION, THUMBNAILS_DIR,
)
from shared.gemini_client import get_client, upload_files_from_bytes, BRAND_FILES, BORDER_REF_FILES
from shared.helpers import parse_form_or_multipart
from shared.state import main_status, main_status_lock, revision_status, revision_status_lock
from thumbnails.brave_search import search_images_brave, download_image_bytes
from thumbnails.generator import (
    generate_ideas, generate_search_queries,
    build_idea_prompts, build_variation_prompts, build_revision_prompts,
    run_generation, save_metadata,
)
from thumbnails.prompts import BORDER_PASS_PROMPT, REVISION_PROMPT, REVISION_CONTEXT_PROMPT

thumbnails_bp = Blueprint("thumbnails", __name__, template_folder="templates")


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
    params = request.args
    _st, _lk = (revision_status, revision_status_lock) if params.get("page") == "revision" else (main_status, main_status_lock)
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
    _st, _lk = (revision_status, revision_status_lock) if request.args.get("page") == "revision" else (main_status, main_status_lock)
    with _lk:
        payload = _st.get("last_api_call", "")
    return jsonify({"ok": True, "text": payload})


@thumbnails_bp.route("/last_border_api_call")
@require_auth
def last_border_api_call():
    _st, _lk = (revision_status, revision_status_lock) if request.args.get("page") == "revision" else (main_status, main_status_lock)
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
    return jsonify({"ok": True, "text": REVISION_PROMPT, "brand_count": len(BRAND_FILES), "border_count": len(BORDER_REF_FILES)})


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
        client = get_client()
        queries = generate_search_queries(client, title, custom_prompt)
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
        client = get_client()
        ideas_list = generate_ideas(client, title, custom_prompt, transcript_text, additional)
        main_status["ideas"] = ideas_list
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
        client = get_client()
        combined_additional = (additional + extra_instruction) if additional else extra_instruction
        new_ideas = generate_ideas(client, title, custom_prompt, transcript_text, combined_additional)
        return jsonify({"ok": True, "ideas": new_ideas})
    except Exception as e:
        return jsonify({"error": f"Idea generation failed: {str(e)[:200]}"})


@thumbnails_bp.route("/generate_from_ideas", methods=["POST"])
@require_auth
def generate_from_ideas():
    with main_status_lock:
        is_running = main_status["running"]
    if is_running:
        return jsonify({"error": "Generation already in progress"})

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

    client = get_client()
    speaker_refs = upload_files_from_bytes(client, files.get("speakers", []), "speaker")
    source_refs = upload_files_from_bytes(client, files.get("sources", []), "source")

    source_urls_json = fields.get("source_urls", "[]")
    try:
        source_urls = json.loads(source_urls_json)
    except json.JSONDecodeError:
        source_urls = []

    for url in source_urls[:15]:
        img_bytes = download_image_bytes(url)
        if img_bytes:
            refs = upload_files_from_bytes(client, [img_bytes], "web_source")
            source_refs.extend(refs)

    slug = re.sub(r"[^a-z0-9]+", "-", title[:40].lower()).strip("-") or "episode"
    date = datetime.date.today().isoformat()
    episode_dir = os.path.join(THUMBNAILS_DIR, f"{slug}-{date}")
    round_dir = os.path.join(episode_dir, "round1")
    os.makedirs(round_dir, exist_ok=True)

    info_dict = {
        "title": title,
        "custom_prompt": custom_prompt,
        "num_ideas": len(ideas_list),
        "num_speaker_photos": len(speaker_refs),
        "num_source_images": len(source_refs),
    }
    with open(os.path.join(episode_dir, "episode.json"), "w") as f:
        json.dump(info_dict, f, indent=2)

    prompts = build_idea_prompts(ideas_list, speaker_refs, source_refs, custom_prompt, additional, variations_per=3)
    save_metadata(round_dir, info_dict, len(prompts), "round1")

    main_status["episode_dir"] = episode_dir
    main_status["speakers"] = speaker_refs
    main_status["sources"] = source_refs
    main_status["ideas"] = ideas_list
    main_status["round_num"] = 1
    main_status["add_border"] = fields.get("add_border") == "1"
    run_generation(client, prompts, round_dir, "round1", target_status=main_status, target_lock=main_status_lock)

    return jsonify({"ok": True, "output_dir": round_dir, "count": len(prompts)})


@thumbnails_bp.route("/riff_idea", methods=["POST"])
@require_auth
def riff_idea():
    with main_status_lock:
        is_running = main_status["running"]
    if is_running:
        return jsonify({"error": "Generation already in progress"})

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

    client = get_client()
    speaker_refs = main_status.get("speakers", [])
    source_refs = list(main_status.get("sources", []))
    riff_image_refs = upload_files_from_bytes(client, files.get("riff_images", []), "riff_img")
    source_refs.extend(riff_image_refs)

    episode_dir = main_status.get("episode_dir", "")
    if not episode_dir:
        episode_dir = os.path.join(THUMBNAILS_DIR, f"riff-{datetime.date.today().isoformat()}")

    with main_status_lock:
        main_status["round_num"] = main_status.get("round_num", 1) + 1
        round_num = main_status["round_num"]
        existing_max = max((img["idea_idx"] for img in main_status["images"]), default=-1)
        ideas_max = len(main_status.get("ideas", [])) - 1
        riff_idea_idx = max(existing_max, ideas_max) + 1
        riff_label = f"(Riff on Idea {idea_idx + 1}) {idea_text}"
        ideas_list = main_status.get("ideas", [])
        while len(ideas_list) <= riff_idea_idx:
            ideas_list.append("")
        ideas_list[riff_idea_idx] = riff_label
        main_status["ideas"] = ideas_list

    round_dir = os.path.join(episode_dir, f"round{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    riff_count = int(fields.get("riff_count", "3"))
    if riff_count < 1:
        riff_count = 3
    if riff_count > 50:
        riff_count = 50
    prompts_raw = build_idea_prompts([idea_text], speaker_refs, source_refs, custom_prompt, additional, variations_per=riff_count)
    prompts = [(riff_idea_idx, var, contents) for (_, var, contents) in prompts_raw]

    main_status["add_border"] = fields.get("add_border") == "1"
    run_generation(client, prompts, round_dir, "riff", target_status=main_status, target_lock=main_status_lock)
    return jsonify({
        "ok": True, "output_dir": round_dir, "count": len(prompts),
        "riff_idea_idx": riff_idea_idx, "riff_label": idea_text,
    })


@thumbnails_bp.route("/revise", methods=["POST"])
@require_auth
def revise_post():
    with main_status_lock:
        is_running = main_status["running"]
    if is_running:
        return jsonify({"error": "Generation already in progress"})

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
    for img_info in main_status["images"]:
        if img_info["idx"] in indices and os.path.isfile(img_info["path"]):
            selected_images.append(Image.open(img_info["path"]))
            if img_info.get("idea_idx", -1) >= 0:
                source_idea_idxs.add(img_info["idea_idx"])

    if not selected_images:
        if not main_status["images"]:
            return jsonify({"error": "No thumbnails in server memory. The server was likely restarted — please regenerate thumbnails first."})
        else:
            return jsonify({"error": "Could not load selected images (indices " + indices_raw + " not found among " + str(len(main_status['images'])) + " known images). Try regenerating."})

    client = get_client()
    attachment_refs = upload_files_from_bytes(client, files.get("revision_images", []), "revision_img")
    speakers = main_status.get("speakers", [])
    episode_dir = main_status.get("episode_dir", "")

    try:
        src_indices = json.loads(source_idea_indices_json)
    except (json.JSONDecodeError, ValueError):
        src_indices = list(source_idea_idxs)

    ideas_list = main_status.get("ideas", [])
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

    with main_status_lock:
        main_status["round_num"] = main_status.get("round_num", 1) + 1
        round_num = main_status["round_num"]
        existing_max = max((img["idea_idx"] for img in main_status["images"]), default=-1)
        ideas_max = len(main_status.get("ideas", [])) - 1
        revision_idea_idx = max(existing_max, ideas_max) + 1
        while len(ideas_list) <= revision_idea_idx:
            ideas_list.append("")
        ideas_list[revision_idea_idx] = revision_label
        main_status["ideas"] = ideas_list

    round_dir = os.path.join(episode_dir, f"round{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    prompts = build_revision_prompts(
        selected_images, speakers, custom_prompt,
        count_per=3, idea_idx=revision_idea_idx,
        attachment_refs=attachment_refs if attachment_refs else None,
    )

    main_status["add_border"] = fields.get("add_border") == "1"
    run_generation(client, prompts, round_dir, "revision", target_status=main_status, target_lock=main_status_lock)

    return jsonify({
        "ok": True, "output_dir": round_dir, "count": len(prompts),
        "revision_idea_idx": revision_idea_idx, "revision_label": revision_label,
    })


@thumbnails_bp.route("/vary")
@require_auth
def vary():
    with main_status_lock:
        is_running = main_status["running"]
    if is_running:
        return jsonify({"error": "Generation already in progress"})

    indices_raw = request.args.get("indices", "")
    indices = [int(x) for x in indices_raw.split(",") if x.strip().isdigit()]
    if not indices:
        return jsonify({"error": "No images selected"})

    selected_images = []
    for img_info in main_status["images"]:
        if img_info["idx"] in indices and os.path.isfile(img_info["path"]):
            selected_images.append(Image.open(img_info["path"]))

    if not selected_images:
        if not main_status["images"]:
            return jsonify({"error": "No thumbnails in server memory. The server was likely restarted — please regenerate thumbnails first."})
        else:
            return jsonify({"error": "Could not load selected images (indices " + indices_raw + " not found among " + str(len(main_status['images'])) + " known images). Try regenerating."})

    speakers = main_status.get("speakers", [])
    episode_dir = main_status.get("episode_dir", "")
    with main_status_lock:
        main_status["round_num"] = main_status.get("round_num", 1) + 1
        round_num = main_status["round_num"]
    round_dir = os.path.join(episode_dir, f"round{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    prompts = build_variation_prompts(selected_images, speakers, count_per=3)
    client = get_client()
    run_generation(client, prompts, round_dir, "variation", target_status=main_status, target_lock=main_status_lock)

    return jsonify({"ok": True, "output_dir": round_dir, "count": len(prompts)})


@thumbnails_bp.route("/save_finals")
@require_auth
def save_finals():
    indices_raw = request.args.get("indices", "")
    indices = [int(x) for x in indices_raw.split(",") if x.strip().isdigit()]
    if not indices:
        return jsonify({"error": "No images selected"})

    episode_dir = main_status.get("episode_dir", "")
    if not episode_dir:
        return jsonify({"error": "No episode directory found"})

    finals_dir = os.path.join(episode_dir, "finals")
    os.makedirs(finals_dir, exist_ok=True)

    count = 0
    for img_info in main_status["images"]:
        if img_info["idx"] in indices and os.path.isfile(img_info["path"]):
            count += 1
            dest = os.path.join(finals_dir, f"final_{count}.png")
            shutil.copy2(img_info["path"], dest)

    return jsonify({"ok": True, "count": count, "finals_dir": finals_dir})


@thumbnails_bp.route("/open_folder")
@require_auth
def open_folder():
    if os.environ.get("NO_BROWSER") == "1":
        return jsonify({"ok": True, "note": "Folder open disabled on server"})
    episode_dir = main_status.get("episode_dir", THUMBNAILS_DIR)
    if os.path.isdir(episode_dir):
        subprocess.Popen(["open", episode_dir])
    return jsonify({"ok": True})


@thumbnails_bp.route("/cancel", methods=["POST"])
@require_auth
def cancel():
    params = request.args
    _st, _lk = (revision_status, revision_status_lock) if params.get("page") == "revision" else (main_status, main_status_lock)
    with _lk:
        if _st.get("running"):
            _st["cancel_requested"] = True
            _st["log"].append("Cancel requested — waiting for in-flight API calls to finish...")
            return jsonify({"ok": True, "message": "Cancel requested"})
        else:
            return jsonify({"ok": True, "message": "Nothing running"})
