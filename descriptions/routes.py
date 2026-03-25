"""Flask blueprint for the descriptions page (/descriptions)."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Blueprint, jsonify, render_template, request

from auth import require_auth
from config import GIT_VERSION
from shared.gemini_client import get_client
from shared.helpers import parse_form_or_multipart
from shared.state import status, status_lock
from descriptions.generators import (
    _build_description_prompt,
    generate_description_gemini,
    generate_description_claude,
    generate_description_gpt,
)

descriptions_bp = Blueprint("descriptions", __name__, template_folder="templates")


def descriptions_index():
    """Render the descriptions page (called from both /descriptions and / when APP_MODE=descriptions)."""
    return render_template("descriptions.html", git_version=GIT_VERSION)


@descriptions_bp.route("/descriptions")
@require_auth
def descriptions_page():
    return descriptions_index()


@descriptions_bp.route("/generate_descriptions", methods=["POST"])
@require_auth
def generate_descriptions():
    fields, _ = parse_form_or_multipart(request)
    title = fields.get("title", "").strip()
    primary_description = fields.get("primary_description", "").strip()
    revise_instructions = fields.get("revise_instructions", "").strip()
    transcript = fields.get("transcript", "").strip()
    channel_samples = fields.get("channel_samples", "").strip()

    if not primary_description:
        return jsonify({"error": "Primary description is required"})
    if not revise_instructions:
        return jsonify({"error": "How to revise is required"})
    if not transcript:
        return jsonify({"error": "Transcript is required"})

    try:
        client = get_client()
        in_chars = len(title) + len(primary_description) + len(revise_instructions) + len(transcript) + len(channel_samples)
        prompt = _build_description_prompt(title, primary_description, revise_instructions, transcript, channel_samples)

        with status_lock:
            status["running"] = True
            status["phase"] = "descriptions"
            status["done"] = False
            status["log"].append("Starting multi-model description generation (Gemini + Claude + GPT)")

        providers = [
            ("Gemini", lambda: generate_description_gemini(client, prompt)),
            ("Claude", lambda: generate_description_claude(prompt)),
            ("GPT", lambda: generate_description_gpt(prompt)),
        ]

        def _run_provider(name, fn):
            with status_lock:
                status["log"].append(f"{name} generation started")
            try:
                output = fn()
            except Exception as e:
                output = f"[{name} error] {str(e)[:260]}"
            out_chars = len(output or "")
            with status_lock:
                status["desc_calls"] = status.get("desc_calls", 0) + 1
                status["desc_input_chars"] = status.get("desc_input_chars", 0) + in_chars
                status["desc_output_chars"] = status.get("desc_output_chars", 0) + out_chars
                status["log"].append(f"{name} generation done (in={in_chars} chars, out={out_chars} chars)")
            return output

        results = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_run_provider, name, fn): name for name, fn in providers}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result(timeout=5)
                except Exception as e:
                    results[name] = f"[{name} timed out] {str(e)[:200]}"
        outputs = [results.get(name, f"[{name} timed out]") for name, _ in providers]

        with status_lock:
            status["running"] = False
            status["phase"] = "idle"
            status["done"] = True

        return jsonify({"ok": True, "output": "\n\n---\n\n".join(outputs)})
    except Exception as e:
        with status_lock:
            status["running"] = False
            status["phase"] = "idle"
            status["log"].append(f"Description generation failed: {str(e)[:180]}")
        return jsonify({"error": f"Description generation failed: {str(e)[:300]}"})
