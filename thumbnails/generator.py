"""Async thumbnail generation via Gemini image API."""

import asyncio
import datetime
import io
import json
import os
import random
import re
import threading

import requests
from google import genai
from google.genai import types
from PIL import Image

from config import (
    GEMINI_MODEL, TEXT_MODEL, CLAUDE_IDEA_MODEL, ANTHROPIC_API_KEY,
    THUMBNAILS_DIR, MAX_CONCURRENT, COST_PER_IMAGE,
    MAX_BRAND_REFS_PER_CALL, MAX_SPEAKER_REFS_PER_CALL, MAX_LIRON_REFS_PER_CALL,
)
from shared.gemini_client import GeminiBackend, get_fallback_backend
from shared.helpers import _record_api_call
from shared.state import main_status, main_status_lock, MAX_LOG_ENTRIES
from thumbnails.prompts import (
    BRAND_GUIDE, IDEA_GENERATION_PROMPT, SEARCH_QUERY_PROMPT,
    IDEA_THUMBNAIL_PROMPT, REVISION_PROMPT, REVISION_CONTEXT_PROMPT,
    VARIATION_PROMPT, BORDER_PASS_PROMPT,
)


# ----- Idea Generation -----


def _parse_json_array(text):
    """Parse a JSON array from text, tolerant of extra text around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def generate_ideas(client, title, custom_prompt, transcript, additional_instructions):
    """Use Claude Opus 4.7 to generate 10 thumbnail ideas. Returns list of strings.

    The `client` arg is a Gemini client kept for call-site compatibility — not used here.
    """
    custom_section = f"CUSTOM PROMPT INFO: {custom_prompt}" if custom_prompt else ""
    transcript_section = f"EPISODE TRANSCRIPT (full):\n{transcript}" if transcript else ""
    addl = f"ADDITIONAL INSTRUCTIONS: {additional_instructions}" if additional_instructions else ""

    prompt = IDEA_GENERATION_PROMPT.format(
        title=title,
        custom_prompt_section=custom_section,
        transcript_section=transcript_section,
        additional_instructions=addl,
    )
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set — required for thumbnail idea generation")
    _record_api_call(CLAUDE_IDEA_MODEL, prompt, phase="idea_generation")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_IDEA_MODEL,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = data.get("content", [])
    text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return _parse_json_array(text)


def generate_search_queries(client, title, custom_prompt):
    """Use Gemini text model to suggest image search queries. Returns list of strings."""
    custom_section = f"CUSTOM PROMPT INFO: {custom_prompt}" if custom_prompt else ""
    prompt = SEARCH_QUERY_PROMPT.format(title=title, custom_prompt_section=custom_section)
    _record_api_call(TEXT_MODEL, prompt, phase="search_query_generation")
    response = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return _parse_json_array(text)


# ----- Prompt Building -----


def _count_identity_refs(refs, max_refs):
    """How many identity refs will be used (used to decide whether to attach them)."""
    if not refs:
        return 0
    return min(len(refs), max_refs)


def _select_identity_refs(refs, max_refs):
    """Choose a small, deterministic subset of identity refs."""
    if not refs:
        return []
    return list(refs[:max_refs])


def _select_brand_refs(backend: GeminiBackend):
    """Choose a minimal style-only subset of brand refs per generation call."""
    if not backend.brand_files:
        return []
    return list(backend.brand_files[:min(MAX_BRAND_REFS_PER_CALL, len(backend.brand_files))])


def _refs_for_backend(backend: GeminiBackend, per_backend_dict):
    """Extract this backend's slice from a per-backend dict like {'primary': [...], 'secondary': [...]}."""
    if not per_backend_dict:
        return []
    return list(per_backend_dict.get(backend.name, []))


def build_idea_prompts(backends, ideas, speaker_refs_by_backend, source_refs_by_backend, custom_prompt, additional_instructions, variations_per=3):
    """Build prompt content lists for each idea x N variations, one contents list per backend.

    Returns list of (idea_idx, variation_num, {backend_name: contents}) tuples.
    speaker_refs_by_backend / source_refs_by_backend are dicts keyed by backend.name
    containing file refs for each backend (since File API handles are per-key).
    """
    custom_section = f"CUSTOM PROMPT INFO: {custom_prompt}" if custom_prompt else ""
    addl = f"ADDITIONAL INSTRUCTIONS: {additional_instructions}" if additional_instructions else ""

    prompts = []
    for idea_idx, idea_text in enumerate(ideas):
        idea_mentions_liron = "liron" in idea_text.lower()

        for v in range(variations_per):
            contents_by_backend = {}
            for backend in backends:
                speaker_refs = _refs_for_backend(backend, speaker_refs_by_backend)
                source_refs = _refs_for_backend(backend, source_refs_by_backend)
                selected_speaker_refs = _select_identity_refs(speaker_refs, MAX_SPEAKER_REFS_PER_CALL)
                speaker_section = (
                    "SPEAKER LIKENESS (CRITICAL): A targeted subset of the episode speaker photo library is attached below. "
                    "The person(s) in the thumbnail MUST closely resemble these photos — same face, same features, "
                    "same skin tone, same hair. Do NOT use generic faces. The speaker photos are the ground truth "
                    "for what the person looks like."
                    if selected_speaker_refs else ""
                )
                selected_liron_refs = _select_identity_refs(backend.liron_files, MAX_LIRON_REFS_PER_CALL) if idea_mentions_liron else []

                prompt_text = IDEA_THUMBNAIL_PROMPT.format(
                    idea_text=idea_text,
                    custom_prompt_section=custom_section,
                    brand_guide=BRAND_GUIDE,
                    speaker_section=speaker_section,
                    liron_section="",
                    additional_instructions=addl,
                    variation_seed=v + 1,
                    variation_total=variations_per,
                )
                contents = [prompt_text]

                if selected_liron_refs:
                    contents.append(
                        "In this image, you must make one of the faces look like Liron Shapira. "
                        "Liron Shapira looks like the following enclosed images:"
                    )
                    for lf in selected_liron_refs:
                        contents.append(lf)
                    contents.append(
                        "The face of Liron in your output MUST be a faithful reproduction of the person in the above photos — "
                        "same facial structure, same nose, same eyes, same beard shape, same skin tone. "
                        "Do NOT generate a generic man's face. Do NOT invent features. Copy Liron's exact likeness."
                    )

                brand_sample = _select_brand_refs(backend)
                if brand_sample:
                    contents.append("=== DOOM DEBATES BRAND STYLE ONLY — match colors, layout, typography, energy. WARNING: These images contain people — COMPLETELY IGNORE all faces/people in these images. Do NOT reproduce any human likeness from these references. ===")
                    contents.extend(brand_sample)

                if selected_speaker_refs:
                    contents.append("=== SPEAKER PHOTOS — targeted subset; the thumbnail MUST use these people's real faces ===")
                    contents.extend(selected_speaker_refs)

                if source_refs:
                    contents.append("=== SOURCE IMAGES — use these as visual reference material ===")
                    contents.extend(source_refs)

                contents_by_backend[backend.name] = contents

            prompts.append((idea_idx, v, contents_by_backend))

    return prompts


def build_riff_prompts(backends, idea_text, idea_idx, speaker_refs_by_backend, source_refs_by_backend, custom_prompt, additional_instructions, count=3):
    """Build prompts for riffing on a single idea."""
    return build_idea_prompts(
        backends, [idea_text], speaker_refs_by_backend, source_refs_by_backend,
        custom_prompt, additional_instructions, variations_per=count,
    )


def build_variation_prompts(backends, selected_images, speaker_refs_by_backend, count_per=3):
    """Build variation prompts from selected images.

    selected_images is a plain list of PIL images (or raw bytes) — not per-backend.
    """
    prompts = []
    for img in selected_images:
        for v in range(count_per):
            contents_by_backend = {}
            for backend in backends:
                speaker_refs = _refs_for_backend(backend, speaker_refs_by_backend)
                selected_speaker_refs = _select_identity_refs(speaker_refs, MAX_SPEAKER_REFS_PER_CALL)
                speaker_section = (
                    "SPEAKER LIKENESS (CRITICAL): A targeted subset of speaker photos is attached — the person(s) MUST closely "
                    "resemble these photos. Same face, features, skin tone, hair."
                    if selected_speaker_refs else ""
                )
                prompt_text = VARIATION_PROMPT.format(
                    speaker_section=speaker_section,
                    variation_seed=v + 1,
                    variation_total=count_per,
                )
                contents = [prompt_text, img]
                brand_sample = _select_brand_refs(backend)
                if brand_sample:
                    contents.append("=== DOOM DEBATES BRAND STYLE ONLY — match colors, layout, typography, energy. WARNING: These images contain people — COMPLETELY IGNORE all faces/people in these images. Do NOT reproduce any human likeness from these references. ===")
                    contents.extend(brand_sample)
                if selected_speaker_refs:
                    contents.append("=== SPEAKER PHOTOS (targeted subset) ===")
                    contents.extend(selected_speaker_refs)
                contents_by_backend[backend.name] = contents
            prompts.append((-1, v, contents_by_backend))
    return prompts


TAG_RE = re.compile(r"\[tag:\s*([a-z0-9_-]+)\s*\]", re.IGNORECASE)
LIRON_TAG_REPLACEMENT = (
    "Liron (whose reference photos are enclosed below, "
    "see 'LIRON REFERENCE PHOTOS')"
)

# Revision reliability (Phase 2): temperature jitter across retry attempts.
REVISION_TEMPS = [0.8, 1.0, 1.2, 1.4, 0.6, 1.5, 0.9, 1.1, 1.3, 0.7]
REVISION_MAX_ATTEMPTS = 30


def _diagnose_no_image(response):
    """Extract Gemini refusal metadata when no image part was returned."""
    info = {"finish_reason": None, "block_reason": None,
            "safety": None, "text_parts": []}
    try:
        if response.candidates:
            info["finish_reason"] = str(getattr(response.candidates[0], "finish_reason", None))
            for p in (response.candidates[0].content.parts or []):
                t = getattr(p, "text", None)
                if t:
                    info["text_parts"].append(t[:500])
        pf = getattr(response, "prompt_feedback", None)
        if pf:
            info["block_reason"] = str(getattr(pf, "block_reason", None))
            sr = getattr(pf, "safety_ratings", None)
            if sr:
                info["safety"] = [str(r) for r in sr]
    except Exception as e:
        info["diagnosis_error"] = str(e)
    return info


def _perturb_contents(contents, level):
    """Strip parts from a revision contents array to reduce safety-block surface.

    Level 0: no change (attempts 1-10).
    Level 1 (attempts 11-20): drop all brand reference Files (display_name
        prefix "brand_"). Brand refs contain people and are the most common
        safety trigger on Gemini image gen.
    Level 2 (attempts 21-30): L1 plus drop the REVISION_CONTEXT_PROMPT text
        block — reduces prompt size to bare essentials.
    """
    if level <= 0:
        return list(contents)
    from thumbnails.prompts import REVISION_CONTEXT_PROMPT
    ctx_prefix = REVISION_CONTEXT_PROMPT[:60]
    out = []
    for c in contents:
        dn = getattr(c, "display_name", None)
        if level >= 1 and isinstance(dn, str) and dn.startswith("brand_"):
            continue
        if level >= 2 and isinstance(c, str) and c.startswith(ctx_prefix):
            continue
        out.append(c)
    return out


def _rewrite_liron_tags(prompt):
    """Replace every [tag: liron] with the verbose inline form.

    Returns (rewritten_prompt, had_liron_tag). Non-liron tags are left
    verbatim and logged as unknown.
    """
    had = False

    def sub(m):
        nonlocal had
        name = m.group(1).lower()
        if name == "liron":
            had = True
            return LIRON_TAG_REPLACEMENT
        print(f"[REVISION] unknown tag: {name}")
        return m.group(0)

    return TAG_RE.sub(sub, prompt or ""), had


def build_revision_prompts(backends, selected_images, speaker_refs_by_backend, custom_prompt, count_per=3, idea_idx=-1, attachment_refs_by_backend=None, context_prompt=None):
    """Build revision prompts with custom instructions.

    Liron refs are only attached when the prompt contains an explicit
    [tag: liron] marker. The tag is rewritten inline to a verbose
    natural-language phrase pointing at the LIRON REFERENCE PHOTOS block.
    """
    prompts = []
    rewritten, had_liron_tag = _rewrite_liron_tags(custom_prompt)
    if had_liron_tag:
        print("[REVISION] [tag: liron] detected — attaching Liron reference photos")

    for img in selected_images:
        for v in range(count_per):
            contents_by_backend = {}
            for backend in backends:
                liron_refs = (
                    _select_identity_refs(backend.liron_files, MAX_LIRON_REFS_PER_CALL)
                    if had_liron_tag else []
                )
                attachment_refs = _refs_for_backend(backend, attachment_refs_by_backend)

                prompt_for_model = rewritten
                if had_liron_tag and not liron_refs:
                    print("[REVISION] [tag: liron] present but no Liron refs uploaded — stripping inline reference")
                    prompt_for_model = rewritten.replace(LIRON_TAG_REPLACEMENT, "Liron")

                prompt_text = REVISION_PROMPT.format(
                    custom_prompt=prompt_for_model,
                    variation_seed=v + 1,
                    variation_total=count_per,
                )
                contents = [prompt_text, img]

                if liron_refs:
                    contents.append(
                        "=== LIRON REFERENCE PHOTOS === "
                        "The person referred to as 'Liron' in the instructions above "
                        "MUST be a faithful likeness of the person in the following photos — "
                        "same facial structure, nose, eyes, beard shape, skin tone. "
                        "Do NOT generate a generic man's face. Copy this exact likeness."
                    )
                    contents.extend(liron_refs)

                contents.append(context_prompt or REVISION_CONTEXT_PROMPT)
                brand_sample = _select_brand_refs(backend)
                if brand_sample:
                    contents.extend(brand_sample)
                if attachment_refs:
                    contents.append("The user has attached the following reference image(s) — use them to guide the revision:")
                    contents.extend(attachment_refs)
                contents_by_backend[backend.name] = contents
            prompts.append((idea_idx, v, contents_by_backend))
    return prompts


# ----- Face Change (GPT Image API inpainting) -----


def apply_face_change(img_data: bytes, face_prompt: str, ref_face_data: bytes = None, mask_data: bytes = None) -> bytes:
    """Swap/adjust a face using Pillow mask + GPT Image API inpainting."""
    import cv2
    import numpy as np
    import openai
    import base64
    from config import OPENAI_API_KEY

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    thumb = Image.open(io.BytesIO(img_data)).convert("RGB")
    thumb = thumb.resize((1280, 720), Image.LANCZOS)

    if mask_data:
        # Hand-drawn mask provided — use it directly
        mask = Image.open(io.BytesIO(mask_data)).convert("RGBA")
        mask = mask.resize((1280, 720), Image.LANCZOS)
    else:
        # Auto-detect faces with OpenCV
        thumb_arr = np.array(thumb)
        gray = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 0:
            raise RuntimeError("No faces detected in the thumbnail")

        prompt_lower = face_prompt.lower()
        if len(faces) == 1:
            target = faces[0]
        else:
            face_centers = [(x + w // 2, i) for i, (x, y, w, h) in enumerate(faces)]
            if "right" in prompt_lower:
                target = faces[max(face_centers, key=lambda c: c[0])[1]]
            elif "left" in prompt_lower:
                target = faces[min(face_centers, key=lambda c: c[0])[1]]
            elif "middle" in prompt_lower or "center" in prompt_lower:
                mid = 640
                target = faces[min(face_centers, key=lambda c: abs(c[0] - mid))[1]]
            else:
                target = faces[max(face_centers, key=lambda c: c[0])[1]]

        fx, fy, fw, fh = int(target[0]), int(target[1]), int(target[2]), int(target[3])
        pad_top = int(fh * 0.3)
        pad_bottom = int(fh * 0.15)
        pad_side = int(fw * 0.1)
        ex = max(0, fx - pad_side)
        ey = max(0, fy - pad_top)
        ew = fw + 2 * pad_side
        eh = fh + pad_top + pad_bottom

        from PIL import ImageDraw
        mask = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
        draw = ImageDraw.Draw(mask)
        draw.ellipse([ex, ey, ex + ew, ey + eh], fill=(0, 0, 0, 0))

    thumb_buf = io.BytesIO()
    thumb.save(thumb_buf, "PNG")
    thumb_buf.seek(0)

    mask_buf = io.BytesIO()
    mask.save(mask_buf, "PNG")
    mask_buf.seek(0)

    # Build image list: thumbnail + optional reference face
    images = [("thumb.png", thumb_buf, "image/png")]
    if ref_face_data:
        images.append(("ref.png", io.BytesIO(ref_face_data), "image/png"))

    import time as _time
    _t0 = _time.time()
    thread_name = threading.current_thread().name
    print(f"[FACE] {thread_name} starting OpenAI call at t={_t0:.1f}")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.images.edit(
        model="gpt-image-2",
        image=images,
        mask=("mask.png", mask_buf, "image/png"),
        prompt=face_prompt,
        size="1280x720",
        quality="high",
    )

    _elapsed = _time.time() - _t0
    print(f"[FACE] {thread_name} finished OpenAI call in {_elapsed:.1f}s")

    result_b64 = response.data[0].b64_json
    if not result_b64:
        raise RuntimeError("GPT Image API returned no image")
    return base64.b64decode(result_b64)


# ----- Pillow Border Composite (revision mode) -----

_border_frame_cache = None
_logo_cache = None
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


def apply_border_pillow(img_data: bytes, logo_corner: str = "bottom-left") -> bytes:
    """Deterministic border composite using Pillow alpha-compositing."""
    global _border_frame_cache, _logo_cache
    if _border_frame_cache is None:
        _border_frame_cache = Image.open(os.path.join(ASSETS_DIR, "border-frame.png")).convert("RGBA")
    if _logo_cache is None:
        _logo_cache = Image.open(os.path.join(ASSETS_DIR, "doom-debates-logo.png")).convert("RGBA")

    thumb = Image.open(io.BytesIO(img_data)).convert("RGBA")
    thumb = thumb.resize((1280, 720), Image.LANCZOS)

    logo = _logo_cache
    # Logo text occupies x=117-268, y=111-165 within the 387x290 asset.
    # Reference position: text at x=57-208, y=619-673 (bottom-left).
    # Paste offset = desired_text_start - text_offset_in_asset
    text_x0, text_y0 = 117, 111  # text top-left within asset
    lw, lh = logo.size
    edge_pad_x, edge_pad_y = 57, 47  # padding from canvas edge to text start
    positions = {
        "bottom-left": (edge_pad_x - text_x0, 720 - edge_pad_y - (lh - text_y0)),
        "bottom-right": (1280 - edge_pad_x - (lw - text_x0), 720 - edge_pad_y - (lh - text_y0)),
        "top-left": (edge_pad_x - text_x0, edge_pad_y - text_y0),
        "top-right": (1280 - edge_pad_x - (lw - text_x0), edge_pad_y - text_y0),
    }
    pos = positions.get(logo_corner, positions["bottom-left"])

    logo_layer = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    logo_layer.paste(logo, pos)
    # Split logo into shadow (dark glow) and text (white)
    import numpy as np
    logo_arr = np.array(logo_layer)
    text_mask = (logo_arr[:,:,0] > 180) & (logo_arr[:,:,1] > 180) & (logo_arr[:,:,2] > 180) & (logo_arr[:,:,3] > 100)
    shadow_only = logo_arr.copy()
    shadow_only[text_mask] = 0
    text_only = np.zeros_like(logo_arr)
    text_only[text_mask] = logo_arr[text_mask]
    shadow_layer = Image.fromarray(shadow_only, "RGBA")
    text_layer = Image.fromarray(text_only, "RGBA")
    # Order: thumbnail → shadow → border → white text (text stays pure white)
    result = Image.alpha_composite(thumb, shadow_layer)
    result = Image.alpha_composite(result, _border_frame_cache)
    result = Image.alpha_composite(result, text_layer)

    out = io.BytesIO()
    result.convert("RGB").save(out, "PNG")
    return out.getvalue()


# ----- Async Generation -----


async def apply_border_pass(backend: GeminiBackend, img_data, prompt_override=None, target_status=None, target_lock=None, client_override=None):
    """Second Gemini pass: add red border + DOOM DEBATES wordmark."""
    _st = target_status if target_status is not None else main_status
    _lk = target_lock if target_lock is not None else main_status_lock
    if not backend.border_ref_files:
        return None
    try:
        prompt_text = prompt_override or BORDER_PASS_PROMPT
        source_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        border_contents = list(backend.border_ref_files) + [source_img, prompt_text]
        _record_api_call(GEMINI_MODEL, border_contents, phase="border_pass", key="last_border_api_call", target_status=_st, target_lock=_lk)
        api_client = client_override or backend.client
        response = await asyncio.wait_for(
            api_client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=border_contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            ),
            timeout=120,
        )
        if (response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts):
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data
        with _lk:
            _st["log"].append(f"Border pass: no image in response")
    except Exception as e:
        with _lk:
            _st["log"].append(f"Border pass error: {str(e)[:150]}")
        print(f"Border pass failed: {e}")
    return None


async def generate_batch(backend: GeminiBackend, prompts, output_dir, phase="round1", target_status=None, target_lock=None, revision_mode=False, judge_fn=None):
    """Fire parallel Gemini API calls and save results.

    prompts is a list of (idea_idx, variation_num, contents_by_backend) where
    contents_by_backend is {backend_name: [contents]} — one contents list per backend.
    On 429 from primary, the batch swaps its current backend to secondary.

    When `revision_mode=True`: each slot gets up to REVISION_MAX_ATTEMPTS
    tries with temperature jitter + progressive prompt perturbation, and
    (if `judge_fn` is provided) the produced image must pass the judge
    before it's accepted. First judge-matching image wins for that slot.
    """
    _st = target_status if target_status is not None else main_status
    _lk = target_lock if target_lock is not None else main_status_lock
    os.makedirs(output_dir, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    with _lk:
        start_idx = len(_st["images"])

    # Create fresh genai.Client instances bound to THIS event loop's async context.
    # The singleton clients from gemini_client.py have httpx.AsyncClient objects that
    # bind to the main thread's event loop at init time, causing "Event loop is closed"
    # when used from a background thread's asyncio.run().
    from shared.gemini_client import get_all_backends
    _local_clients = {}
    for b in get_all_backends():
        _local_clients[b.name] = genai.Client(api_key=b.client._api_client.api_key)

    # Mutable, shared across all concurrent generate_one tasks. On 429, one task
    # may swap this to the fallback backend; all subsequent tasks pick up the new backend.
    current_backend_ref = [backend]
    swap_lock = threading.Lock()

    def _maybe_swap(log_prefix):
        """Swap current backend to fallback if available. Return the (possibly new) backend."""
        with swap_lock:
            cur = current_backend_ref[0]
            fb = get_fallback_backend(cur)
            if fb is None or fb is cur:
                return cur
            current_backend_ref[0] = fb
            with _lk:
                _st["log"].append(f"{log_prefix}: primary rate-limited, swapping batch to {fb.name}")
            return fb

    async def generate_one(idx, idea_idx, contents_by_backend):
        async with sem:
            with _lk:
                if _st.get("cancel_requested"):
                    _st["log"].append(f"thumb_{idx:03d}: cancelled")
                    return None
            attempt_cap = REVISION_MAX_ATTEMPTS if revision_mode else 3
            for attempt in range(attempt_cap):
                with _lk:
                    if _st.get("cancel_requested"):
                        _st["log"].append(f"thumb_{idx:03d}: cancelled")
                        return None
                cur_backend = current_backend_ref[0]
                contents = contents_by_backend.get(cur_backend.name)
                if contents is None:
                    with _lk:
                        _st["errors"] += 1
                        _st["log"].append(f"thumb_{idx:03d}: no contents for backend {cur_backend.name}")
                    return None

                # Revision mode: temperature jitter + prompt perturbation schedule.
                if revision_mode:
                    temp = REVISION_TEMPS[attempt % len(REVISION_TEMPS)]
                    pert_level = 0 if attempt < 10 else (1 if attempt < 20 else 2)
                    attempt_contents = _perturb_contents(contents, pert_level)
                    config = types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="16:9"),
                        temperature=temp,
                    )
                    with _lk:
                        _st["log"].append(
                            f"thumb_{idx:03d}: attempt {attempt+1}/{attempt_cap} temp={temp} pert=L{pert_level}"
                        )
                else:
                    attempt_contents = contents
                    config = types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="16:9"),
                    )

                try:
                    _record_api_call(GEMINI_MODEL, attempt_contents, phase=phase, target_status=_st, target_lock=_lk)
                    local_client = _local_clients[cur_backend.name]
                    response = await asyncio.wait_for(
                        local_client.aio.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=attempt_contents,
                            config=config,
                        ),
                        timeout=120,
                    )
                    img_data = None
                    if (response.candidates
                            and response.candidates[0].content
                            and response.candidates[0].content.parts):
                        for part in response.candidates[0].content.parts:
                            if part.inline_data and part.inline_data.data:
                                img_data = part.inline_data.data
                                break

                    if img_data is None:
                        # No image in response — dump full Gemini refusal metadata.
                        info = _diagnose_no_image(response)
                        msg = f"thumb_{idx:03d}: no image | {json.dumps(info, default=str)[:800]}"
                        with _lk:
                            _st["log"].append(msg)
                        print(f"[THUMB] {msg}")
                        if revision_mode and attempt < attempt_cap - 1:
                            continue
                        with _lk:
                            _st["errors"] += 1
                        return None

                    # Revision mode + judge configured: gate on likeness BEFORE saving.
                    if revision_mode and judge_fn is not None:
                        try:
                            verdict = judge_fn(img_data)
                        except Exception as je:
                            verdict = {"match": "no", "confidence": 0,
                                       "reason": f"judge error: {str(je)[:150]}"}
                        match = (verdict or {}).get("match", "no")
                        conf = (verdict or {}).get("confidence", 0)
                        reason = str((verdict or {}).get("reason", ""))[:150]
                        if match != "yes":
                            with _lk:
                                _st["log"].append(
                                    f"thumb_{idx:03d}: attempt {attempt+1} judge REJECTED "
                                    f"(conf={conf} | {reason})"
                                )
                            if attempt < attempt_cap - 1:
                                continue
                            # Out of attempts — record failure, do not save.
                            with _lk:
                                _st["errors"] += 1
                                _st["log"].append(
                                    f"thumb_{idx:03d}: FAILED after {attempt+1} attempts "
                                    f"(judge never matched)"
                                )
                            return None
                        with _lk:
                            _st["log"].append(
                                f"thumb_{idx:03d}: attempt {attempt+1} judge MATCHED "
                                f"(conf={conf} | {reason})"
                            )

                    # Accept: optional border pass, then save.
                    if _st.get("add_border"):
                        if revision_mode:
                            with _lk:
                                _st["log"].append(f"thumb_{idx:03d}: applying Pillow border composite...")
                            try:
                                logo_corner = _st.get("logo_corner", "bottom-left")
                                img_data = apply_border_pillow(img_data, logo_corner)
                            except Exception as e:
                                with _lk:
                                    _st["log"].append(f"thumb_{idx:03d}: Pillow border failed: {str(e)[:150]}")
                        elif cur_backend.border_ref_files:
                            with _lk:
                                _st["log"].append(f"thumb_{idx:03d}: applying border pass...")
                            border_result = await apply_border_pass(cur_backend, img_data, _st.get("border_prompt"), target_status=_st, target_lock=_lk, client_override=local_client)
                            if border_result:
                                img_data = border_result
                                with _lk:
                                    _st["cost"] += COST_PER_IMAGE
                                    _st["session_cost"] += COST_PER_IMAGE
                            else:
                                with _lk:
                                    _st["log"].append(f"thumb_{idx:03d}: border pass failed, using original")

                    filename = f"thumb_{idx:03d}.png"
                    filepath = os.path.join(output_dir, filename)
                    img = Image.open(io.BytesIO(img_data))
                    img.save(filepath, "PNG")

                    with _lk:
                        _st["completed"] += 1
                        _st["cost"] += COST_PER_IMAGE
                        _st["session_cost"] += COST_PER_IMAGE
                        img_entry = {
                            "idx": idx,
                            "path": filepath,
                            "filename": filename,
                            "status": "ok",
                            "idea_idx": idea_idx,
                        }
                        _st["images"].append(img_entry)

                        if idea_idx >= 0:
                            _st["idea_groups"].setdefault(idea_idx, []).append(img_entry)

                        _st["log"].append(
                            f"[{_st['completed']}/{_st['total']}] thumb_{idx:03d}.png OK ({cur_backend.name})"
                        )
                    return filepath

                except asyncio.TimeoutError:
                    with _lk:
                        _st["log"].append(f"thumb_{idx:03d}: attempt {attempt+1} timed out after 120s")
                    if revision_mode and attempt < attempt_cap - 1:
                        continue
                    with _lk:
                        _st["errors"] += 1
                    return None

                except Exception as e:
                    err = str(e)
                    # Rate-limit path: existing backoff-and-swap logic.
                    if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < attempt_cap - 1:
                        new_backend = _maybe_swap(f"thumb_{idx:03d}")
                        if new_backend is cur_backend:
                            wait = (2 ** min(attempt, 5)) + random.random()
                            with _lk:
                                if _st.get("cancel_requested"):
                                    _st["log"].append(f"thumb_{idx:03d}: cancelled")
                                    return None
                                _st["log"].append(
                                    f"thumb_{idx:03d}: rate limited on {cur_backend.name}, retrying in {wait:.1f}s..."
                                )
                            await asyncio.sleep(wait)
                        continue
                    with _lk:
                        _st["log"].append(f"thumb_{idx:03d}: attempt {attempt+1} ERROR — {err[:200]}")
                    if revision_mode and attempt < attempt_cap - 1:
                        continue
                    with _lk:
                        _st["errors"] += 1
                    return None

    tasks = [
        generate_one(start_idx + i + 1, idea_idx, contents_by_backend)
        for i, (idea_idx, _var, contents_by_backend) in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]


def run_generation(backend: GeminiBackend, prompts, output_dir, phase="round1", target_status=None, target_lock=None, revision_mode=False, judge_fn=None):
    """Run async generation in a background thread."""
    _st = target_status if target_status is not None else main_status
    _lk = target_lock if target_lock is not None else main_status_lock
    images_cleared = phase == "round1"
    print(f"[THUMB] run_generation | phase={phase} | prompts={len(prompts)} | images_cleared={images_cleared} | existing_images={len(_st.get('images', []))} | existing_ideas={len(_st.get('ideas', []))}")

    with _lk:
        _st["running"] = True
        _st["cancel_requested"] = False
        _st["phase"] = phase
        _st["total"] = len(prompts)
        _st["completed"] = 0
        _st["errors"] = 0
        _st["log"] = [f"Starting {phase}: generating {len(prompts)} thumbnails..."]
        if phase == "round1":
            _st["images"] = []
            _st["idea_groups"] = {}
            _st["cost"] = 0.0
        elif phase == "revision_page":
            _st["idea_groups"] = {}
            _st["cost"] = 0.0
        _st["done"] = False
        _st["output_dir"] = output_dir

    def _run():
        try:
            results = asyncio.run(
                generate_batch(
                    backend, prompts, output_dir, phase,
                    target_status=_st, target_lock=_lk,
                    revision_mode=revision_mode, judge_fn=judge_fn,
                )
            )
            with _lk:
                _st["log"].append(
                    f"Done! {len(results)} images generated, {_st['errors']} errors. "
                    f"Estimated cost: ${_st['cost']:.2f}"
                )
        except Exception as e:
            with _lk:
                _st["log"].append(f"FATAL ERROR: {e}")
        finally:
            with _lk:
                _st["running"] = False
                _st["done"] = True
                # Cap log to prevent unbounded memory growth
                if len(_st["log"]) > MAX_LOG_ENTRIES:
                    _st["log"] = _st["log"][-MAX_LOG_ENTRIES:]

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def save_metadata(output_dir, info_dict, num_prompts, phase):
    """Save generation metadata for reproducibility."""
    meta = {
        "info": info_dict,
        "generation": {
            "model": GEMINI_MODEL,
            "phase": phase,
            "count_requested": num_prompts,
            "timestamp": datetime.datetime.now().isoformat(),
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
