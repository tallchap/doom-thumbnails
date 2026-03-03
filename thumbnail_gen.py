#!/usr/bin/env python3
"""
Doom Debates Thumbnail Generator v2 — Idea-First Workflow

Generates YouTube thumbnail candidates via Google Gemini image generation,
with a browser UI for idea generation, source image gathering, and iteration.

Usage:
    python thumbnail_gen.py

Opens http://127.0.0.1:9200 in your browser.
"""

import asyncio
import base64
import datetime
import http.server
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import urllib.parse
import webbrowser

# Lock for thread-safe access to the shared `status` dict.
# The background generation thread mutates status["images"],
# status["idea_groups"], status["log"], etc. while the HTTP
# handler reads them for JSON serialization.  Without the lock,
# dict iteration during json.dumps can raise RuntimeError when
# a new key is added to idea_groups concurrently.
status_lock = threading.Lock()

import requests
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

from google import genai
from google.genai import types
from PIL import Image

# ----- Config -----

PORT = int(os.environ.get("PORT", 9200))
APP_USER = os.environ.get("APP_USERNAME", "doom")
APP_PASS = os.environ.get("APP_PASSWORD", "")
GEMINI_MODEL = "gemini-3.1-flash-image-preview"
TEXT_MODEL = "gemini-2.5-flash"
MAX_CONCURRENT = 15
THUMBNAILS_DIR = os.path.join(SCRIPT_DIR, "thumbnails")
EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "doom_debates_thumbnails")
LIRON_DIR = os.path.join(SCRIPT_DIR, "liron_reactions")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
COST_PER_IMAGE = 0.045  # $0.045 per 512px image

BRAND_GUIDE = """BRAND — "DOOM DEBATES":
YouTube channel by Liron Shapira about AI existential risk.
Study the attached brand reference thumbnails for VISUAL STYLE ONLY — colors, composition, typography, energy.
CRITICAL: The brand thumbnails contain images of people — COMPLETELY IGNORE all human faces in the brand references. Do NOT reproduce, copy, or be inspired by ANY faces from the brand references. Use ONLY the colors, layout, typography, and visual energy. The ONLY faces allowed in your output are from separately labeled speaker/host photo sections.
Key brand traits:
- Red-toned background gradient or texture as the base
- Large bold headline text in white or yellow, dramatic and punchy
- Guest/host photos with intense, exaggerated expressions (shock, concern, confrontation)
- Composite imagery — people combined with dramatic AI/apocalyptic visuals
- Overall feel: high-stakes, provocative, debate energy
- Color palette: deep reds, blacks, whites, with yellow accents for emphasis"""

IDEA_GENERATION_PROMPT = """You are a YouTube thumbnail strategist for "Doom Debates", a channel about AI existential risk hosted by Liron Shapira.

Generate exactly 10 thumbnail concept ideas for this episode.

EPISODE TITLE: {title}
{custom_prompt_section}
{transcript_section}

Each idea should be a short, vivid visual description (1-2 sentences) that would make a compelling, clickable thumbnail. Include specific imagery, people positioning, and a suggested headline (1-5 words).

{additional_instructions}

Return as a JSON array of exactly 10 strings. Example:
["Doomsday clock at 11:59 with a terminator looming behind Liron, headline: TIMES UP", "Split screen of human brain vs AI chip, both glowing red, headline: WHO WINS?"]"""

SEARCH_QUERY_PROMPT = """Given this episode info, suggest 3-5 image search queries to find useful source images for a YouTube thumbnail. Return as a JSON array of strings.

EPISODE TITLE: {title}
{custom_prompt_section}

Focus on: guest headshots, topic-relevant imagery (logos, icons, dramatic visuals), anything that could be composited into a thumbnail.
Example: ["Daniel Kokotajlo headshot", "doomsday clock icon", "AI robot dramatic red lighting"]"""

IDEA_THUMBNAIL_PROMPT = """Generate a YouTube thumbnail image.

MOST IMPORTANT RULE — TEXT: The image must contain ONLY ONE text element. This single text element must be 1-5 words. No subtitles, no secondary text, no labels, no captions, no watermarks. Just ONE short punchy headline. If in doubt, use FEWER words.

THUMBNAIL CONCEPT:
{idea_text}

{custom_prompt_section}

BRAND STYLE:
Apply the Doom Debates brand style. Use the attached brand reference thumbnails for visual style ONLY (colors, composition, typography, energy). Do NOT copy any faces or people from the brand references.

{brand_guide}

{speaker_section}

{liron_section}

{additional_instructions}

RULES:
- 16:9 aspect ratio, photorealistic, sharp focus
- Large expressive faces (40-60% of frame)
- High contrast, clean composition, one focal point
- PEOPLE RULE (CRITICAL): The ONLY human faces/people allowed in this thumbnail are: (1) Liron Shapira if his reference photos are attached — his face MUST faithfully match those reference photos, (2) the episode guest if speaker photos are attached. Do NOT generate, copy, or include ANY other human faces. The brand reference thumbnails contain people — IGNORE those people entirely, use ONLY the color/layout/typography style.
- Remember: ONLY 1-5 words of text total in the entire image. ONE text element only.

Variation #{variation_seed} — make this meaningfully different from other variations."""

REVISION_PROMPT = """Revise this YouTube thumbnail for "Doom Debates" podcast.

REVISION INSTRUCTIONS: {custom_prompt}

Keep the core composition but apply the requested changes.
Maintain 16:9 aspect ratio. ONLY 1-5 words of text in the entire image — one short headline, nothing else.
TEXT FIDELITY (CRITICAL): If the revision instructions include text wrapped in quotes (single or double), preserve that quoted text EXACTLY as written — same words, order, punctuation, apostrophes, and capitalization. Do NOT paraphrase, normalize, or substitute synonyms for quoted text.
{speaker_section}
- The ONLY human faces allowed are from the attached speaker/host photos (if any). Do NOT generate faces from brand references.

Variation #{variation_seed} — try something meaningfully different from the other revisions."""

VARIATION_PROMPT = """Create a variation of the attached YouTube thumbnail for "Doom Debates" podcast.
Keep the same general composition, mood, and subject, but vary:
- Color treatment and lighting
- Expression intensity
- Background details and atmosphere

The variation should feel like a sibling of the original, not a copy.
Maintain 16:9 aspect ratio. ONLY 1-5 words of text in the entire image — one short headline, nothing else.
{speaker_section}
- The ONLY human faces allowed are from the attached speaker/host photos (if any). Do NOT generate faces from brand references.

Variation #{variation_seed} — try something meaningfully different."""

# ----- State -----

status = {
    "running": False,
    "phase": "idle",
    "total": 0,
    "completed": 0,
    "errors": 0,
    "log": [],
    "images": [],
    "done": False,
    "output_dir": "",
    "episode_dir": "",
    "speakers": [],
    "sources": [],
    "round_num": 0,
    "ideas": [],
    "idea_groups": {},
    "cost": 0.0,
    "session_cost": 0.0,
}

# ----- API Client & File API -----

BRAND_FILES = []
LIRON_FILES = []

# Keep identity/style references intentionally small and targeted per request.
MAX_BRAND_REFS_PER_CALL = 3
MAX_SPEAKER_REFS_PER_CALL = 4
MAX_LIRON_REFS_PER_CALL = 2


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        print("Add it to .env file or export GEMINI_API_KEY='your-key'")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def upload_brand_references(client):
    """Upload all brand thumbnails to Gemini File API on startup."""
    global BRAND_FILES
    if not os.path.isdir(EXAMPLES_DIR):
        print(f"Brand directory not found: {EXAMPLES_DIR}")
        return
    files = sorted(
        f for f in os.listdir(EXAMPLES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if not files:
        print("No brand reference images found.")
        return
    print(f"Uploading {len(files)} brand references to File API...")
    for i, f in enumerate(files):
        filepath = os.path.join(EXAMPLES_DIR, f)
        try:
            uploaded = client.files.upload(
                file=filepath,
                config=types.UploadFileConfig(display_name=f"brand_doom_debates_{i+1:02d}"),
            )
            BRAND_FILES.append(uploaded)
        except Exception as e:
            print(f"  Failed to upload {f}: {e}")
    print(f"Uploaded {len(BRAND_FILES)} brand references.")


def upload_liron_references(client):
    """Upload Liron reaction photos to Gemini File API on startup."""
    global LIRON_FILES
    if not os.path.isdir(LIRON_DIR):
        print(f"Liron reactions directory not found: {LIRON_DIR}")
        return
    files = sorted(
        f for f in os.listdir(LIRON_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if not files:
        print("No Liron reaction images found.")
        return
    print(f"Uploading {len(files)} Liron reaction photos to File API...")
    for i, f in enumerate(files):
        filepath = os.path.join(LIRON_DIR, f)
        try:
            uploaded = client.files.upload(
                file=filepath,
                config=types.UploadFileConfig(display_name=f"liron_{os.path.splitext(f)[0]}"),
            )
            LIRON_FILES.append(uploaded)
        except Exception as e:
            print(f"  Failed to upload {f}: {e}")
    print(f"Uploaded {len(LIRON_FILES)} Liron reaction photos.")


def upload_files_from_bytes(client, file_bytes_list, name_prefix):
    """Upload raw bytes to Gemini File API, return file reference objects."""
    refs = []
    for i, data in enumerate(file_bytes_list):
        try:
            tmp_path = os.path.join(THUMBNAILS_DIR, f"_tmp_{name_prefix}_{i+1}.jpg")
            with open(tmp_path, "wb") as f:
                f.write(data)
            uploaded = client.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(display_name=f"{name_prefix}_{i+1}"),
            )
            refs.append(uploaded)
            os.remove(tmp_path)
        except Exception as e:
            print(f"  Failed to upload {name_prefix}_{i+1}: {e}")
    return refs


# ----- Brave Search -----


def search_images_brave(queries):
    """Search Brave Image API for source images."""
    if not BRAVE_API_KEY:
        return []
    results = []
    for query in queries:
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/images/search",
                headers={"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"},
                params={"q": query, "count": 5, "safesearch": "off"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                for r in data.get("results", []):
                    thumb_url = ""
                    thumb_obj = r.get("thumbnail", {})
                    if isinstance(thumb_obj, dict):
                        thumb_url = thumb_obj.get("src", "")
                    props = r.get("properties", {})
                    img_url = ""
                    if isinstance(props, dict):
                        img_url = props.get("url", "")
                    if not img_url:
                        img_url = r.get("url", "")
                    if thumb_url or img_url:
                        results.append({
                            "url": img_url,
                            "thumbnail": thumb_url or img_url,
                            "title": r.get("title", ""),
                            "query": query,
                        })
        except Exception as e:
            print(f"Brave search error for '{query}': {e}")
    return results


def download_image_bytes(url, timeout=10):
    """Download an image from URL, return bytes or None."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and "image" in resp.headers.get("content-type", ""):
            return resp.content
    except Exception:
        pass
    return None


# ----- Idea Generation -----


def generate_ideas(client, title, custom_prompt, transcript, additional_instructions):
    """Use Gemini text model to generate 10 thumbnail ideas. Returns list of strings."""
    custom_section = f"CUSTOM PROMPT INFO: {custom_prompt}" if custom_prompt else ""
    transcript_section = f"EPISODE TRANSCRIPT (excerpt):\n{transcript[:1500]}" if transcript else ""
    addl = f"ADDITIONAL INSTRUCTIONS: {additional_instructions}" if additional_instructions else ""

    prompt = IDEA_GENERATION_PROMPT.format(
        title=title,
        custom_prompt_section=custom_section,
        transcript_section=transcript_section,
        additional_instructions=addl,
    )
    response = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return _parse_json_array(text)


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


def generate_search_queries(client, title, custom_prompt):
    """Use Gemini text model to suggest image search queries. Returns list of strings."""
    custom_section = f"CUSTOM PROMPT INFO: {custom_prompt}" if custom_prompt else ""
    prompt = SEARCH_QUERY_PROMPT.format(title=title, custom_prompt_section=custom_section)
    response = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return _parse_json_array(text)


# ----- Prompt Building -----


def _select_identity_refs(refs, max_refs):
    """Choose a small, deterministic subset of identity refs (avoid random mixing)."""
    if not refs:
        return []
    return list(refs[:max_refs])


def _select_brand_refs():
    """Choose a minimal style-only subset of brand refs per generation call."""
    if not BRAND_FILES:
        return []
    return list(BRAND_FILES[:min(MAX_BRAND_REFS_PER_CALL, len(BRAND_FILES))])


def build_idea_prompts(ideas, speaker_refs, source_refs, custom_prompt, additional_instructions, variations_per=3):
    """Build prompt content lists for each idea x N variations.
    Returns list of (idea_idx, variation_num, contents)."""
    custom_section = f"CUSTOM PROMPT INFO: {custom_prompt}" if custom_prompt else ""
    addl = f"ADDITIONAL INSTRUCTIONS: {additional_instructions}" if additional_instructions else ""
    selected_speaker_refs = _select_identity_refs(speaker_refs, MAX_SPEAKER_REFS_PER_CALL)
    speaker_section = (
        "SPEAKER LIKENESS (CRITICAL): A targeted subset of the episode speaker photo library is attached below. "
        "The person(s) in the thumbnail MUST closely resemble these photos — same face, same features, "
        "same skin tone, same hair. Do NOT use generic faces. The speaker photos are the ground truth "
        "for what the person looks like."
        if selected_speaker_refs else ""
    )

    prompts = []
    for idea_idx, idea_text in enumerate(ideas):
        # Detect if Liron is mentioned in this idea
        idea_mentions_liron = "liron" in idea_text.lower()
        selected_liron_refs = _select_identity_refs(LIRON_FILES, MAX_LIRON_REFS_PER_CALL) if idea_mentions_liron else []
        liron_section = (
            "LIRON SHAPIRA (HOST) — CRITICAL FACE MATCH REQUIREMENT:\n"
            "A targeted subset of Liron Shapira reference photos is attached below. "
            "If Liron appears in this thumbnail, his face MUST be a faithful reproduction of the person in these photos — "
            "same facial structure, same nose, same eyes, same beard shape, same skin tone. "
            "Do NOT generate a generic man's face. Do NOT invent features. Copy Liron's exact likeness from the reference photos."
            if selected_liron_refs else ""
        )

        for v in range(variations_per):
            prompt_text = IDEA_THUMBNAIL_PROMPT.format(
                idea_text=idea_text,
                custom_prompt_section=custom_section,
                brand_guide=BRAND_GUIDE,
                speaker_section=speaker_section,
                liron_section=liron_section,
                additional_instructions=addl,
                variation_seed=v + 1,
            )
            contents = [prompt_text]

            brand_sample = _select_brand_refs()
            if brand_sample:
                contents.append("=== DOOM DEBATES BRAND STYLE ONLY — match colors, layout, typography, energy. WARNING: These images contain people — COMPLETELY IGNORE all faces/people in these images. Do NOT reproduce any human likeness from these references. ===")
                contents.extend(brand_sample)

            if selected_liron_refs:
                contents.append(f"=== LIRON SHAPIRA (HOST) REFERENCE PHOTOS — targeted subset ({len(selected_liron_refs)} image(s)) from the Liron library. His face in your output MUST match these photos exactly. ===")
                for i, lf in enumerate(selected_liron_refs):
                    contents.append(f"[Liron photo {i+1} of {len(selected_liron_refs)}]")
                    contents.append(lf)

            if selected_speaker_refs:
                contents.append("=== SPEAKER PHOTOS — targeted subset; the thumbnail MUST use these people's real faces ===")
                contents.extend(selected_speaker_refs)

            if source_refs:
                contents.append("=== SOURCE IMAGES — use these as visual reference material ===")
                contents.extend(source_refs)

            prompts.append((idea_idx, v, contents))

    return prompts


def build_riff_prompts(idea_text, idea_idx, speaker_refs, source_refs, custom_prompt, additional_instructions, count=3):
    """Build prompts for riffing on a single idea."""
    return build_idea_prompts(
        [idea_text], speaker_refs, source_refs, custom_prompt, additional_instructions, variations_per=count
    )


def build_variation_prompts(selected_images, speaker_refs, count_per=3):
    """Build variation prompts from selected images."""
    selected_speaker_refs = _select_identity_refs(speaker_refs, MAX_SPEAKER_REFS_PER_CALL)
    speaker_section = (
        "SPEAKER LIKENESS (CRITICAL): A targeted subset of speaker photos is attached — the person(s) MUST closely "
        "resemble these photos. Same face, features, skin tone, hair."
        if selected_speaker_refs else ""
    )
    prompts = []
    for img in selected_images:
        for v in range(count_per):
            prompt_text = VARIATION_PROMPT.format(
                speaker_section=speaker_section,
                variation_seed=v + 1,
            )
            contents = [prompt_text, img]
            brand_sample = _select_brand_refs()
            if brand_sample:
                contents.append("=== DOOM DEBATES BRAND STYLE ONLY — match colors, layout, typography, energy. WARNING: These images contain people — COMPLETELY IGNORE all faces/people in these images. Do NOT reproduce any human likeness from these references. ===")
                contents.extend(brand_sample)
            if selected_speaker_refs:
                contents.append("=== SPEAKER PHOTOS (targeted subset) ===")
                contents.extend(selected_speaker_refs)
            prompts.append((-1, v, contents))
    return prompts


def build_revision_prompts(selected_images, speaker_refs, custom_prompt, count_per=3, idea_idx=-1, attachment_refs=None):
    """Build revision prompts with custom instructions."""
    selected_speaker_refs = _select_identity_refs(speaker_refs, MAX_SPEAKER_REFS_PER_CALL)
    speaker_section = (
        "SPEAKER LIKENESS (CRITICAL): A targeted subset of speaker photos is attached — the person(s) MUST closely "
        "resemble these photos. Same face, features, skin tone, hair."
        if selected_speaker_refs else ""
    )
    prompts = []
    for img in selected_images:
        for v in range(count_per):
            prompt_text = REVISION_PROMPT.format(
                custom_prompt=custom_prompt,
                speaker_section=speaker_section,
                variation_seed=v + 1,
            )
            contents = [prompt_text, img]
            # Include user-uploaded attachment images in the revision
            if attachment_refs:
                contents.append("=== USER ATTACHMENT IMAGES — incorporate these into the revision as directed ===")
                contents.extend(attachment_refs)
            brand_sample = _select_brand_refs()
            if brand_sample:
                contents.append("=== DOOM DEBATES BRAND STYLE ONLY — match colors, layout, typography, energy. WARNING: These images contain people — COMPLETELY IGNORE all faces/people in these images. Do NOT reproduce any human likeness from these references. ===")
                contents.extend(brand_sample)
            if selected_speaker_refs:
                contents.append("=== SPEAKER PHOTOS (targeted subset) ===")
                contents.extend(selected_speaker_refs)
            prompts.append((idea_idx, v, contents))
    return prompts


# ----- Async Generation -----


async def generate_batch(client, prompts, output_dir, phase="round1"):
    """Fire parallel Gemini API calls and save results.
    prompts is a list of (idea_idx, variation_num, contents)."""
    global status
    os.makedirs(output_dir, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    with status_lock:
        start_idx = len(status["images"])

    async def generate_one(idx, idea_idx, contents):
        async with sem:
            for attempt in range(3):
                try:
                    response = await asyncio.wait_for(
                        client.aio.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                response_modalities=["IMAGE"],
                                image_config=types.ImageConfig(
                                    aspect_ratio="16:9",
                                ),
                            ),
                        ),
                        timeout=120,
                    )
                    if (response.candidates
                            and response.candidates[0].content
                            and response.candidates[0].content.parts):
                        for part in response.candidates[0].content.parts:
                            if part.inline_data and part.inline_data.data:
                                img_data = part.inline_data.data
                                filename = f"thumb_{idx:03d}.png"
                                filepath = os.path.join(output_dir, filename)
                                img = Image.open(io.BytesIO(img_data))
                                img.save(filepath, "PNG")

                                with status_lock:
                                    status["completed"] += 1
                                    status["cost"] += COST_PER_IMAGE
                                    status["session_cost"] += COST_PER_IMAGE
                                    img_entry = {
                                        "idx": idx,
                                        "path": filepath,
                                        "filename": filename,
                                        "status": "ok",
                                        "idea_idx": idea_idx,
                                    }
                                    status["images"].append(img_entry)

                                    if idea_idx >= 0:
                                        status["idea_groups"].setdefault(idea_idx, []).append(img_entry)

                                    status["log"].append(
                                        f"[{status['completed']}/{status['total']}] thumb_{idx:03d}.png OK"
                                    )
                                return filepath

                    with status_lock:
                        status["errors"] += 1
                        status["log"].append(f"thumb_{idx:03d}: no image in response")
                    return None

                except asyncio.TimeoutError:
                    with status_lock:
                        status["errors"] += 1
                        status["log"].append(f"thumb_{idx:03d}: timed out after 120s")
                    return None

                except Exception as e:
                    err = str(e)
                    if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < 2:
                        wait = (2 ** attempt) + random.random()
                        with status_lock:
                            status["log"].append(
                                f"thumb_{idx:03d}: rate limited, retrying in {wait:.1f}s..."
                            )
                        await asyncio.sleep(wait)
                        continue
                    with status_lock:
                        status["errors"] += 1
                        status["log"].append(f"thumb_{idx:03d}: ERROR — {err[:120]}")
                    return None

    tasks = [
        generate_one(start_idx + i + 1, idea_idx, contents)
        for i, (idea_idx, _var, contents) in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]


def run_generation(client, prompts, output_dir, phase="round1"):
    """Run async generation in a background thread."""
    global status
    with status_lock:
        status["running"] = True
        status["phase"] = phase
        status["total"] = len(prompts)
        status["completed"] = 0
        status["errors"] = 0
        status["log"] = [f"Starting {phase}: generating {len(prompts)} thumbnails..."]
        if phase in ("round1", "revision_page"):
            status["images"] = []
            status["idea_groups"] = {}
            status["cost"] = 0.0
        # session_cost is never reset — it accumulates for the entire session
        status["done"] = False
        status["output_dir"] = output_dir

    def _run():
        global status
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                generate_batch(client, prompts, output_dir, phase)
            )
            with status_lock:
                status["log"].append(
                    f"Done! {len(results)} images generated, {status['errors']} errors. "
                    f"Estimated cost: ${status['cost']:.2f}"
                )
        except Exception as e:
            with status_lock:
                status["log"].append(f"FATAL ERROR: {e}")
        finally:
            with status_lock:
                status["running"] = False
                status["done"] = True
            loop.close()

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


# ----- Multipart Parser -----


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


# ----- HTML UI -----

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Doom Debates — Thumbnail Generator v2</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #1a1a2e; color: #e0e0e0; padding: 24px;
    min-height: 100vh;
  }
  h1 { color: #fff; font-size: 26px; margin-bottom: 8px; }
  .subtitle { color: #a0a0b0; font-size: 14px; margin-bottom: 24px; }
  .card {
    background: #16213e; border-radius: 12px; padding: 24px;
    margin-bottom: 16px; border: 1px solid #0f3460;
  }
  .card.hidden { display: none; }
  label.section {
    display: block; font-size: 13px; color: #a0a0b0; margin-bottom: 6px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .field-hint { font-size: 12px; color: #666; margin-top: 4px; margin-bottom: 12px; }
  input[type="text"], textarea {
    width: 100%; padding: 10px 14px; border-radius: 8px;
    border: 1px solid #0f3460; background: #0d1b3e; color: #fff;
    font-size: 15px; outline: none; font-family: inherit;
  }
  textarea { resize: vertical; }
  input[type="text"]:focus, textarea:focus { border-color: #e94560; }
  .mb { margin-bottom: 16px; }

  .file-upload {
    border: 2px dashed #0f3460; border-radius: 10px; padding: 20px;
    text-align: center; cursor: pointer; transition: all 0.2s;
    position: relative;
  }
  .file-upload:hover { border-color: #e94560; }
  .file-upload.dragover { border-color: #e94560; background: rgba(233,69,96,0.08); }
  .file-upload input[type="file"] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer;
  }
  .file-upload .upload-label { color: #a0a0b0; font-size: 14px; }
  .file-upload .upload-label strong { color: #e94560; }
  .file-previews {
    display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;
    justify-content: center;
  }
  .file-previews img {
    width: 80px; height: 50px; object-fit: cover; border-radius: 6px;
    border: 1px solid #0f3460;
  }

  .btn {
    padding: 10px 24px; border-radius: 8px; border: none;
    font-size: 15px; font-weight: 600; cursor: pointer;
    transition: all 0.15s; display: inline-flex; align-items: center; gap: 6px;
  }
  .btn-primary { background: #e94560; color: #fff; }
  .btn-primary:hover { background: #d63851; }
  .btn-primary:disabled { background: #555; cursor: not-allowed; }
  .btn-secondary { background: #0f3460; color: #fff; }
  .btn-secondary:hover { background: #1a4a80; }
  .btn-secondary:disabled { background: #333; cursor: not-allowed; }
  .btn-sm { padding: 6px 14px; font-size: 13px; }

  .progress-bar {
    width: 100%; height: 8px; background: #0d1b3e; border-radius: 4px;
    overflow: hidden; margin: 12px 0;
  }
  .progress-fill {
    height: 100%; background: #e94560; border-radius: 4px;
    transition: width 0.3s;
  }
  .log {
    background: #0d1b3e; border-radius: 8px; padding: 12px;
    font-family: "SF Mono", Monaco, monospace; font-size: 12px;
    max-height: 160px; overflow-y: auto; color: #a0a0b0;
    margin-top: 12px;
  }
  .log div { margin-bottom: 2px; }

  /* Source image gallery */
  .source-gallery {
    display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px;
  }
  .source-item {
    position: relative; width: 120px; border-radius: 8px; overflow: hidden;
    border: 2px solid #0f3460; background: #0d1b3e;
  }
  .source-item img { width: 100%; height: 80px; object-fit: cover; display: block; }
  .source-item .source-title {
    padding: 4px 6px; font-size: 10px; color: #a0a0b0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .source-item .remove-btn {
    position: absolute; top: 4px; right: 4px; background: rgba(233,69,96,0.9);
    color: #fff; border: none; border-radius: 50%; width: 22px; height: 22px;
    cursor: pointer; font-size: 14px; line-height: 22px; text-align: center;
    display: flex; align-items: center; justify-content: center;
  }
  .source-item .remove-btn:hover { background: #e94560; }

  /* Ideas list */
  .idea-item {
    display: flex; gap: 10px; align-items: flex-start; padding: 10px 12px;
    background: #0d1b3e; border-radius: 8px; margin-bottom: 8px;
    border: 1px solid #0f3460;
  }
  .idea-num {
    color: #e94560; font-weight: 700; font-size: 16px; min-width: 28px;
    padding-top: 2px;
  }
  .idea-text {
    flex: 1; color: #e0e0e0; font-size: 14px; line-height: 1.4;
  }
  .idea-text textarea {
    font-size: 14px; padding: 4px 8px; min-height: 40px;
  }
  .idea-actions { display: flex; gap: 6px; align-items: center; }
  .idea-actions button {
    background: none; border: none; cursor: pointer; font-size: 16px;
    color: #a0a0b0; padding: 4px;
  }
  .idea-actions button:hover { color: #e94560; }

  /* Thumbnail grid grouped by idea */
  .idea-group {
    margin-bottom: 24px; border: 1px solid #0f3460; border-radius: 10px;
    padding: 16px; background: #0d1b3e;
  }
  .idea-group-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px; gap: 12px; flex-wrap: wrap;
  }
  .idea-group-header .idea-label {
    color: #e94560; font-weight: 600; font-size: 14px; flex: 1;
  }
  .thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 12px;
  }
  .thumb-card {
    position: relative; border-radius: 8px; overflow: hidden;
    border: 3px solid transparent; cursor: pointer;
    transition: border-color 0.15s, transform 0.1s;
    background: #16213e;
  }
  .thumb-card:hover { transform: scale(1.02); }
  .thumb-card.selected { border-color: #4ade80; }
  .thumb-card.selected::after {
    content: "\2713"; position: absolute; top: 8px; right: 8px;
    background: #4ade80; color: #000; border-radius: 50%;
    width: 28px; height: 28px; display: flex; align-items: center;
    justify-content: center; font-weight: bold; font-size: 16px;
  }
  .thumb-card img {
    width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover;
  }
  .thumb-label {
    padding: 4px 8px; font-size: 11px; color: #a0a0b0;
    display: flex; justify-content: space-between; align-items: center;
  }
  .thumb-dl {
    background: none; border: none; color: #a0a0b0; cursor: pointer;
    font-size: 14px; padding: 2px 6px; border-radius: 4px;
  }
  .thumb-dl:hover { background: #0f3460; color: #4ade80; }

  .actions-bar {
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }
  .selected-count { color: #4ade80; font-weight: 600; font-size: 14px; }

  .riff-panel {
    margin: 12px 0; padding: 14px; background: #16213e;
    border-radius: 8px; border: 1px solid #0f3460;
  }
  .riff-panel .file-upload { padding: 12px; }

  .revision-panel {
    margin-top: 16px; padding: 16px; background: #0d1b3e;
    border-radius: 10px; border: 1px solid #0f3460;
    display: none;
  }
  .revision-panel.visible { display: block; }
  .revision-panel textarea { margin-bottom: 12px; }
  .revision-panel .revision-actions { display: flex; gap: 12px; }

  .section-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 16px; flex-wrap: wrap; gap: 12px;
  }
  .section-header h2 { color: #fff; font-size: 20px; }

  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; background: #0f3460; color: #a0a0b0; }
  .tag-required { background: #e94560; color: #fff; }

  .cost-display {
    position: fixed; top: 16px; left: 16px; background: #16213e;
    border: 1px solid #0f3460; border-radius: 8px; padding: 8px 14px;
    font-size: 13px; color: #a0a0b0; z-index: 100;
  }
  .cost-display strong { color: #4ade80; }

  /* Computation window — fixed upper-right */
  .compute-window {
    position: fixed; top: 16px; right: 16px; width: 380px;
    background: #16213e; border: 1px solid #0f3460; border-radius: 10px;
    z-index: 200; box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    font-size: 13px; overflow: hidden;
    transition: height 0.2s;
  }
  .compute-window.collapsed {
    height: auto;
  }
  .compute-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; background: #0f3460; cursor: pointer;
    user-select: none;
  }
  .compute-header .compute-title {
    color: #fff; font-weight: 600; font-size: 13px;
    display: flex; align-items: center; gap: 8px;
  }
  .compute-header .compute-toggle {
    color: #a0a0b0; font-size: 18px; background: none; border: none;
    cursor: pointer; padding: 0 4px;
  }
  .compute-header .compute-toggle:hover { color: #fff; }
  .compute-body {
    padding: 12px 14px; max-height: 400px; overflow-y: auto;
  }
  .compute-window.collapsed .compute-body { display: none; }
  .compute-status {
    display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
    color: #a0a0b0;
  }
  .compute-status .status-dot {
    width: 8px; height: 8px; border-radius: 50%; background: #555;
    flex-shrink: 0;
  }
  .compute-status .status-dot.active {
    background: #4ade80; animation: pulse-dot 1.5s infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .compute-progress {
    width: 100%; height: 6px; background: #0d1b3e; border-radius: 3px;
    overflow: hidden; margin: 8px 0;
  }
  .compute-progress-fill {
    height: 100%; background: #e94560; border-radius: 3px;
    transition: width 0.3s;
  }
  .compute-log {
    background: #0d1b3e; border-radius: 6px; padding: 8px 10px;
    font-family: "SF Mono", Monaco, monospace; font-size: 11px;
    max-height: 220px; overflow-y: auto; color: #a0a0b0;
    margin-top: 8px;
  }
  .compute-log div { margin-bottom: 2px; }
  .compute-cost {
    margin-top: 8px; color: #a0a0b0; font-size: 12px;
  }
  .compute-cost strong { color: #4ade80; }

  .step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%; background: #e94560;
    color: #fff; font-weight: 700; font-size: 14px; margin-right: 8px;
    flex-shrink: 0;
  }

  .spinner {
    display: inline-block; width: 18px; height: 18px;
    border: 2px solid #a0a0b0; border-top-color: #e94560;
    border-radius: 50%; animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .btn-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
</style>
</head>
<body>

<h1>Doom Debates — Thumbnail Generator</h1>
<div class="subtitle">Model: Nano Banana 2 (gemini-3.1-flash-image-preview) — Idea-first workflow</div>
<div style="margin-bottom:16px;"><a href="/revision" style="color:#4ade80;text-decoration:none;font-weight:600;">→ Open Thumbnail Revision Page</a></div>

<!-- STEP 1: INPUTS -->
<div class="card" id="inputCard">
  <div class="section-header">
    <h2><span class="step-num">1</span> Episode Info</h2>
  </div>

  <div class="mb">
    <label class="section">Episode Title <span class="tag tag-required">Required</span></label>
    <input type="text" id="episodeTitle" placeholder="e.g. Is GPT-5 the End? with Daniel Kokotajlo">
  </div>

  <div class="mb">
    <label class="section">Custom Prompt Info <span class="tag">Optional</span></label>
    <textarea id="customPrompt" rows="2" placeholder="e.g. Guest: Daniel Kokotajlo, former OpenAI researcher. Focus on AI timelines and existential risk."></textarea>
    <div class="field-hint">Any context that should inform the thumbnail concepts and generation.</div>
  </div>

  <div class="mb">
    <label class="section">Episode Transcript <span class="tag">Optional</span></label>
    <div class="file-upload" id="transcriptUpload">
      <input type="file" id="transcriptFile" accept=".txt,.md,.text,.rtf,.doc,.docx,.pdf,.srt,.vtt">
      <div class="upload-label"><strong>Click to browse</strong> or drag & drop a transcript file</div>
      <div id="transcriptFileName" style="color:#4ade80;margin-top:8px;font-size:13px;"></div>
    </div>
  </div>

  <div class="mb">
    <label class="section">Speaker Images <span class="tag">Optional</span></label>
    <div class="file-upload" id="speakerUpload">
      <input type="file" id="speakerFiles" multiple accept="image/*">
      <div class="upload-label"><strong>Click to browse</strong> or drag & drop speaker photos</div>
      <div class="file-previews" id="speakerPreviews"></div>
    </div>
    <div class="field-hint">Photos of speakers — the generated faces will match these likenesses.</div>
  </div>

  <div class="mb">
    <label class="section">Source Images <span class="tag">Optional</span></label>
    <div class="file-upload" id="sourceUpload">
      <input type="file" id="sourceFiles" multiple accept="image/*">
      <div class="upload-label"><strong>Click to browse</strong> or drag & drop logos, reference visuals, etc.</div>
      <div class="file-previews" id="sourcePreviews"></div>
    </div>
    <div class="field-hint">Logos, icons, or reference images to use as visual material (not faces).</div>
  </div>

  <div class="mb">
    <label class="section">Additional Instructions <span class="tag">Optional — affects all future generations</span></label>
    <textarea id="additionalInstructions" rows="2" placeholder="e.g. Make Liron look more concerned. Use mushroom clouds. Always include a doomsday clock."></textarea>
  </div>

  <div class="btn-row">
    <button class="btn btn-secondary" id="gatherBtn" onclick="gatherSourceImages()">Gather Source Images from Web</button>
    <button class="btn btn-primary" id="skipToIdeasBtn" onclick="skipToIdeas()">Skip to Generate Ideas</button>
  </div>
</div>

<!-- STEP 2: SOURCE IMAGE GALLERY -->
<div class="card hidden" id="sourceGalleryCard">
  <div class="section-header">
    <h2><span class="step-num">2</span> Source Images</h2>
    <span id="gatherStatus" style="color:#a0a0b0;font-size:13px;"></span>
  </div>
  <p style="color:#a0a0b0;font-size:13px;margin-bottom:12px;">Click X to remove images you don't want. These will be used as visual reference in thumbnail generation.</p>
  <div class="source-gallery" id="sourceGallery"></div>
  <div class="btn-row">
    <button class="btn btn-primary" onclick="proceedToIdeas()">Continue to Generate Ideas</button>
    <button class="btn btn-secondary" onclick="document.getElementById('addMoreSourceInput').click()">Upload More</button>
    <input type="file" id="addMoreSourceInput" multiple accept="image/*" style="display:none" onchange="addMoreSourceImages(this)">
  </div>
</div>

<!-- STEP 3: IDEAS -->
<div class="card hidden" id="ideasCard">
  <div class="section-header">
    <h2><span class="step-num">3</span> Thumbnail Ideas</h2>
    <span id="ideasStatus" style="color:#a0a0b0;font-size:13px;"></span>
  </div>
  <div id="ideasList"></div>
  <div class="btn-row">
    <button class="btn btn-primary" id="genThumbsBtn" onclick="generateThumbnails()">Generate Thumbnails (3 per idea)</button>
    <button class="btn btn-secondary" onclick="generateMoreIdeas()">Generate More Ideas</button>
  </div>
</div>

<!-- STEP 4: THUMBNAILS GRID -->
<div class="card hidden" id="gridCard">
  <div class="section-header">
    <h2 id="gridLabel">Select Your Favorites</h2>
    <div class="actions-bar">
      <span class="selected-count"><span id="selectedCount">0</span> selected</span>
    </div>
  </div>

  <div class="revision-panel" id="revisionPanel">
    <label class="section">Revise Selected</label>
    <textarea id="revisionPrompt" rows="2" placeholder="e.g. Make the background darker, add more fire, change text to EXTINCTION..."></textarea>
    <div class="mb" style="margin-top:8px;">
      <label class="section">Attachments <span class="tag">Optional</span></label>
      <div class="file-upload" id="revisionUpload" style="padding:12px;">
        <input type="file" id="revisionFiles" multiple accept="image/*">
        <div class="upload-label"><strong>Click to browse</strong> or drag & drop images to include in the revision</div>
        <div class="file-previews" id="revisionPreviews"></div>
      </div>
    </div>
    <div class="revision-actions">
      <button class="btn btn-primary btn-sm" onclick="startRevision()">Revise with Prompt</button>
    </div>
  </div>

  <div id="ideaGroupsContainer"></div>

</div>

<!-- COMPUTATION WINDOW — fixed upper-right -->
<div class="compute-window collapsed" id="computeWindow">
  <div class="compute-header" onclick="toggleComputeWindow()">
    <div class="compute-title">
      <span class="status-dot" id="computeDot"></span>
      <span id="computeTitle">Computation</span>
    </div>
    <button class="compute-toggle" id="computeToggleBtn">&#9660;</button>
  </div>
  <div class="compute-body" id="computeBody">
    <div class="compute-status">
      <span id="computePhase">Idle</span>
      <span id="computeProgressText" style="margin-left:auto;"></span>
    </div>
    <div class="compute-progress">
      <div class="compute-progress-fill" id="computeProgressFill" style="width:0%"></div>
    </div>
    <div class="compute-log" id="computeLog"></div>
    <div class="compute-cost" id="computeCost" style="display:none;">
      Est. cost: <strong id="costAmount">$0.00</strong>
    </div>
  </div>
</div>

<!-- Session cost display — fixed top-left -->
<div class="cost-display" id="sessionCostDisplay">Session cost: <strong>$0.00</strong></div>

<script>
// ----- State -----
let ideas = [];
let sourceImages = [];  // {url, thumbnail, title, selected, bytes_uploaded}
let selected = new Set();
let allImages = [];
let pollInterval = null;
let transcript = '';

// ----- Client-side session cost tracking -----
// Persists in sessionStorage so it survives page refreshes and server restarts.
let clientSessionCost = parseFloat(sessionStorage.getItem('doomSessionCost') || '0');
let lastServerCost = 0;  // tracks the last server-reported session_cost to detect deltas

function updateCostDisplay(cost) {
  clientSessionCost = cost;
  sessionStorage.setItem('doomSessionCost', cost.toFixed(4));
  const fixedCost = document.getElementById('sessionCostDisplay');
  if (fixedCost) fixedCost.innerHTML = 'Session cost: <strong>$' + cost.toFixed(2) + '</strong>';
  const costAmount = document.getElementById('costAmount');
  if (costAmount) {
    costAmount.textContent = '$' + cost.toFixed(2);
    document.getElementById('computeCost').style.display = 'block';
  }
}

// Show saved cost immediately on page load
updateCostDisplay(clientSessionCost);

// Fetch current status on page load to sync cost & enable riff buttons if idle
fetch('/status').then(r => r.json()).then(data => {
  // Sync server cost into client tracking
  if (data.session_cost > 0) {
    lastServerCost = data.session_cost;
    // Take the higher of client-tracked or server-reported
    if (data.session_cost > clientSessionCost) {
      updateCostDisplay(data.session_cost);
    }
  }
  // If server is idle (not running), make sure riff buttons are enabled
  if (data.done || !data.running) {
    setRiffButtonsDisabled(false);
  }
}).catch(() => {});

// ----- File upload helpers -----
function setupFileUpload(uploadId, inputId, previewId) {
  const zone = document.getElementById(uploadId);
  const input = document.getElementById(inputId);
  const previews = document.getElementById(previewId);
  if (!zone || !input) return;

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      const dt = new DataTransfer();
      for (const f of input.files) dt.items.add(f);
      for (const f of e.dataTransfer.files) dt.items.add(f);
      input.files = dt.files;
      if (previews) showPreviews(input, previews);
    }
  });
  if (previews) input.addEventListener('change', () => showPreviews(input, previews));
}

function showPreviews(input, container) {
  container.innerHTML = '';
  for (const file of input.files) {
    if (!file.type.startsWith('image/')) continue;
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    container.appendChild(img);
  }
}

setupFileUpload('speakerUpload', 'speakerFiles', 'speakerPreviews');
setupFileUpload('sourceUpload', 'sourceFiles', 'sourcePreviews');
setupFileUpload('revisionUpload', 'revisionFiles', 'revisionPreviews');

// Transcript
(function() {
  const zone = document.getElementById('transcriptUpload');
  const input = document.getElementById('transcriptFile');
  const nameEl = document.getElementById('transcriptFileName');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      input.files = e.dataTransfer.files;
      nameEl.textContent = input.files[0].name;
    }
  });
  input.addEventListener('change', () => {
    nameEl.textContent = input.files.length ? input.files[0].name : '';
  });
})();

// ----- Step 2: Gather Source Images -----
async function gatherSourceImages() {
  const title = document.getElementById('episodeTitle').value.trim();
  if (!title) { alert('Please enter an episode title.'); return; }

  const btn = document.getElementById('gatherBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Searching...';

  try {
    const fd = new FormData();
    fd.append('title', title);
    fd.append('custom_prompt', document.getElementById('customPrompt').value);

    const resp = await fetch('/gather_images', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }

    sourceImages = (data.images || []).map((img, i) => ({...img, id: i, removed: false}));
    renderSourceGallery();
    document.getElementById('sourceGalleryCard').classList.remove('hidden');
    document.getElementById('gatherStatus').textContent = `Found ${sourceImages.length} images`;
  } catch(e) {
    alert('Error: ' + e);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Gather Source Images from Web';
  }
}

function renderSourceGallery() {
  const gallery = document.getElementById('sourceGallery');
  gallery.innerHTML = '';
  sourceImages.forEach((img, i) => {
    if (img.removed) return;
    const div = document.createElement('div');
    div.className = 'source-item';
    div.innerHTML =
      '<img src="' + escHtml(img.thumbnail || img.url) + '" onerror="this.parentElement.style.display=\'none\'">' +
      '<div class="source-title">' + escHtml(img.title || img.query || '') + '</div>' +
      '<button class="remove-btn" onclick="removeSourceImage(' + i + ')">&times;</button>';
    gallery.appendChild(div);
  });
}

function removeSourceImage(idx) {
  sourceImages[idx].removed = true;
  renderSourceGallery();
}

function addMoreSourceImages(input) {
  for (const file of input.files) {
    const url = URL.createObjectURL(file);
    sourceImages.push({url: url, thumbnail: url, title: file.name, removed: false, localFile: file});
  }
  renderSourceGallery();
}

function skipToIdeas() {
  const title = document.getElementById('episodeTitle').value.trim();
  if (!title) { alert('Please enter an episode title.'); return; }
  proceedToIdeas();
}

// ----- Step 3: Generate Ideas -----
async function proceedToIdeas() {
  const title = document.getElementById('episodeTitle').value.trim();
  if (!title) { alert('Please enter an episode title.'); return; }

  const card = document.getElementById('ideasCard');
  card.classList.remove('hidden');
  document.getElementById('ideasStatus').innerHTML = '<span class="spinner"></span> Generating ideas...';

  // Read transcript if provided (truncate — server only uses first 1500 chars anyway)
  const fileInput = document.getElementById('transcriptFile');
  if (fileInput.files.length > 0 && !transcript) {
    transcript = await fileInput.files[0].text();
  }

  try {
    const fd = new FormData();
    fd.append('title', title);
    fd.append('custom_prompt', document.getElementById('customPrompt').value);
    fd.append('transcript', transcript.slice(0, 5000));
    fd.append('additional_instructions', document.getElementById('additionalInstructions').value);

    const resp = await fetch('/generate_ideas', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) { alert(data.error); document.getElementById('ideasStatus').textContent = 'Error'; return; }

    ideas = data.ideas || [];
    renderIdeas();
    document.getElementById('ideasStatus').textContent = ideas.length + ' ideas generated';
  } catch(e) {
    alert('Error: ' + e);
    document.getElementById('ideasStatus').textContent = 'Error';
  }
}

async function generateMoreIdeas() {
  flushActiveEdits();
  const title = document.getElementById('episodeTitle').value.trim();
  if (!title) { alert('Please enter an episode title.'); return; }

  document.getElementById('ideasCard').classList.remove('hidden');
  document.getElementById('ideasStatus').innerHTML = '<span class="spinner"></span> Generating more ideas...';

  try {
    const fd = new FormData();
    fd.append('title', title);
    fd.append('custom_prompt', document.getElementById('customPrompt').value);
    fd.append('transcript', transcript.slice(0, 5000));
    fd.append('additional_instructions', document.getElementById('additionalInstructions').value);
    fd.append('existing_ideas', JSON.stringify(ideas));

    const resp = await fetch('/more_ideas', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }

    const newIdeas = data.ideas || [];
    ideas = ideas.concat(newIdeas);
    renderIdeas();
    document.getElementById('ideasStatus').textContent = ideas.length + ' ideas total';
  } catch(e) {
    alert('Error: ' + e);
  }
}

function renderIdeas() {
  const list = document.getElementById('ideasList');
  list.innerHTML = '';
  ideas.forEach((idea, i) => {
    const div = document.createElement('div');
    div.className = 'idea-item';
    div.id = 'idea-' + i;
    div.innerHTML =
      '<span class="idea-num">' + (i+1) + '</span>' +
      '<div class="idea-text" id="idea-text-' + i + '">' + escHtml(idea) + '</div>' +
      '<div class="idea-actions">' +
        '<button onclick="editIdea(' + i + ')" title="Edit">&#9998;</button>' +
        '<button onclick="deleteIdea(' + i + ')" title="Delete">&times;</button>' +
      '</div>';
    list.appendChild(div);
  });
}

function editIdea(idx) {
  const el = document.getElementById('idea-text-' + idx);
  const current = ideas[idx];
  el.innerHTML = '<textarea oninput="ideas[' + idx + '] = this.value.trim() || ideas[' + idx + ']" onblur="finishEditIdea(' + idx + ')" onkeydown="if(event.key===\'Enter\'&&!event.shiftKey){event.preventDefault();this.blur();}">' + escHtml(current) + '</textarea>';
  el.querySelector('textarea').focus();
}

function finishEditIdea(idx) {
  // ideas[idx] is already updated by oninput — just re-render to show static text
  renderIdeas();
}

function deleteIdea(idx) {
  ideas.splice(idx, 1);
  renderIdeas();
}

// ----- Helpers -----
function flushActiveEdits() {
  // Save any idea that's currently being edited in a textarea
  const ideasList = document.getElementById('ideasList');
  if (!ideasList) return;
  const textareas = ideasList.querySelectorAll('textarea');
  textareas.forEach(ta => {
    const match = ta.parentElement && ta.parentElement.id && ta.parentElement.id.match(/idea-text-(\d+)/);
    if (match) {
      const idx = parseInt(match[1]);
      const val = ta.value.trim();
      if (val && idx < ideas.length) ideas[idx] = val;
    }
  });
}

function setRiffButtonsDisabled(disabled) {
  // Disable BOTH the outer "Riff" toggle buttons AND the inner
  // "Generate Riffs" buttons inside open panels.
  document.querySelectorAll('.riff-btn, .riff-generate-btn').forEach(btn => {
    btn.disabled = disabled;
    if (disabled) btn.title = 'Wait for current generation to finish';
    else btn.title = '';
  });
}

// ----- Step 4: Generate Thumbnails -----
async function generateThumbnails() {
  flushActiveEdits();
  if (ideas.length === 0) { alert('No ideas to generate thumbnails for.'); return; }

  const btn = document.getElementById('genThumbsBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Starting...';

  const fd = new FormData();
  fd.append('title', document.getElementById('episodeTitle').value);
  fd.append('ideas', JSON.stringify(ideas));
  fd.append('custom_prompt', document.getElementById('customPrompt').value);
  fd.append('additional_instructions', document.getElementById('additionalInstructions').value);
  fd.append('transcript', transcript.slice(0, 5000));

  // Attach speaker images
  for (const f of document.getElementById('speakerFiles').files) {
    fd.append('speakers', f);
  }

  // Attach locally uploaded source images
  for (const f of document.getElementById('sourceFiles').files) {
    fd.append('sources', f);
  }

  // Attach web-gathered source images (send URLs for server to download)
  const selectedSources = sourceImages.filter(s => !s.removed).map(s => s.url).filter(Boolean);
  fd.append('source_urls', JSON.stringify(selectedSources));

  try {
    const resp = await fetch('/generate_from_ideas', { method: 'POST', body: fd, credentials: 'same-origin' });
    const data = await resp.json();
    if (data.error) { alert(data.error); btn.disabled = false; btn.innerHTML = 'Generate Thumbnails (3 per idea)'; return; }

    selected.clear();
    allImages = [];
    document.getElementById('ideaGroupsContainer').innerHTML = '';
    updateSelectedUI();

    // Sync cost baseline — take current value so deltas apply correctly
    fetch('/status').then(r => r.json()).then(d => {
      if (d.session_cost > lastServerCost) lastServerCost = d.session_cost;
    }).catch(() => {});

    document.getElementById('gridCard').classList.remove('hidden');
    showComputeWindow();
    startPolling();
  } catch(e) {
    alert('Error: ' + e);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Generate Thumbnails (3 per idea)';
  }
}

function toggleRiffPanel(ideaIdx) {
  const panel = document.getElementById('riff-panel-' + ideaIdx);
  if (!panel) return;
  panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
}

function showRiffPreviews(ideaIdx) {
  const input = document.getElementById('riff-images-' + ideaIdx);
  const container = document.getElementById('riff-previews-' + ideaIdx);
  if (!input || !container) return;
  container.innerHTML = '';
  for (const file of input.files) {
    if (!file.type.startsWith('image/')) continue;
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    container.appendChild(img);
  }
}

async function executeRiff(ideaIdx) {
  flushActiveEdits();
  const idea = ideas[ideaIdx];
  if (!idea) { alert('Idea not found at index ' + ideaIdx); return; }

  const riffPrompt = (document.getElementById('riff-prompt-' + ideaIdx) || {}).value || '';
  const riffImagesInput = document.getElementById('riff-images-' + ideaIdx);

  const riffCountInput = document.getElementById('riff-count-' + ideaIdx);
  const riffCount = riffCountInput ? parseInt(riffCountInput.value) || 3 : 3;

  // Disable ALL riff buttons immediately to prevent concurrent requests
  setRiffButtonsDisabled(true);

  const fd = new FormData();
  fd.append('idea_text', idea);
  fd.append('idea_idx', ideaIdx);
  fd.append('custom_prompt', (document.getElementById('customPrompt') || {}).value || '');
  fd.append('additional_instructions', (document.getElementById('additionalInstructions') || {}).value || '');
  fd.append('riff_prompt', riffPrompt);
  fd.append('riff_count', riffCount);

  // Do NOT resend speakers/sources/source_urls — server reuses stored refs from initial generation.
  // Only send riff-specific images if the user added them.
  if (riffImagesInput && riffImagesInput.files.length > 0) {
    for (const f of riffImagesInput.files) {
      fd.append('riff_images', f);
    }
  }

  try {
    const resp = await fetch('/riff_idea', { method: 'POST', body: fd, credentials: 'same-origin' });
    if (!resp.ok) {
      alert('Riff request failed (HTTP ' + resp.status + '). Try again.');
      setRiffButtonsDisabled(false);
      return;
    }
    const data = await resp.json();
    if (data.error) {
      alert(data.error);
      setRiffButtonsDisabled(false);
      return;
    }

    // Register riff as a new idea so addImageToGrid can find its text
    const riffIdeaIdx = data.riff_idea_idx;
    while (ideas.length <= riffIdeaIdx) ideas.push('');
    ideas[riffIdeaIdx] = '(Riff on Idea ' + (ideaIdx + 1) + ') ' + idea;

    // Collapse the riff panel
    const panel = document.getElementById('riff-panel-' + ideaIdx);
    if (panel) panel.style.display = 'none';

    // Sync cost baseline — take current value so deltas apply correctly
    fetch('/status').then(r => r.json()).then(d => {
      if (d.session_cost > lastServerCost) lastServerCost = d.session_cost;
    }).catch(() => {});

    document.getElementById('gridCard').classList.remove('hidden');
    showComputeWindow();
    startPolling();
    // Buttons will be re-enabled by pollStatus when data.done === true
  } catch(e) {
    alert('Riff error: ' + e);
    setRiffButtonsDisabled(false);
  }
}

// ----- Computation Window -----
function toggleComputeWindow() {
  const win = document.getElementById('computeWindow');
  const btn = document.getElementById('computeToggleBtn');
  if (win.classList.contains('collapsed')) {
    win.classList.remove('collapsed');
    btn.innerHTML = '&#9650;';
  } else {
    win.classList.add('collapsed');
    btn.innerHTML = '&#9660;';
  }
}

function showComputeWindow() {
  const win = document.getElementById('computeWindow');
  win.style.display = 'block';
  win.classList.remove('collapsed');
  document.getElementById('computeToggleBtn').innerHTML = '&#9650;';
}

// ----- Polling -----
function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(pollStatus, 500);
}

function pollStatus() {
  fetch('/status').then(r => r.json()).then(data => {
    const doneCount = data.completed + data.errors;
    const pct = data.total > 0 ? (doneCount / data.total * 100) : 0;

    // Update computation window
    document.getElementById('computeProgressFill').style.width = pct + '%';
    document.getElementById('computeProgressText').textContent =
      data.completed + ' / ' + data.total + (data.errors > 0 ? ' (' + data.errors + ' failed)' : '');

    const phaseNames = {
      round1: 'Generating Thumbnails...',
      riff: 'Riffing on Idea...',
      revision: 'Revising...',
      variation: 'Generating Variations...'
    };
    const phaseName = phaseNames[data.phase] || 'Generating...';
    document.getElementById('computePhase').textContent = phaseName;
    document.getElementById('computeTitle').textContent = data.done ? 'Computation — Done' : phaseName;

    // Status dot
    const dot = document.getElementById('computeDot');
    if (data.done) {
      dot.classList.remove('active');
    } else {
      dot.classList.add('active');
    }

    // Log
    const logArea = document.getElementById('computeLog');
    logArea.innerHTML = data.log.slice(-50).map(l => '<div>' + escHtml(l) + '</div>').join('');
    logArea.scrollTop = logArea.scrollHeight;

    // Update cost (session-wide aggregate, tracked client-side)
    if (data.session_cost !== undefined && data.session_cost > 0) {
      const delta = data.session_cost - lastServerCost;
      if (delta > 0) {
        clientSessionCost += delta;
        lastServerCost = data.session_cost;
      }
      // Always take the max of server vs client
      const displayCost = Math.max(clientSessionCost, data.session_cost);
      updateCostDisplay(displayCost);
    }

    // Add new images to grid
    let addedNew = false;
    while (allImages.length < data.images.length) {
      const img = data.images[allImages.length];
      allImages.push(img);
      addImageToGrid(img);
      addedNew = true;
    }

    // Auto-scroll to the newest group when new images arrive
    if (addedNew) {
      const container = document.getElementById('ideaGroupsContainer');
      const groups = container.children;
      if (groups.length > 0) {
        const lastGroup = groups[groups.length - 1];
        lastGroup.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }

    // Disable riff buttons while generating, re-enable when done
    setRiffButtonsDisabled(!data.done);

    if (data.done) {
      clearInterval(pollInterval);
      pollInterval = null;
      // Collapse compute window after a delay
      setTimeout(() => {
        const win = document.getElementById('computeWindow');
        // Don't collapse if user is hovering
        win.classList.add('collapsed');
        document.getElementById('computeToggleBtn').innerHTML = '&#9660;';
      }, 3000);
    }
  }).catch(() => {});
}

function addImageToGrid(img) {
  const ideaIdx = img.idea_idx;
  const container = document.getElementById('ideaGroupsContainer');

  // Find or create idea group
  let group = document.getElementById('idea-group-' + ideaIdx);
  if (!group) {
    group = document.createElement('div');
    group.className = 'idea-group';
    group.id = 'idea-group-' + ideaIdx;

    const ideaText = (ideaIdx >= 0 && ideaIdx < ideas.length) ? ideas[ideaIdx] : 'Variation';
    group.innerHTML =
      '<div class="idea-group-header">' +
        '<div class="idea-label">' + (ideaIdx >= 0 ? '<strong>Idea ' + (ideaIdx+1) + ':</strong> ' : '') + escHtml(ideaText) + '</div>' +
      '</div>' +
      '<div class="thumb-grid" id="idea-grid-' + ideaIdx + '"></div>';
    container.appendChild(group);
  }

  const grid = document.getElementById('idea-grid-' + ideaIdx);
  const card = document.createElement('div');
  card.className = 'thumb-card';
  card.dataset.idx = img.idx;
  card.onclick = () => toggleSelect(img.idx);
  card.innerHTML =
    '<img src="/image?path=' + encodeURIComponent(img.path) + '" loading="lazy">' +
    '<div class="thumb-label"><span>#' + img.idx + '</span>' +
    '<button class="thumb-dl" onclick="event.stopPropagation(); downloadImage(\'' + encodeURIComponent(img.path) + '\', \'thumb_' + img.idx + '.png\')" title="Download">&#8681;</button></div>';
  grid.appendChild(card);
}

// ----- Selection -----
function toggleSelect(idx) {
  const card = document.querySelector('[data-idx="' + idx + '"]');
  if (!card) return;
  if (selected.has(idx)) {
    selected.delete(idx);
    card.classList.remove('selected');
  } else {
    selected.add(idx);
    card.classList.add('selected');
  }
  updateSelectedUI();
}

function updateSelectedUI() {
  document.getElementById('selectedCount').textContent = selected.size;
  const panel = document.getElementById('revisionPanel');
  if (selected.size > 0) panel.classList.add('visible');
  else panel.classList.remove('visible');
}

function startRevision() {
  const prompt = document.getElementById('revisionPrompt').value.trim();
  if (!prompt) { alert('Enter revision instructions.'); return; }
  const indices = Array.from(selected).join(',');

  // Determine which idea(s) the selected images belong to for labeling
  const selectedIdeas = new Set();
  allImages.filter(img => selected.has(img.idx)).forEach(img => {
    if (img.idea_idx >= 0 && img.idea_idx < ideas.length) {
      selectedIdeas.add(img.idea_idx);
    }
  });
  const sourceIdeasJson = JSON.stringify(Array.from(selectedIdeas));

  const fd = new FormData();
  fd.append('indices', indices);
  fd.append('prompt', prompt);
  fd.append('source_idea_indices', sourceIdeasJson);

  // Include attachment files if any
  const revisionFiles = document.getElementById('revisionFiles');
  if (revisionFiles && revisionFiles.files.length > 0) {
    for (const file of revisionFiles.files) {
      fd.append('revision_images', file);
    }
  }

  fetch('/revise', {
    method: 'POST',
    body: fd,
  }).then(r => r.json()).then(data => {
    if (data.error) { alert(data.error); return; }

    // Register the revision as a new idea so addImageToGrid shows the label
    const revIdeaIdx = data.revision_idea_idx;
    if (revIdeaIdx !== undefined) {
      while (ideas.length <= revIdeaIdx) ideas.push('');
      ideas[revIdeaIdx] = data.revision_label || ('(Revision) ' + prompt);
    }

    document.getElementById('gridLabel').textContent = 'Revisions \u2014 Select Your Finals';
    selected.clear(); updateSelectedUI();
    // Clear revision attachments
    if (revisionFiles) revisionFiles.value = '';
    const previews = document.getElementById('revisionPreviews');
    if (previews) previews.innerHTML = '';
    showComputeWindow();
    startPolling();
  });
}

function downloadImage(encodedPath, filename) {
  fetch('/download?path=' + encodedPath)
    .then(r => {
      if (!r.ok) throw new Error('Download failed: ' + r.status);
      return r.blob();
    })
    .then(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    })
    .catch(e => alert('Download error: ' + e.message));
}

function downloadSelected() {
  if (selected.size === 0) return;
  const imgs = allImages.filter(img => selected.has(img.idx));
  let delay = 0;
  imgs.forEach(img => {
    setTimeout(() => downloadImage(encodeURIComponent(img.path), 'thumb_' + img.idx + '.png'), delay);
    delay += 300;
  });
}

function escHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>"""

HTML_REVISION = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Doom Debates — Thumbnail Revision</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#1a1a2e; color:#e0e0e0; margin:0; padding:24px; }
  .card { background:#16213e; border:1px solid #0f3460; border-radius:12px; padding:20px; margin-bottom:16px; }
  h1 { margin:0 0 8px; }
  .subtitle { color:#a0a0b0; margin-bottom:14px; }
  label { display:block; margin:10px 0 6px; color:#a0a0b0; font-size:13px; text-transform:uppercase; }
  input[type="file"], textarea { width:100%; background:#0d1b3e; border:1px solid #0f3460; color:#fff; border-radius:8px; padding:10px; }
  textarea { min-height:90px; resize:vertical; }
  .btn { background:#e94560; color:#fff; border:none; border-radius:8px; padding:10px 16px; font-weight:700; cursor:pointer; }
  .btn:disabled { opacity:.6; cursor:not-allowed; }
  .hint { color:#a0a0b0; font-size:12px; margin-top:6px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; margin-top:12px; }
  .item { border:1px solid #0f3460; border-radius:8px; overflow:hidden; background:#0d1b3e; }
  .item img { width:100%; display:block; aspect-ratio:16/9; object-fit:cover; }
  .item .meta { padding:8px; color:#a0a0b0; font-size:12px; display:flex; justify-content:space-between; align-items:center; gap:8px; }
  .status { color:#4ade80; font-size:13px; margin-top:8px; }
  .preview-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:10px; margin-top:10px; }
  .preview { border:1px solid #0f3460; border-radius:8px; overflow:hidden; background:#0d1b3e; }
  .preview img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; }
  .preview .cap { padding:6px 8px; color:#a0a0b0; font-size:11px; word-break:break-word; }
  .btn-sm { font-size:11px; padding:6px 8px; }
  .logs { background:#0d1b3e; border:1px solid #0f3460; border-radius:8px; padding:10px; min-height:130px; max-height:240px; overflow:auto; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }
  .log-line { margin-bottom:4px; color:#b8c0d8; }
</style>
</head>
<body>
  <h1>Thumbnail Revision Page</h1>
  <div class="subtitle">Upload one thumbnail + revision feedback. Runs 10 attempts in parallel, shows logs live, and supports follow-up revisions on any output.</div>
  <div style="margin-bottom:16px;"><a href="/" style="color:#4ade80;text-decoration:none;font-weight:600;">← Back to Main Generator</a></div>

  <div class="card">
    <label>Thumbnail to revise (required for first run)</label>
    <input type="file" id="baseThumb" accept="image/*">
    <div class="hint">Tip: after first run, click "Use as Base" under any result to do fast follow-up revisions.</div>
    <div class="hint">Paste support: click this section, then paste (⌘V / Ctrl+V) an image copied from Chrome.</div>

    <div id="basePreviewWrap" class="preview-grid"></div>

    <label>Revision feedback (required)</label>
    <textarea id="feedback" placeholder="e.g. Make title bigger, darken background, add warning icon, keep face likeness."></textarea>

    <label>Extra reference images (optional)</label>
    <input type="file" id="extraFiles" accept="image/*" multiple>
    <div class="hint">Upload additional refs to incorporate. Previews shown below.</div>
    <div class="hint">Paste support: click this section, then paste (⌘V / Ctrl+V) to append pasted images.</div>
    <div id="extraPreviews" class="preview-grid"></div>

    <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;">
      <button id="runBtn" class="btn" onclick="runRevision()">Generate 10 Revision Attempts</button>
      <button id="clearFollowupBtn" class="btn" style="background:#0f3460;" onclick="clearFollowUpBase()">Clear Follow-up Base</button>
    </div>
    <div id="statusText" class="status"></div>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Live Logs</h3>
    <div id="logBox" class="logs"></div>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Results</h3>
    <div id="resultsGrid" class="grid"></div>
  </div>

<script>
let pollInterval = null;
let seen = new Set();
let activeOutputDir = '';
let followUpBasePath = '';
let pasteTarget = 'base';

function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

function mergeFiles(input, incomingFiles, replaceExisting) {
  const dt = new DataTransfer();
  if (!replaceExisting) {
    for (const f of (input.files || [])) dt.items.add(f);
  }
  for (const f of incomingFiles) dt.items.add(f);
  input.files = dt.files;
}

function getPastedImageFiles(e) {
  const out = [];
  const items = (e.clipboardData && e.clipboardData.items) ? e.clipboardData.items : [];
  for (const item of items) {
    if (item.type && item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) out.push(file);
    }
  }
  return out;
}

function renderUploadPreview(containerId, files, labelPrefix) {
  const wrap = document.getElementById(containerId);
  wrap.innerHTML = '';
  Array.from(files || []).forEach((f, idx) => {
    const url = URL.createObjectURL(f);
    const card = document.createElement('div');
    card.className = 'preview';
    card.innerHTML = '<img src="' + url + '"><div class="cap">' + esc(labelPrefix) + ' #' + (idx + 1) + ' — ' + esc(f.name) + '</div>';
    wrap.appendChild(card);
  });
}

function renderBasePathPreview(path) {
  const wrap = document.getElementById('basePreviewWrap');
  wrap.innerHTML = '';
  if (!path) return;
  const card = document.createElement('div');
  card.className = 'preview';
  card.innerHTML = '<img src="/image?path=' + encodeURIComponent(path) + '"><div class="cap">Follow-up base from prior result</div>';
  wrap.appendChild(card);
}

document.getElementById('baseThumb').addEventListener('change', (e) => {
  followUpBasePath = '';
  pasteTarget = 'base';
  renderUploadPreview('basePreviewWrap', e.target.files, 'Base image');
});

document.getElementById('extraFiles').addEventListener('change', (e) => {
  pasteTarget = 'extra';
  renderUploadPreview('extraPreviews', e.target.files, 'Reference');
});

document.getElementById('basePreviewWrap').addEventListener('click', () => { pasteTarget = 'base'; });
document.getElementById('extraPreviews').addEventListener('click', () => { pasteTarget = 'extra'; });
document.getElementById('baseThumb').addEventListener('click', () => { pasteTarget = 'base'; });
document.getElementById('extraFiles').addEventListener('click', () => { pasteTarget = 'extra'; });

document.addEventListener('paste', (e) => {
  const pasted = getPastedImageFiles(e);
  if (!pasted.length) return;
  e.preventDefault();

  if (pasteTarget === 'extra') {
    const input = document.getElementById('extraFiles');
    mergeFiles(input, pasted, false);
    renderUploadPreview('extraPreviews', input.files, 'Reference');
    document.getElementById('statusText').textContent = 'Pasted ' + pasted.length + ' reference image(s).';
    return;
  }

  // Default: base image slot (replace existing base)
  const baseInput = document.getElementById('baseThumb');
  followUpBasePath = '';
  mergeFiles(baseInput, [pasted[0]], true);
  renderUploadPreview('basePreviewWrap', baseInput.files, 'Base image');
  document.getElementById('statusText').textContent = 'Pasted base image from clipboard.';
});

function clearFollowUpBase() {
  followUpBasePath = '';
  document.getElementById('baseThumb').value = '';
  document.getElementById('basePreviewWrap').innerHTML = '';
}

async function runRevision() {
  const fileInput = document.getElementById('baseThumb');
  const feedback = document.getElementById('feedback').value.trim();
  if (!fileInput.files.length && !followUpBasePath) { alert('Upload a thumbnail (or use a prior output as base).'); return; }
  if (!feedback) { alert('Enter revision feedback.'); return; }

  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.textContent = 'Starting...';
  document.getElementById('statusText').textContent = 'Submitting revision request...';

  const fd = new FormData();
  if (fileInput.files.length) fd.append('base_thumbnail', fileInput.files[0]);
  if (followUpBasePath) fd.append('base_path', followUpBasePath);
  fd.append('prompt', feedback);
  const extras = document.getElementById('extraFiles');
  for (const f of extras.files) fd.append('revision_images', f);

  try {
    const resp = await fetch('/revise_upload', { method:'POST', body: fd });
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }
    activeOutputDir = data.output_dir || '';
    seen = new Set();
    document.getElementById('resultsGrid').innerHTML = '';
    document.getElementById('statusText').textContent = 'Running 10 parallel attempts...';
    startPolling();
  } catch (e) {
    alert('Error: ' + e);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate 10 Revision Attempts';
  }
}

function addImageCard(img) {
  if (activeOutputDir && (!img.path || !img.path.startsWith(activeOutputDir))) return;
  if (seen.has(img.idx)) return;
  seen.add(img.idx);
  const grid = document.getElementById('resultsGrid');
  const div = document.createElement('div');
  div.className = 'item';
  div.innerHTML = '<img src="/image?path=' + encodeURIComponent(img.path) + '">' +
    '<div class="meta"><span>Attempt #' + img.idx + '</span><button class="btn btn-sm" onclick="setFollowUpBase(\'' + encodeURIComponent(img.path) + '\')">Use as Base</button></div>';
  grid.appendChild(div);
}

function setFollowUpBase(encodedPath) {
  followUpBasePath = decodeURIComponent(encodedPath);
  document.getElementById('baseThumb').value = '';
  renderBasePathPreview(followUpBasePath);
  document.getElementById('statusText').textContent = 'Follow-up base selected. Enter feedback and run again.';
}

function renderLogs(logs) {
  const box = document.getElementById('logBox');
  box.innerHTML = (logs || []).slice(-80).map(l => '<div class="log-line">' + esc(l) + '</div>').join('');
  box.scrollTop = box.scrollHeight;
}

function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const r = await fetch('/status');
      const d = await r.json();
      (d.images || []).forEach(addImageCard);
      renderLogs(d.log || []);
      document.getElementById('statusText').textContent = d.running
        ? `Generating ${d.completed}/${d.total}... (${d.errors || 0} errors)`
        : `Done: ${d.completed} generated (${d.errors || 0} errors)`;
      if (d.done && !d.running) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    } catch (_) {}
  }, 1000);
}

startPolling();
</script>
</body>
</html>"""

# ----- HTTP Server -----


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _check_auth(self):
        if not APP_PASS:
            return True
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(auth[6:]).decode()
            return decoded == f"{APP_USER}:{APP_PASS}"
        except Exception:
            return False

    def _send_auth_required(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Doom Debates Thumbnail Generator"')
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Authentication required")

    def do_GET(self):
        if not self._check_auth():
            self._send_auth_required()
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = dict(urllib.parse.parse_qsl(parsed.query))

        try:
            self._route_get(path, params)
        except Exception as e:
            print(f"GET {path} error: {e}")
            try:
                self._json_response({"error": f"Server error: {str(e)[:300]}"})
            except Exception:
                pass

    def _route_get(self, path, params):
        if path == "/":
            self._serve_html()
        elif path == "/revision":
            self._serve_revision_html()
        elif path == "/status":
            # Thread-safe snapshot: copy mutable containers to prevent
            # RuntimeError from concurrent dict/list modification by
            # the background generation thread.
            try:
                with status_lock:
                    safe = {
                        "running": status["running"],
                        "phase": status["phase"],
                        "total": status["total"],
                        "completed": status["completed"],
                        "errors": status["errors"],
                        "log": list(status["log"]),
                        "images": list(status["images"]),
                        "done": status["done"],
                        "output_dir": status["output_dir"],
                        "episode_dir": status["episode_dir"],
                        "round_num": status["round_num"],
                        "ideas": list(status.get("ideas", [])),
                        "idea_groups": {k: list(v) for k, v in status["idea_groups"].items()},
                        "cost": status["cost"],
                        "session_cost": status.get("session_cost", 0.0),
                    }
            except Exception:
                # Fallback: return minimal status so polling doesn't break
                safe = {
                    "running": status.get("running", False),
                    "done": status.get("done", False),
                    "completed": status.get("completed", 0),
                    "total": status.get("total", 0),
                    "errors": status.get("errors", 0),
                    "log": [], "images": [], "idea_groups": {},
                    "phase": status.get("phase", "idle"),
                    "output_dir": "", "episode_dir": "",
                    "round_num": 0, "ideas": [],
                    "cost": 0, "session_cost": 0,
                }
            self._json_response(safe)
        elif path == "/image":
            self._serve_image(params.get("path", ""))
        elif path == "/download":
            self._serve_download(params.get("path", ""))
        elif path == "/vary":
            self._handle_vary(params)
        elif path == "/save_finals":
            self._handle_save_finals(params)
        elif path == "/open_folder":
            self._handle_open_folder()
        else:
            self.send_error(404)

    def do_POST(self):
        if not self._check_auth():
            self._send_auth_required()
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            if path == "/gather_images":
                self._handle_gather_images(body)
            elif path == "/revise_upload":
                self._handle_revise_upload(body)
            elif path == "/generate_ideas":
                self._handle_generate_ideas(body)
            elif path == "/generate_from_ideas":
                self._handle_generate_from_ideas(body)
            elif path == "/riff_idea":
                self._handle_riff_idea(body)
            elif path == "/more_ideas":
                self._handle_more_ideas(body)
            elif path == "/revise":
                self._handle_revise_post(body)
            else:
                self.send_error(404)
        except Exception as e:
            print(f"POST {path} error: {e}")
            try:
                self._json_response({"error": f"Server error: {str(e)[:300]}"})
            except Exception:
                pass

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

    def _serve_revision_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_REVISION.encode("utf-8"))

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _serve_image(self, filepath):
        if not filepath or not os.path.isfile(filepath):
            self.send_error(404)
            return
        real = os.path.realpath(filepath)
        if not real.startswith(os.path.realpath(THUMBNAILS_DIR)):
            self.send_error(403)
            return
        ext = os.path.splitext(filepath)[1].lower()
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp",
        }.get(ext, "image/png")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(os.path.getsize(filepath)))
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def _serve_download(self, filepath):
        if not filepath or not os.path.isfile(filepath):
            self.send_error(404)
            return
        real = os.path.realpath(filepath)
        if not real.startswith(os.path.realpath(THUMBNAILS_DIR)):
            self.send_error(403)
            return
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp",
        }.get(ext, "image/png")
        filesize = os.path.getsize(filepath)
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(filesize))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    # ----- New Endpoints -----

    def _handle_gather_images(self, body):
        """Generate search queries via Gemini, then search Brave for images."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            fields, _ = parse_multipart(self.headers, body)
        else:
            fields = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="replace")))

        title = fields.get("title", "").strip()
        custom_prompt = fields.get("custom_prompt", "").strip()

        if not title:
            self._json_response({"error": "Episode title is required"})
            return

        if not BRAVE_API_KEY:
            self._json_response({"error": "BRAVE_API_KEY not configured. Upload source images manually."})
            return

        try:
            client = get_client()
            queries = generate_search_queries(client, title, custom_prompt)
            images = search_images_brave(queries)
            self._json_response({"ok": True, "queries": queries, "images": images})
        except Exception as e:
            self._json_response({"error": f"Search failed: {str(e)[:200]}"})

    def _handle_generate_ideas(self, body):
        """Generate 10 thumbnail ideas via Gemini text model."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            fields, _ = parse_multipart(self.headers, body)
        else:
            fields = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="replace")))

        title = fields.get("title", "").strip()
        custom_prompt = fields.get("custom_prompt", "").strip()
        transcript_text = fields.get("transcript", "").strip()
        additional = fields.get("additional_instructions", "").strip()

        if not title:
            self._json_response({"error": "Episode title is required"})
            return

        try:
            client = get_client()
            ideas_list = generate_ideas(client, title, custom_prompt, transcript_text, additional)
            status["ideas"] = ideas_list
            self._json_response({"ok": True, "ideas": ideas_list})
        except Exception as e:
            self._json_response({"error": f"Idea generation failed: {str(e)[:200]}"})

    def _handle_generate_from_ideas(self, body):
        """Generate 3 thumbnails per idea."""
        global status
        with status_lock:
            is_running = status["running"]
        if is_running:
            self._json_response({"error": "Generation already in progress"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            fields, files = parse_multipart(self.headers, body)
        else:
            fields = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="replace")))
            files = {}

        title = fields.get("title", "").strip()
        ideas_json = fields.get("ideas", "[]")
        custom_prompt = fields.get("custom_prompt", "").strip()
        additional = fields.get("additional_instructions", "").strip()

        try:
            ideas_list = json.loads(ideas_json)
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid ideas JSON"})
            return

        if not ideas_list:
            self._json_response({"error": "No ideas provided"})
            return

        # Upload speaker and source images to Gemini File API
        client = get_client()
        speaker_refs = upload_files_from_bytes(client, files.get("speakers", []), "speaker")
        source_refs = upload_files_from_bytes(client, files.get("sources", []), "source")

        # Download and upload web-gathered source images
        source_urls_json = fields.get("source_urls", "[]")
        try:
            source_urls = json.loads(source_urls_json)
        except json.JSONDecodeError:
            source_urls = []

        for url in source_urls[:15]:  # Cap at 15 web images
            img_bytes = download_image_bytes(url)
            if img_bytes:
                refs = upload_files_from_bytes(client, [img_bytes], "web_source")
                source_refs.extend(refs)

        # Build output dir
        slug = re.sub(r"[^a-z0-9]+", "-", title[:40].lower()).strip("-") or "episode"
        date = datetime.date.today().isoformat()
        episode_dir = os.path.join(THUMBNAILS_DIR, f"{slug}-{date}")
        round_dir = os.path.join(episode_dir, "round1")
        os.makedirs(round_dir, exist_ok=True)

        # Save metadata
        info_dict = {
            "title": title,
            "custom_prompt": custom_prompt,
            "num_ideas": len(ideas_list),
            "num_speaker_photos": len(speaker_refs),
            "num_source_images": len(source_refs),
        }
        with open(os.path.join(episode_dir, "episode.json"), "w") as f:
            json.dump(info_dict, f, indent=2)

        # Build prompts
        prompts = build_idea_prompts(ideas_list, speaker_refs, source_refs, custom_prompt, additional, variations_per=3)
        save_metadata(round_dir, info_dict, len(prompts), "round1")

        # Store state
        status["episode_dir"] = episode_dir
        status["speakers"] = speaker_refs
        status["sources"] = source_refs
        status["ideas"] = ideas_list
        status["round_num"] = 1
        run_generation(client, prompts, round_dir, "round1")

        self._json_response({"ok": True, "output_dir": round_dir, "count": len(prompts)})

    def _handle_riff_idea(self, body):
        """Generate more thumbnails for a single idea."""
        global status
        with status_lock:
            is_running = status["running"]
        if is_running:
            self._json_response({"error": "Generation already in progress"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            fields, files = parse_multipart(self.headers, body)
        else:
            fields = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="replace")))
            files = {}

        idea_text = fields.get("idea_text", "").strip()
        idea_idx = int(fields.get("idea_idx", "-1"))
        custom_prompt = fields.get("custom_prompt", "").strip()
        additional = fields.get("additional_instructions", "").strip()
        riff_prompt = fields.get("riff_prompt", "").strip()

        if not idea_text:
            self._json_response({"error": "No idea text provided"})
            return

        # Append riff prompt to additional instructions so it's included in the generation
        if riff_prompt:
            additional = (additional + "\n\nRIFF INSTRUCTIONS: " + riff_prompt) if additional else "RIFF INSTRUCTIONS: " + riff_prompt

        client = get_client()

        # Reuse stored refs from initial generation (riff no longer resends full library)
        speaker_refs = status.get("speakers", [])
        source_refs = list(status.get("sources", []))

        # Upload riff-specific images if the user added any
        riff_image_refs = upload_files_from_bytes(client, files.get("riff_images", []), "riff_img")
        source_refs.extend(riff_image_refs)

        episode_dir = status.get("episode_dir", "")
        if not episode_dir:
            episode_dir = os.path.join(THUMBNAILS_DIR, f"riff-{datetime.date.today().isoformat()}")

        with status_lock:
            status["round_num"] = status.get("round_num", 1) + 1
            round_num = status["round_num"]

            # Compute next available idea index (beyond all existing images and ideas)
            existing_max = max((img["idea_idx"] for img in status["images"]), default=-1)
            ideas_max = len(status.get("ideas", [])) - 1
            riff_idea_idx = max(existing_max, ideas_max) + 1

            # Track this riff idea server-side so future riff_idea_idx
            # computations stay correct even without image data.
            riff_label = f"(Riff on Idea {idea_idx + 1}) {idea_text}"
            ideas_list = status.get("ideas", [])
            while len(ideas_list) <= riff_idea_idx:
                ideas_list.append("")
            ideas_list[riff_idea_idx] = riff_label
            status["ideas"] = ideas_list

        round_dir = os.path.join(episode_dir, f"round{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        # Build riff prompts — assign a NEW idea_idx so riffs appear as a new group
        riff_count = int(fields.get("riff_count", "3"))
        if riff_count < 1:
            riff_count = 3
        if riff_count > 50:
            riff_count = 50
        prompts_raw = build_idea_prompts([idea_text], speaker_refs, source_refs, custom_prompt, additional, variations_per=riff_count)

        prompts = [(riff_idea_idx, var, contents) for (_, var, contents) in prompts_raw]

        run_generation(client, prompts, round_dir, "riff")
        self._json_response({
            "ok": True, "output_dir": round_dir, "count": len(prompts),
            "riff_idea_idx": riff_idea_idx, "riff_label": idea_text,
        })

    def _handle_more_ideas(self, body):
        """Generate more ideas, avoiding duplicates."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            fields, _ = parse_multipart(self.headers, body)
        else:
            fields = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="replace")))

        title = fields.get("title", "").strip()
        custom_prompt = fields.get("custom_prompt", "").strip()
        transcript_text = fields.get("transcript", "").strip()
        additional = fields.get("additional_instructions", "").strip()
        existing_json = fields.get("existing_ideas", "[]")

        if not title:
            self._json_response({"error": "Episode title is required"})
            return

        try:
            existing = json.loads(existing_json)
        except json.JSONDecodeError:
            existing = []

        # Add existing ideas to the prompt so it avoids duplicates (truncate to avoid prompt bloat)
        extra_instruction = ""
        if existing:
            # Only include first 80 chars of each idea, max 20 ideas
            summaries = [e[:80] for e in existing[:20]]
            extra_instruction = "\n\nAVOID duplicating these existing ideas:\n" + "\n".join(f"- {s}" for s in summaries)

        try:
            client = get_client()
            combined_additional = (additional + extra_instruction) if additional else extra_instruction
            new_ideas = generate_ideas(client, title, custom_prompt, transcript_text, combined_additional)
            self._json_response({"ok": True, "ideas": new_ideas})
        except Exception as e:
            self._json_response({"error": f"Idea generation failed: {str(e)[:200]}"})

    def _handle_revise_upload(self, body):
        global status
        with status_lock:
            is_running = status["running"]
        if is_running:
            self._json_response({"error": "Generation already in progress"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            fields, files = parse_multipart(self.headers, body)
        else:
            fields = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="replace")))
            files = {}

        prompt = fields.get("prompt", "").strip()
        base_files = files.get("base_thumbnail", [])
        base_path = fields.get("base_path", "").strip()

        if not prompt:
            self._json_response({"error": "Revision prompt is required"})
            return
        if not base_files and not base_path:
            self._json_response({"error": "Upload one base thumbnail image (or choose a prior output as follow-up base)"})
            return

        base_img = None
        if base_files:
            try:
                base_img = Image.open(io.BytesIO(base_files[0])).convert("RGB")
            except Exception:
                self._json_response({"error": "Could not parse uploaded thumbnail image"})
                return
        else:
            try:
                real = os.path.realpath(base_path)
                if not real.startswith(os.path.realpath(THUMBNAILS_DIR)) or not os.path.isfile(real):
                    self._json_response({"error": "Invalid follow-up base image path"})
                    return
                base_img = Image.open(real).convert("RGB")
            except Exception:
                self._json_response({"error": "Could not load follow-up base image"})
                return

        client = get_client()
        attachment_refs = upload_files_from_bytes(client, files.get("revision_images", []), "revision_img")

        episode_dir = os.path.join(THUMBNAILS_DIR, f"revision-page-{datetime.date.today().isoformat()}")
        with status_lock:
            status["round_num"] = status.get("round_num", 0) + 1
            round_num = status["round_num"]
            revision_idea_idx = 0
            status["ideas"] = [f"Revision page: {prompt[:120]}"]
            status["episode_dir"] = episode_dir

        round_dir = os.path.join(episode_dir, f"round{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        prompts = build_revision_prompts(
            [base_img],
            status.get("speakers", []),
            prompt,
            count_per=10,
            idea_idx=revision_idea_idx,
            attachment_refs=attachment_refs if attachment_refs else None,
        )

        run_generation(client, prompts, round_dir, "revision_page")
        with status_lock:
            base_src = "uploaded file" if base_files else "follow-up result"
            status["log"].append(f"Revision base: {base_src}")
            status["log"].append(f"Prompt: {prompt[:180]}")
            status["log"].append(f"Extra refs attached: {len(files.get('revision_images', []))}")

        self._json_response({"ok": True, "output_dir": round_dir, "count": len(prompts)})

    def _handle_revise_post(self, body):
        global status
        with status_lock:
            is_running = status["running"]
        if is_running:
            self._json_response({"error": "Generation already in progress"})
            return

        # Parse multipart/form-data (for attachments) or url-encoded
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            fields, files = parse_multipart(self.headers, body)
        else:
            fields = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="replace")))
            files = {}

        indices_raw = fields.get("indices", "")
        custom_prompt = fields.get("prompt", "").strip()
        source_idea_indices_json = fields.get("source_idea_indices", "[]")

        indices = [int(x) for x in indices_raw.split(",") if x.strip().isdigit()]
        if not indices:
            self._json_response({"error": "No images selected"})
            return
        if not custom_prompt:
            self._json_response({"error": "Revision prompt is required"})
            return

        selected_images = []
        # Track which idea(s) the selected images belong to
        source_idea_idxs = set()
        for img_info in status["images"]:
            if img_info["idx"] in indices and os.path.isfile(img_info["path"]):
                selected_images.append(Image.open(img_info["path"]))
                if img_info.get("idea_idx", -1) >= 0:
                    source_idea_idxs.add(img_info["idea_idx"])

        if not selected_images:
            if not status["images"]:
                self._json_response({"error": "No thumbnails in server memory. The server was likely restarted — please regenerate thumbnails first."})
            else:
                self._json_response({"error": "Could not load selected images (indices " + indices_raw + " not found among " + str(len(status['images'])) + " known images). Try regenerating."})
            return

        client = get_client()

        # Upload attachment images if provided
        attachment_refs = upload_files_from_bytes(client, files.get("revision_images", []), "revision_img")

        speakers = status.get("speakers", [])
        episode_dir = status.get("episode_dir", "")

        # Build revision label from source ideas
        try:
            src_indices = json.loads(source_idea_indices_json)
        except (json.JSONDecodeError, ValueError):
            src_indices = list(source_idea_idxs)

        ideas_list = status.get("ideas", [])
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

        with status_lock:
            status["round_num"] = status.get("round_num", 1) + 1
            round_num = status["round_num"]

            # Compute next available idea index (like riff does)
            existing_max = max((img["idea_idx"] for img in status["images"]), default=-1)
            ideas_max = len(status.get("ideas", [])) - 1
            revision_idea_idx = max(existing_max, ideas_max) + 1

            # Track this revision idea server-side
            while len(ideas_list) <= revision_idea_idx:
                ideas_list.append("")
            ideas_list[revision_idea_idx] = revision_label
            status["ideas"] = ideas_list

        round_dir = os.path.join(episode_dir, f"round{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        prompts = build_revision_prompts(
            selected_images,
            speakers,
            custom_prompt,
            count_per=3,
            idea_idx=revision_idea_idx,
            attachment_refs=attachment_refs if attachment_refs else None,
        )

        run_generation(client, prompts, round_dir, "revision")

        self._json_response({
            "ok": True,
            "output_dir": round_dir,
            "count": len(prompts),
            "revision_idea_idx": revision_idea_idx,
            "revision_label": revision_label,
        })

    def _handle_vary(self, params):
        global status
        with status_lock:
            is_running = status["running"]
        if is_running:
            self._json_response({"error": "Generation already in progress"})
            return

        indices_raw = params.get("indices", "")
        indices = [int(x) for x in indices_raw.split(",") if x.strip().isdigit()]
        if not indices:
            self._json_response({"error": "No images selected"})
            return

        selected_images = []
        for img_info in status["images"]:
            if img_info["idx"] in indices and os.path.isfile(img_info["path"]):
                selected_images.append(Image.open(img_info["path"]))

        if not selected_images:
            if not status["images"]:
                self._json_response({"error": "No thumbnails in server memory. The server was likely restarted — please regenerate thumbnails first."})
            else:
                self._json_response({"error": "Could not load selected images (indices " + indices_raw + " not found among " + str(len(status['images'])) + " known images). Try regenerating."})
            return

        speakers = status.get("speakers", [])
        episode_dir = status.get("episode_dir", "")
        with status_lock:
            status["round_num"] = status.get("round_num", 1) + 1
            round_num = status["round_num"]
        round_dir = os.path.join(episode_dir, f"round{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        prompts = build_variation_prompts(selected_images, speakers, count_per=3)

        client = get_client()
        run_generation(client, prompts, round_dir, "variation")

        self._json_response({"ok": True, "output_dir": round_dir, "count": len(prompts)})

    def _handle_save_finals(self, params):
        indices_raw = params.get("indices", "")
        indices = [int(x) for x in indices_raw.split(",") if x.strip().isdigit()]
        if not indices:
            self._json_response({"error": "No images selected"})
            return

        episode_dir = status.get("episode_dir", "")
        if not episode_dir:
            self._json_response({"error": "No episode directory found"})
            return

        finals_dir = os.path.join(episode_dir, "finals")
        os.makedirs(finals_dir, exist_ok=True)

        count = 0
        for img_info in status["images"]:
            if img_info["idx"] in indices and os.path.isfile(img_info["path"]):
                count += 1
                dest = os.path.join(finals_dir, f"final_{count}.png")
                shutil.copy2(img_info["path"], dest)

        self._json_response({"ok": True, "count": count, "finals_dir": finals_dir})

    def _handle_open_folder(self):
        if os.environ.get("NO_BROWSER") == "1":
            self._json_response({"ok": True, "note": "Folder open disabled on server"})
            return
        episode_dir = status.get("episode_dir", THUMBNAILS_DIR)
        if os.path.isdir(episode_dir):
            subprocess.Popen(["open", episode_dir])
        self._json_response({"ok": True})


# ----- Main -----


def main():
    os.makedirs(THUMBNAILS_DIR, exist_ok=True)

    client = get_client()
    upload_brand_references(client)
    upload_liron_references(client)
    print(f"Doom Debates Thumbnail Generator v2")
    print(f"Image Model: {GEMINI_MODEL}")
    print(f"Text Model: {TEXT_MODEL}")
    print(f"Output: {THUMBNAILS_DIR}")
    print(f"Brand Refs: {len(BRAND_FILES)} images from {EXAMPLES_DIR}")
    print(f"Liron Refs: {len(LIRON_FILES)} images from {LIRON_DIR}")
    print(f"Brave Search: {'enabled' if BRAVE_API_KEY else 'disabled (no BRAVE_API_KEY)'}")
    print(f"Server: http://0.0.0.0:{PORT}")
    if APP_PASS:
        print(f"Auth: enabled (user={APP_USER})")
    print()

    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    if os.environ.get("NO_BROWSER") != "1":
        webbrowser.open(f"http://127.0.0.1:{PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
