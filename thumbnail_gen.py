#!/usr/bin/env python3
"""
Doom Debates Thumbnail Generator — Nano Banana Pro

Bulk-generates YouTube thumbnail candidates via Google Gemini image generation,
with a browser UI for selection, revision, and iteration.

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
MAX_CONCURRENT = 15
DEFAULT_COUNT = 100
THUMBNAILS_DIR = os.path.join(SCRIPT_DIR, "thumbnails")
EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "doom_debates_thumbnails")

DEFAULT_STYLE = """VISUAL STYLE: Choose colors, lighting, and mood that match the energy of the transcript.
If the conversation is tense or confrontational — use bold contrasts, dramatic lighting, split compositions.
If it's hopeful or exploratory — use warmer tones, open compositions, curious expressions.
If it's technical or analytical — use clean modern aesthetics, sharp lines, cool tones.
Let the content dictate the look. Optimize for maximum clicks and watch time."""

BRAND_GUIDE = """BRAND — "DOOM DEBATES":
YouTube channel by Liron Shapira about AI existential risk.
Study the attached brand reference thumbnails for VISUAL STYLE ONLY — colors, composition, typography, energy.
Do NOT copy or reproduce any faces or people from the brand references. The only people in your thumbnail
should come from the speaker photos (if provided).
Key brand traits:
- Red-toned background gradient or texture as the base
- Large bold headline text in white or yellow, dramatic and punchy
- Guest/host photos with intense, exaggerated expressions (shock, concern, confrontation)
- Composite imagery — people combined with dramatic AI/apocalyptic visuals
- Overall feel: high-stakes, provocative, debate energy
- Color palette: deep reds, blacks, whites, with yellow accents for emphasis"""

MEGA_PROMPT = """Generate a YouTube thumbnail image.

MOST IMPORTANT RULE — TEXT: The image must contain ONLY ONE text element. This single text element must be 1-5 words. No subtitles, no secondary text, no labels, no captions, no watermarks. Just ONE short punchy headline. Examples: "WHO WINS?", "AI TAKEOVER", "THE END?", "P(DOOM)". Nothing longer. If in doubt, use FEWER words.

STEP 1 — CONCEPT FROM TRANSCRIPT:
Read the episode transcript below and decide what the thumbnail should depict. What's the most compelling, clickable visual moment?

EPISODE TRANSCRIPT:
{transcript_excerpt}

{visual_concept_section}

STEP 2 — APPLY BRAND STYLE:
Now apply the Doom Debates brand style to your concept. Use the attached brand reference thumbnails for visual style ONLY (colors, composition, typography, energy). Do NOT copy any faces or people from the brand references.

{brand_guide}

{speaker_section}

{inspiration_section}

RULES:
- 16:9 aspect ratio, photorealistic, sharp focus
- Large expressive faces (40-60% of frame)
- High contrast, clean composition, one focal point
- The ONLY people in the thumbnail should be from the speaker photos (if provided), NOT from brand references
- Remember: ONLY 1-5 words of text total in the entire image. ONE text element only.

Variation #{variation_seed} — make this meaningfully different from other variations."""

REVISION_PROMPT = """Revise this YouTube thumbnail for "Doom Debates" podcast.

REVISION INSTRUCTIONS: {custom_prompt}

Keep the core composition but apply the requested changes.
Maintain 16:9 aspect ratio. ONLY 1-5 words of text in the entire image — one short headline, nothing else.
{speaker_section}

Variation #{variation_seed} — try something meaningfully different from the other revisions."""

VARIATION_PROMPT = """Create a variation of the attached YouTube thumbnail for "Doom Debates" podcast.
Keep the same general composition, mood, and subject, but vary:
- Color treatment and lighting
- Expression intensity
- Background details and atmosphere

The variation should feel like a sibling of the original, not a copy.
Maintain 16:9 aspect ratio. ONLY 1-5 words of text in the entire image — one short headline, nothing else.
{speaker_section}

Variation #{variation_seed} — try something meaningfully different."""

INSPO_PROMPT = """Generate a YouTube thumbnail image.

MOST IMPORTANT RULE — TEXT: The image must contain ONLY ONE text element. This single text element must be 1-5 words. No subtitles, no secondary text, no labels, no captions, no watermarks. Just ONE short punchy headline. Examples: "WHO WINS?", "AI TAKEOVER", "THE END?", "P(DOOM)". Nothing longer. If in doubt, use FEWER words.

STEP 1 — CONCEPT FROM TRANSCRIPT:
Read the episode transcript below and decide what the thumbnail should depict. What's the most compelling, clickable visual moment?

EPISODE TRANSCRIPT:
{transcript_excerpt}

{visual_concept_section}

STEP 2 — APPLY BRAND STYLE:
Now apply the Doom Debates brand style to your concept. Use the attached brand reference thumbnails for visual style ONLY (colors, composition, typography, energy). Do NOT copy any faces or people from the brand references.

{brand_guide}

{speaker_section}

STEP 3 — MATCH INSPIRATION:
An inspiration thumbnail is attached. Heavily match its visual style, composition, color treatment, and energy. Your thumbnail should look like it came from the same designer as the inspiration image. Apply the inspiration's style to the concept you chose in Step 1.

RULES:
- 16:9 aspect ratio, photorealistic, sharp focus
- Large expressive faces (40-60% of frame)
- High contrast, clean composition, one focal point
- The ONLY people in the thumbnail should be from the speaker photos (if provided), NOT from brand references
- Remember: ONLY 1-5 words of text total in the entire image. ONE text element only.

Variation #{variation_seed} — make this meaningfully different from other variations."""

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
    "round_num": 0,
}

# ----- API Client & File API -----

BRAND_FILES = []


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


def upload_files_from_bytes(client, file_bytes_list, name_prefix):
    """Upload raw bytes to Gemini File API, return file reference objects."""
    refs = []
    for i, data in enumerate(file_bytes_list):
        try:
            # Write to temp file for upload
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


# ----- Image Loading -----


def load_images_from_dir(dirpath, max_count=5, randomize=False):
    """Load images from a directory as PIL Images."""
    images = []
    if not os.path.isdir(dirpath):
        return images
    files = sorted(
        f for f in os.listdir(dirpath)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if randomize:
        random.shuffle(files)
    for f in files[:max_count]:
        try:
            img = Image.open(os.path.join(dirpath, f))
            img.load()
            images.append(img)
        except Exception:
            pass
    return images


def load_images_from_bytes(file_bytes_list):
    """Load images from raw bytes (uploaded files)."""
    images = []
    for data in file_bytes_list:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            images.append(img)
        except Exception:
            pass
    return images


# ----- Prompt Building -----


def build_all_prompts(transcript, visual_concept, speaker_refs, inspo_refs, count=100):
    """Build `count` distinct prompt+image content lists for Gemini.

    If inspo images are provided, splits 50/50 between transcript-driven (MEGA_PROMPT)
    and inspo-driven (INSPO_PROMPT) streams.
    """
    transcript_excerpt = transcript[:1500] if transcript else ""

    if visual_concept.strip():
        visual_section = f"VISUAL CONCEPT: {visual_concept}"
    else:
        visual_section = DEFAULT_STYLE

    speaker_section = (
        "SPEAKER LIKENESS (CRITICAL): Photos of the episode's speaker(s) are attached below. "
        "The person(s) in the thumbnail MUST closely resemble these photos — same face, same features, "
        "same skin tone, same hair. Do NOT use generic faces. The speaker photos are the ground truth "
        "for what the person looks like."
        if speaker_refs else ""
    )

    inspiration_section = (
        "Inspiration thumbnails are attached below — draw style inspiration from these."
        if inspo_refs else ""
    )

    # Split count between transcript-driven and inspo-driven streams
    if inspo_refs:
        count_a = count // 2
        count_b = count - count_a
    else:
        count_a = count
        count_b = 0

    prompts = []

    # Stream A: transcript-driven (MEGA_PROMPT)
    for i in range(count_a):
        prompt_text = MEGA_PROMPT.format(
            transcript_excerpt=transcript_excerpt,
            visual_concept_section=visual_section,
            speaker_section=speaker_section,
            inspiration_section=inspiration_section,
            brand_guide=BRAND_GUIDE,
            variation_seed=i + 1,
        )
        contents = [prompt_text]
        if BRAND_FILES:
            brand_sample = random.sample(BRAND_FILES, min(10, len(BRAND_FILES)))
            contents.append("=== DOOM DEBATES BRAND — match ONLY the visual style (colors, composition, typography, energy). Do NOT copy any faces or people from these reference images ===")
            contents.extend(brand_sample)
        if speaker_refs:
            contents.append("=== SPEAKER PHOTOS — the thumbnail MUST use these people's real faces ===")
            contents.extend(speaker_refs)
        if inspo_refs:
            contents.append("=== INSPIRATION — draw style inspiration from these ===")
            contents.extend(inspo_refs)
        prompts.append(contents)

    # Stream B: inspo-driven (INSPO_PROMPT) — one sub-batch PER inspo image
    if inspo_refs and count_b > 0:
        per_inspo = max(1, count_b // len(inspo_refs))
        for inspo_ref in inspo_refs:
            for i in range(per_inspo):
                prompt_text = INSPO_PROMPT.format(
                    transcript_excerpt=transcript_excerpt,
                    visual_concept_section=visual_section,
                    speaker_section=speaker_section,
                    brand_guide=BRAND_GUIDE,
                    variation_seed=i + 1,
                )
                contents = [prompt_text]
                # Single inspo image first (highest priority)
                contents.append("=== INSPIRATION — heavily match this thumbnail's visual style, composition, and energy ===")
                contents.append(inspo_ref)
                if BRAND_FILES:
                    brand_sample = random.sample(BRAND_FILES, min(10, len(BRAND_FILES)))
                    contents.append("=== DOOM DEBATES BRAND — visual style only, do NOT copy faces from these ===")
                    contents.extend(brand_sample)
                if speaker_refs:
                    contents.append("=== SPEAKER PHOTOS — use these people's real faces ===")
                    contents.extend(speaker_refs)
                prompts.append(contents)

    random.shuffle(prompts)  # Mix streams so results aren't grouped
    return prompts


def build_variation_prompts(selected_images, speaker_refs, count_per=15):
    """Build variation prompts from selected images."""
    speaker_section = (
        "SPEAKER LIKENESS (CRITICAL): Speaker photos are attached — the person(s) MUST closely "
        "resemble these photos. Same face, features, skin tone, hair."
        if speaker_refs else ""
    )
    prompts = []
    for img in selected_images:
        for v in range(count_per):
            prompt_text = VARIATION_PROMPT.format(
                speaker_section=speaker_section,
                variation_seed=v + 1,
            )
            contents = [prompt_text, img]
            if BRAND_FILES:
                brand_sample = random.sample(BRAND_FILES, min(10, len(BRAND_FILES)))
                contents.append("=== BRAND REFERENCES — visual style only, do NOT copy faces ===")
                contents.extend(brand_sample)
            if speaker_refs:
                contents.append("=== SPEAKER PHOTOS ===")
                contents.extend(speaker_refs)
            prompts.append(contents)
    return prompts


def build_revision_prompts(selected_images, speaker_refs, custom_prompt, count_per=15):
    """Build revision prompts with custom instructions."""
    speaker_section = (
        "SPEAKER LIKENESS (CRITICAL): Speaker photos are attached — the person(s) MUST closely "
        "resemble these photos. Same face, features, skin tone, hair."
        if speaker_refs else ""
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
            if BRAND_FILES:
                brand_sample = random.sample(BRAND_FILES, min(10, len(BRAND_FILES)))
                contents.append("=== BRAND REFERENCES — visual style only, do NOT copy faces ===")
                contents.extend(brand_sample)
            if speaker_refs:
                contents.append("=== SPEAKER PHOTOS ===")
                contents.extend(speaker_refs)
            prompts.append(contents)
    return prompts


# ----- Async Generation -----


async def generate_batch(client, prompts, output_dir, phase="round1"):
    """Fire parallel Gemini API calls and save results."""
    global status
    os.makedirs(output_dir, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    start_idx = len(status["images"])

    async def generate_one(idx, contents):
        async with sem:
            for attempt in range(3):
                try:
                    response = await client.aio.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=types.ImageConfig(
                                aspect_ratio="16:9",
                            ),
                        ),
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

                                status["completed"] += 1
                                status["images"].append({
                                    "idx": idx,
                                    "path": filepath,
                                    "filename": filename,
                                    "status": "ok",
                                })
                                status["log"].append(
                                    f"[{status['completed']}/{status['total']}] thumb_{idx:03d}.png OK"
                                )
                                return filepath

                    status["errors"] += 1
                    status["log"].append(f"thumb_{idx:03d}: no image in response")
                    return None

                except Exception as e:
                    err = str(e)
                    if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < 2:
                        wait = (2 ** attempt) + random.random()
                        status["log"].append(
                            f"thumb_{idx:03d}: rate limited, retrying in {wait:.1f}s..."
                        )
                        await asyncio.sleep(wait)
                        continue
                    status["errors"] += 1
                    status["log"].append(f"thumb_{idx:03d}: ERROR — {err[:120]}")
                    return None

    tasks = [generate_one(start_idx + i + 1, c) for i, c in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]


def run_generation(client, prompts, output_dir, phase="round1"):
    """Run async generation in a background thread."""
    global status
    status["running"] = True
    status["phase"] = phase
    status["total"] = len(prompts)
    status["completed"] = 0
    status["errors"] = 0
    status["log"] = [f"Starting {phase}: generating {len(prompts)} thumbnails..."]
    if phase == "round1":
        status["images"] = []
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
            status["log"].append(
                f"Done! {len(results)} images generated, {status['errors']} errors."
            )
        except Exception as e:
            status["log"].append(f"FATAL ERROR: {e}")
        finally:
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
    files = {}  # name -> list of bytes

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

        # Strip trailing \r\n-- from content
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
            if filename:  # non-empty filename means actual file
                files.setdefault(name, []).append(content)
        else:
            fields[name] = content.decode("utf-8", errors="replace")

    return fields, files


# ----- HTML UI -----

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Thumbnail Generator — Nano Banana Pro</title>
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
  label.section {
    display: block; font-size: 13px; color: #a0a0b0; margin-bottom: 6px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .field-hint { font-size: 12px; color: #666; margin-top: 4px; margin-bottom: 12px; }
  textarea {
    width: 100%; padding: 10px 14px; border-radius: 8px;
    border: 1px solid #0f3460; background: #0d1b3e; color: #fff;
    font-size: 15px; outline: none; font-family: inherit;
    resize: vertical;
  }
  textarea:focus { border-color: #e94560; }
  .mb { margin-bottom: 16px; }

  .file-upload {
    border: 2px dashed #0f3460; border-radius: 10px; padding: 20px;
    text-align: center; cursor: pointer; transition: all 0.2s;
    position: relative;
  }
  .file-upload:hover { border-color: #e94560; }
  .file-upload.dragover {
    border-color: #e94560; background: rgba(233,69,96,0.08);
  }
  .file-upload input[type="file"] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer;
  }
  .file-upload .upload-label {
    color: #a0a0b0; font-size: 14px;
  }
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
    padding: 12px 28px; border-radius: 8px; border: none;
    font-size: 16px; font-weight: 600; cursor: pointer;
    transition: all 0.15s;
  }
  .btn-primary { background: #e94560; color: #fff; }
  .btn-primary:hover { background: #d63851; }
  .btn-primary:disabled { background: #555; cursor: not-allowed; }
  .btn-secondary { background: #0f3460; color: #fff; }
  .btn-secondary:hover { background: #1a4a80; }
  .btn-secondary:disabled { background: #333; cursor: not-allowed; }

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

  .thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px; padding: 0; margin-top: 16px;
  }
  .thumb-card {
    position: relative; border-radius: 8px; overflow: hidden;
    border: 3px solid transparent; cursor: pointer;
    transition: border-color 0.15s, transform 0.1s;
    background: #0d1b3e;
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
    padding: 6px 10px; font-size: 12px; color: #a0a0b0;
  }

  .actions-bar {
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }
  .selected-count {
    color: #4ade80; font-weight: 600; font-size: 14px;
  }

  .revision-panel {
    margin-top: 16px; padding: 16px; background: #0d1b3e;
    border-radius: 10px; border: 1px solid #0f3460;
    display: none;
  }
  .revision-panel.visible { display: block; }
  .revision-panel textarea {
    margin-bottom: 12px;
  }
  .revision-panel .revision-actions {
    display: flex; gap: 12px;
  }

  #inputSection, #progressSection, #gridSection { display: none; }
  .active { display: block !important; }

  .section-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 16px; flex-wrap: wrap; gap: 12px;
  }
  .section-header h2 { color: #fff; font-size: 20px; }

  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; background: #0f3460; color: #a0a0b0; }
  .tag-required { background: #e94560; color: #fff; }
</style>
</head>
<body>

<h1>Thumbnail Generator</h1>
<div class="subtitle">Model: Nano Banana 2 (gemini-3.1-flash-image-preview) — Bulk-generate thumbnails, pick your favorites</div>

<!-- INPUT FORM -->
<div id="inputSection" class="active">
  <div class="card">
    <div class="mb">
      <label class="section">Episode Transcript <span class="tag tag-required">Required</span></label>
      <div class="file-upload" id="transcriptUpload">
        <input type="file" id="transcriptFile" accept=".txt,.md,.text,.rtf,.doc,.docx,.pdf,.srt,.vtt">
        <div class="upload-label"><strong>Click to browse</strong> or drag & drop a transcript file</div>
        <div id="transcriptFileName" style="color:#4ade80;margin-top:8px;font-size:13px;"></div>
      </div>
      <div class="field-hint">Upload a transcript file (.txt, .md, .srt, etc). The more context, the better the thumbnails.</div>
    </div>

    <div class="mb">
      <label class="section">Visual Concept <span class="tag">Optional</span></label>
      <textarea id="visualConcept" rows="3" placeholder="e.g. Two faces in dramatic confrontation, split red/blue background, text says 'P(DOOM)?'&#10;&#10;Leave blank for automatic style (YouTube best practices + Doom Debates branding)"></textarea>
      <div class="field-hint">Describe what you want to see. If left blank, uses a default dramatic podcast style.</div>
    </div>
  </div>

  <div class="card">
    <div class="mb">
      <label class="section">Speaker Photos <span class="tag">Optional</span></label>
      <div class="file-upload" id="speakerUpload">
        <input type="file" id="speakerFiles" multiple accept="image/*">
        <div class="upload-label"><strong>Click to browse</strong> or drag & drop speaker photos</div>
        <div class="file-previews" id="speakerPreviews"></div>
      </div>
      <div class="field-hint">Upload photos of the guest/host so generated faces match their likeness.</div>
    </div>

    <div class="mb">
      <label class="section">Inspiration Thumbnails <span class="tag">Optional</span></label>
      <div class="file-upload" id="inspoUpload">
        <input type="file" id="inspoFiles" multiple accept="image/*">
        <div class="upload-label"><strong>Click to browse</strong> or drag & drop example thumbnails</div>
        <div class="file-previews" id="inspoPreviews"></div>
      </div>
      <div class="field-hint">Upload thumbnails you like as style references. If none provided, uses YouTube thumbnail best practices.</div>
    </div>
  </div>

  <div style="display:flex;gap:12px;align-items:center;">
    <label class="section" style="margin:0;">Count</label>
    <input type="number" id="countInput" min="5" max="100" value="20" step="5"
           style="width:80px; padding:8px; border-radius:8px; border:1px solid #0f3460;
           background:#0d1b3e; color:#fff; font-size:15px;">
    <button class="btn btn-primary" id="generateBtn" onclick="startGeneration()">Generate Thumbnails</button>
  </div>
</div>

<!-- PROGRESS -->
<div id="progressSection">
  <div class="card">
    <div class="section-header">
      <h2 id="phaseLabel">Generating...</h2>
      <span id="progressText" style="color:#a0a0b0;">0 / 100</span>
    </div>
    <div class="progress-bar">
      <div class="progress-fill" id="progressFill" style="width:0%"></div>
    </div>
    <div class="log" id="logArea"></div>
  </div>
</div>

<!-- GRID + SELECTION -->
<div id="gridSection">
  <div class="card">
    <div class="section-header">
      <h2 id="gridLabel">Select Your Favorites</h2>
      <div class="actions-bar">
        <span class="selected-count"><span id="selectedCount">0</span> selected</span>
        <button class="btn btn-primary" id="saveBtn" onclick="saveFinals()" disabled>Save Finals</button>
        <button class="btn btn-secondary" onclick="openFolder()">Open Folder</button>
      </div>
    </div>

    <div class="revision-panel" id="revisionPanel">
      <label class="section">Revise Selected</label>
      <textarea id="revisionPrompt" rows="2" placeholder="e.g. Make the background darker, add more fire, change text to EXTINCTION, make the faces larger..."></textarea>
      <div class="revision-actions">
        <button class="btn btn-primary" id="reviseBtn" onclick="startRevision()">Revise with Prompt</button>
        <button class="btn btn-secondary" id="similarBtn" onclick="startVariations()">Generate Similar</button>
      </div>
    </div>

    <div class="thumb-grid" id="thumbGrid"></div>
  </div>
</div>

<script>
let selected = new Set();
let allImages = [];
let pollInterval = null;

// File upload previews
function setupFileUpload(uploadId, inputId, previewId) {
  const zone = document.getElementById(uploadId);
  const input = document.getElementById(inputId);
  const previews = document.getElementById(previewId);

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      // Merge with existing files
      const dt = new DataTransfer();
      for (const f of input.files) dt.items.add(f);
      for (const f of e.dataTransfer.files) dt.items.add(f);
      input.files = dt.files;
      showPreviews(input, previews);
    }
  });

  input.addEventListener('change', () => showPreviews(input, previews));
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
setupFileUpload('inspoUpload', 'inspoFiles', 'inspoPreviews');

// Transcript file upload with drag-and-drop
(function() {
  const zone = document.getElementById('transcriptUpload');
  const input = document.getElementById('transcriptFile');
  const nameEl = document.getElementById('transcriptFileName');

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      input.files = e.dataTransfer.files;
      nameEl.textContent = input.files[0].name;
    }
  });
  input.addEventListener('change', () => {
    nameEl.textContent = input.files.length ? input.files[0].name : '';
  });
})();

async function startGeneration() {
  const fileInput = document.getElementById('transcriptFile');
  if (!fileInput.files.length) {
    alert('Please upload a transcript file.');
    return;
  }

  const transcript = await fileInput.files[0].text();
  if (!transcript.trim()) {
    alert('Transcript file is empty.');
    return;
  }

  const fd = new FormData();
  fd.append('transcript', transcript);
  fd.append('visual_concept', document.getElementById('visualConcept').value);
  fd.append('count', document.getElementById('countInput').value);

  for (const f of document.getElementById('speakerFiles').files) {
    fd.append('speakers', f);
  }
  for (const f of document.getElementById('inspoFiles').files) {
    fd.append('inspiration', f);
  }

  document.getElementById('generateBtn').disabled = true;

  fetch('/generate', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      if (data.error) { alert(data.error); document.getElementById('generateBtn').disabled = false; return; }
      document.getElementById('inputSection').classList.remove('active');
      document.getElementById('progressSection').classList.add('active');
      document.getElementById('gridSection').classList.add('active');
      selected.clear();
      allImages = [];
      document.getElementById('thumbGrid').innerHTML = '';
      updateSelectedUI();
      startPolling();
    })
    .catch(e => { alert('Error: ' + e); document.getElementById('generateBtn').disabled = false; });
}

function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(pollStatus, 500);
}

function pollStatus() {
  fetch('/status').then(r => r.json()).then(data => {
    const doneCount = data.completed + data.errors;
    const pct = data.total > 0 ? (doneCount / data.total * 100) : 0;
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('progressText').textContent = data.completed + ' / ' + data.total + (data.errors > 0 ? ' (' + data.errors + ' failed)' : '');

    const phaseNames = { round1: 'Generating Round 1...', revision: 'Revising...', variation: 'Generating Variations...' };
    document.getElementById('phaseLabel').textContent = phaseNames[data.phase] || 'Generating...';

    const logArea = document.getElementById('logArea');
    logArea.innerHTML = data.log.slice(-30).map(l => '<div>' + escHtml(l) + '</div>').join('');
    logArea.scrollTop = logArea.scrollHeight;

    while (allImages.length < data.images.length) {
      const img = data.images[allImages.length];
      allImages.push(img);
      addImageToGrid(img);
    }

    if (data.done) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }).catch(() => {});
}

function addImageToGrid(img) {
  const grid = document.getElementById('thumbGrid');
  const card = document.createElement('div');
  card.className = 'thumb-card';
  card.dataset.idx = img.idx;
  card.onclick = () => toggleSelect(img.idx);
  card.innerHTML =
    '<img src="/image?path=' + encodeURIComponent(img.path) + '" loading="lazy">' +
    '<div class="thumb-label">#' + img.idx + '</div>';
  grid.appendChild(card);
}

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
  document.getElementById('saveBtn').disabled = selected.size === 0;
  const panel = document.getElementById('revisionPanel');
  if (selected.size > 0) {
    panel.classList.add('visible');
  } else {
    panel.classList.remove('visible');
  }
}

function startRevision() {
  const prompt = document.getElementById('revisionPrompt').value.trim();
  if (!prompt) { alert('Enter revision instructions.'); return; }
  const indices = Array.from(selected).join(',');

  fetch('/revise', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: 'indices=' + indices + '&prompt=' + encodeURIComponent(prompt),
  }).then(r => r.json()).then(data => {
    if (data.error) { alert(data.error); return; }
    document.getElementById('gridLabel').textContent = 'Revisions — Select Your Finals';
    selected.clear();
    updateSelectedUI();
    startPolling();
  });
}

function startVariations() {
  const indices = Array.from(selected).join(',');
  fetch('/vary?indices=' + indices).then(r => r.json()).then(data => {
    if (data.error) { alert(data.error); return; }
    document.getElementById('gridLabel').textContent = 'Variations — Select Your Finals';
    selected.clear();
    updateSelectedUI();
    startPolling();
  });
}

function saveFinals() {
  const indices = Array.from(selected).join(',');
  fetch('/save_finals?indices=' + indices).then(r => r.json()).then(data => {
    if (data.error) { alert(data.error); return; }
    alert('Saved ' + data.count + ' finals to:\\n' + data.finals_dir);
  });
}

function openFolder() { fetch('/open_folder'); }

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
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

        if path == "/":
            self._serve_html()
        elif path == "/status":
            safe = {k: v for k, v in status.items() if k != "speakers"}
            self._json_response(safe)
        elif path == "/image":
            self._serve_image(params.get("path", ""))
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

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if path == "/generate":
            self._handle_generate_post(body)
        elif path == "/revise":
            self._handle_revise_post(body)
        else:
            self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

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
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def _handle_generate_post(self, body):
        global status
        if status["running"]:
            self._json_response({"error": "Generation already in progress"})
            return

        content_type = self.headers.get("Content-Type", "")

        if "multipart/form-data" in content_type:
            fields, files = parse_multipart(self.headers, body)
        else:
            # URL-encoded fallback
            fields = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="replace")))
            files = {}

        transcript = fields.get("transcript", "").strip()
        visual_concept = fields.get("visual_concept", "").strip()

        if not transcript:
            self._json_response({"error": "Episode transcript is required"})
            return

        # Upload speaker/inspiration files to Gemini File API
        client = get_client()
        speaker_refs = upload_files_from_bytes(client, files.get("speakers", []), "speaker")
        inspo_refs = upload_files_from_bytes(client, files.get("inspiration", []), "inspiration")

        # Build output dir
        slug = re.sub(r"[^a-z0-9]+", "-", transcript[:40].lower()).strip("-") or "episode"
        date = datetime.date.today().isoformat()
        episode_dir = os.path.join(THUMBNAILS_DIR, f"{slug}-{date}")
        round_dir = os.path.join(episode_dir, "round1")
        os.makedirs(round_dir, exist_ok=True)

        # Save input info
        info_dict = {
            "transcript_length": len(transcript),
            "visual_concept": visual_concept,
            "num_speaker_photos": len(speaker_refs),
            "num_inspiration_images": len(inspo_refs),
        }
        with open(os.path.join(episode_dir, "episode.json"), "w") as f:
            json.dump(info_dict, f, indent=2)

        # Parse count
        try:
            count = int(fields.get("count", "20"))
        except ValueError:
            count = 20
        count = max(5, min(100, count))

        # Build prompts
        prompts = build_all_prompts(transcript, visual_concept, speaker_refs, inspo_refs, count)
        save_metadata(round_dir, info_dict, len(prompts), "round1")

        # Start generation
        status["episode_dir"] = episode_dir
        status["speakers"] = speaker_refs
        status["round_num"] = 1
        run_generation(client, prompts, round_dir, "round1")

        self._json_response({"ok": True, "output_dir": round_dir, "count": len(prompts)})

    def _handle_revise_post(self, body):
        global status
        if status["running"]:
            self._json_response({"error": "Generation already in progress"})
            return

        params = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="replace")))
        indices_raw = params.get("indices", "")
        custom_prompt = params.get("prompt", "").strip()

        indices = [int(x) for x in indices_raw.split(",") if x.strip().isdigit()]
        if not indices:
            self._json_response({"error": "No images selected"})
            return
        if not custom_prompt:
            self._json_response({"error": "Revision prompt is required"})
            return

        selected_images = []
        for img_info in status["images"]:
            if img_info["idx"] in indices and os.path.isfile(img_info["path"]):
                selected_images.append(Image.open(img_info["path"]))

        if not selected_images:
            self._json_response({"error": "Could not load selected images"})
            return

        speakers = status.get("speakers", [])
        episode_dir = status.get("episode_dir", "")
        status["round_num"] = status.get("round_num", 1) + 1
        round_dir = os.path.join(episode_dir, f"round{status['round_num']}")
        os.makedirs(round_dir, exist_ok=True)

        count_per = 3
        prompts = build_revision_prompts(selected_images, speakers, custom_prompt, count_per)

        client = get_client()
        run_generation(client, prompts, round_dir, "revision")

        self._json_response({"ok": True, "output_dir": round_dir, "count": len(prompts)})

    def _handle_vary(self, params):
        global status
        if status["running"]:
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
            self._json_response({"error": "Could not load selected images"})
            return

        speakers = status.get("speakers", [])
        episode_dir = status.get("episode_dir", "")
        status["round_num"] = status.get("round_num", 1) + 1
        round_dir = os.path.join(episode_dir, f"round{status['round_num']}")
        os.makedirs(round_dir, exist_ok=True)

        count_per = 3
        prompts = build_variation_prompts(selected_images, speakers, count_per)

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
    print(f"Nano Banana 2 Thumbnail Generator")
    print(f"Model: {GEMINI_MODEL}")
    print(f"Output: {THUMBNAILS_DIR}")
    print(f"Examples: {EXAMPLES_DIR}")
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
