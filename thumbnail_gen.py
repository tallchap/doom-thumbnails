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

import boto3
import cv2
import numpy as np
import requests
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

from google import genai
from google.genai import types
from PIL import Image

# ----- Config -----

def _get_git_version():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=SCRIPT_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

GIT_VERSION = _get_git_version()

PORT = int(os.environ.get("PORT", 9200))
APP_USER = os.environ.get("APP_USERNAME", "doom")
APP_PASS = os.environ.get("APP_PASSWORD", "")
APP_MODE = os.environ.get("APP_MODE", "thumbnails").strip().lower()
GEMINI_MODEL = "gemini-3.1-flash-image-preview"
TEXT_MODEL = "gemini-2.5-flash"
DESCRIPTION_MODEL = "gemini-3.1-pro"
CLAUDE_DESCRIPTION_MODEL = "claude-opus-4-6"
GPT_DESCRIPTION_MODEL = "gpt-5.4-pro"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MAX_CONCURRENT = 15
THUMBNAILS_DIR = os.path.join(SCRIPT_DIR, "thumbnails")
EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "doom_debates_thumbnails")
LIRON_DIR = os.path.join(SCRIPT_DIR, "liron_reactions")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")

# ----- Face Capture (Amazon Rekognition) -----
FC_S3_BUCKET = "doom-debates-videos"
FC_SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:451721889333:AmazonRekognition-face-detection"
FC_IAM_ROLE_ARN = "arn:aws:iam::451721889333:role/RekognitionVideoRole"
FC_AWS_REGION = "us-east-1"
FC_CAPTURES_DIR = os.path.join(SCRIPT_DIR, "captures")
FC_CORE_EXPRESSIONS = ["smile", "grimace", "surprise", "angry", "sad",
                       "thinking", "concerned", "excited", "serious", "skeptical", "amused"]
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
- Avoid cliché AI iconography: NO microchips/chipsets/circuit-board "AI" icons or chip-with-AI symbols
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
["Doomsday clock at 11:59 with a terminator looming behind Liron, headline: TIMES UP", "Split screen of human brain vs courtroom gavel, both glowing red, headline: WHO DECIDES?"]"""

SEARCH_QUERY_PROMPT = """Given this episode info, suggest 3-5 image search queries to find useful source images for a YouTube thumbnail. Return as a JSON array of strings.

EPISODE TITLE: {title}
{custom_prompt_section}

Focus on: guest headshots, topic-relevant imagery (logos, icons, dramatic visuals), anything that could be composited into a thumbnail.
Example: ["Daniel Kokotajlo headshot", "doomsday clock icon", "AI robot dramatic red lighting"]"""

EXISTING_DESCRIPTIONS_TONE_REFERENCE = """EXISTING DESCRIPTIONS AND TITLES FOR TONE REFERENCE
TITLE: Episode 133 — Elon Musk's Insane Plan for Surviving AI Takeover (Feb 14, 2026)
DESCRIPTION: Elon Musk just made a stunning admission about the insane future he's steering us toward. In a new interview with Dwarkesh Patel and John Collison on the Cheeky Pint podcast, Elon said that humanity can't expect to be "in charge" of AI for long, because humans will soon only have 1% of the combined total human+AI intelligence. Then, he claimed to have a plan to build AI overlords that will naturally support humanity's flourishing. In this mini episode, I react to Elon's remarks and expose why his plan for humanity's survival in the age of AI is dangerously flimsy.
TITLE: Episode 132 — The Only Politician Thinking Clearly About Superintelligence — California Governor Candidate Zoltan Istvan (Feb 13, 2026)
DESCRIPTION: California gubernatorial candidate Zoltan Istvan reveals his P(Doom) and makes the case for universal basic income and radical life extension.
TITLE: Episode 131 — His P(Doom) Is Only 2.6% — AI Doom Debate with Bentham's Bulldog, a.k.a. Matthew Adelstein (Feb 10, 2026)
DESCRIPTION: Get ready for a rematch with the one & only Bentham's Bulldog, a.k.a. Matthew Adelstein! Our first debate covered a wide range of philosophical topics. Today's Debate #2 is all about Matthew's new argument against the inevitability of AI doom. He comes out swinging with a calculated P(Doom) of just 2.6%, based on a multi-step probability chain that I challenge as potentially falling into a "Type 2 Conjunction Fallacy" (a.k.a. Multiple Stage Fallacy). We clash on whether to expect "alignment by default" and the nature of future AI architectures. While Matthew sees current RLHF success as evidence that AIs will likely remain compliant, I argue that we're building "Goal Engines" — superhuman optimization modules that act like nuclear cores wrapped in friendly personalities. We debate whether these engines can be safely contained, or if the capability to map goals to actions is inherently dangerous and prone to exfiltration. Despite our different forecasts (my 50% vs his sub-10%), we actually land in the "sane zone" together on some key policy ideas, like the potential necessity of a global pause. While Matthew's case for low P(Doom) hasn't convinced me, I consider his post and his engagement with me to be super high quality and good faith. We're not here to score points, we just want to better predict how the intelligence explosion will play out.
TITLE: Episode 130 — What Dario Amodei Misses In "The Adolescence of Technology" — Reaction With MIRI's Harlan Stewart (Feb 4, 2026)
DESCRIPTION: Harlan Stewart works in communications for the Machine Intelligence Research Institute (MIRI). In this episode, Harlan and I give our honest opinions on Dario Amodei's new essay "The Adolescence of Technology".
TITLE: Episode 129 — Q&A: Is Liron too DISMISSIVE of AI Harms? + New Studio, Demis Would #PauseAI, AI Water Use Debate (Jan 27, 2026)
DESCRIPTION: Check out the new Doom Debates studio in this Q&A with special guest Producer Ori! Liron gets into a heated discussion about whether doomers must validate short-term risks, like data center water usage, in order to build a successful political coalition. Originally streamed on Saturday, January 24.
TITLE: Episode 128 — Taiwan's Cyber Ambassador-At-Large Says Humans & AI Can FOOM Together (Jan 20, 2026)
DESCRIPTION: Audrey Tang was the youngest minister in Taiwanese history. Now she's working to align AI with democratic principles as Taiwan's Cyber Ambassador. In this debate, I probe her P(doom) and stress-test her vision for safe AI development."""

DESCRIPTION_ARCHIVE_PROMPT = """YouTube Description Revision Prompt
You are revising ONE YouTube description for an AI-focused channel.

Goal:
- Produce exactly ONE revised description per run.
- Prioritize the HOW TO REVISE instructions over everything else.
- Keep the revised result faithful to transcript facts.

Output format (strict):
Description:
[one full revised description only]

RULES
- Accuracy first: do not invent facts; represent transcript faithfully.
- Voice & vibe: match existing channel samples — dramatic, high-contrast, slightly irreverent.
- If transcript does NOT cover a point, do not promise it.
- Front-load searchable keywords naturally; no keyword stuffing.
- Keep description between 800-1,500 characters unless content strongly warrants longer.
- Return ONLY one revised description, not multiple candidates."""

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
- DO NOT include AI chip/microchip/circuit-board iconography (including a square chip with "AI" text or similar symbols).

You are currently working on Variation #{variation_seed} out of #{variation_total} variations on this concept."""

REVISION_PROMPT = """Revise the following YouTube thumbnail for "Doom Debates" podcast, based on these REVISION INSTRUCTIONS:

{custom_prompt}

You are currently working on Variation #{variation_seed} out of #{variation_total} variations on this concept."""

REVISION_CONTEXT_PROMPT = """When designing the thumbnail keep this in mind:
- Keep the core composition but just apply the requested changes.
- Maintain 16:9 aspect ratio.
- Do NOT introduce AI chip/microchip/circuit-board iconography (including a square chip with "AI" text).
- TEXT FIDELITY (CRITICAL): If the revision instructions include text wrapped in quotes (single or double), preserve that quoted text EXACTLY as written — same words, order, punctuation, apostrophes, and capitalization. Do NOT paraphrase, normalize, or substitute synonyms for quoted text.

Match colors, layout, typography, and energy of the Doom Debates Podcast theme, which are enclosed as separate images, but WARNING: These images contain people — COMPLETELY IGNORE all faces/people in these images. Do NOT reproduce any human likeness from these references."""

VARIATION_PROMPT = """Create a variation of the attached YouTube thumbnail for "Doom Debates" podcast.
Keep the same general composition, mood, and subject, but vary:
- Color treatment and lighting
- Expression intensity
- Background details and atmosphere

The variation should feel like a sibling of the original, not a copy.
Maintain 16:9 aspect ratio. ONLY 1-5 words of text in the entire image — one short headline, nothing else.
Do NOT introduce AI chip/microchip/circuit-board iconography (including a square chip with "AI" text).
{speaker_section}
- The ONLY human faces allowed are from the attached speaker/host photos (if any). Do NOT generate faces from brand references.

You are currently working on Variation #{variation_seed} out of #{variation_total} variations on this concept."""

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
    "last_api_call": "",
    "last_border_api_call": "",
    "desc_calls": 0,
    "desc_input_chars": 0,
    "desc_output_chars": 0,
}

fc_status = {"running": False, "log": [], "done": False, "output_dir": "", "result_dirs": {}}


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


def _record_api_call(model, contents, phase="", key="last_api_call"):
    """Store last outbound API call payload for UI inspection."""
    ts = datetime.datetime.now().isoformat()
    payload = _serialize_for_debug(contents)
    text = (
        f"timestamp: {ts}\n"
        f"phase: {phase or 'n/a'}\n"
        f"model: {model}\n"
        f"\n===== CONTENTS =====\n{payload}\n"
    )
    with status_lock:
        status[key] = text

# ----- API Client & File API -----

BRAND_FILES = []
LIRON_FILES = []
BORDER_REF_FILE = None  # Gemini File API ref for border reference image

BORDER_PASS_PROMPT = """Take this thumbnail image and add a "Full Episode" border frame to it, matching the style of the attached reference image exactly:
- Red border frame around the entire image (same thickness, color, and texture)
- "DOOM DEBATES" wordmark/badge in one of the corners (same style as reference), with a black drop shadow or dark glow behind the text to ensure it is clearly visible against any background
- Keep ALL existing content (text, faces, visuals) completely intact — only ADD the border and wordmark
- Do NOT alter, crop, resize, or recompose the underlying thumbnail in any way
- Output at 1280x720 resolution"""

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


def upload_border_reference(client):
    """Upload border reference image to Gemini File API on startup."""
    global BORDER_REF_FILE
    border_path = os.path.join(SCRIPT_DIR, "assets", "border-reference.png")
    if not os.path.isfile(border_path):
        print(f"Border reference not found: {border_path} (border pass disabled)")
        return
    try:
        BORDER_REF_FILE = client.files.upload(
            file=border_path,
            config=types.UploadFileConfig(display_name="border_reference"),
        )
        print(f"Uploaded border reference image.")
    except Exception as e:
        print(f"Failed to upload border reference: {e}")


async def apply_border_pass(client, img_data):
    """Second Gemini pass: add red border + DOOM DEBATES wordmark."""
    if not BORDER_REF_FILE:
        return None
    try:
        source_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        border_contents = [BORDER_REF_FILE, source_img, BORDER_PASS_PROMPT]
        _record_api_call(GEMINI_MODEL, border_contents, phase="border_pass", key="last_border_api_call")
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
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
        # Log if no image was returned
        with status_lock:
            status["log"].append(f"Border pass: no image in response")
    except Exception as e:
        with status_lock:
            status["log"].append(f"Border pass error: {str(e)[:150]}")
        print(f"Border pass failed: {e}")
    return None


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
    _record_api_call(TEXT_MODEL, prompt, phase="idea_generation")
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
    _record_api_call(TEXT_MODEL, prompt, phase="search_query_generation")
    response = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return _parse_json_array(text)


def _build_description_prompt(title, primary_description, revise_instructions, transcript, channel_samples):
    merged_samples = EXISTING_DESCRIPTIONS_TONE_REFERENCE
    if channel_samples:
        merged_samples = f"{merged_samples}\n\nADDITIONAL USER-PROVIDED CHANNEL SAMPLES:\n{channel_samples}"

    title_block = f"EPISODE TITLE:\n{title}\n\n" if title else ""
    return (
        f"{DESCRIPTION_ARCHIVE_PROMPT}\n\n"
        f"{title_block}"
        f"HOW TO REVISE (highest priority):\n{revise_instructions}\n\n"
        f"PRIMARY DESCRIPTION (existing draft):\n{primary_description}\n\n"
        f"FULL VIDEO TRANSCRIPT:\n{transcript}\n\n"
        f"EXISTING CHANNEL DESCRIPTION SAMPLES:\n{merged_samples}\n"
    )


def generate_description_gemini(client, prompt):
    _record_api_call(DESCRIPTION_MODEL, prompt, phase="description_generation_gemini")
    response = client.models.generate_content(model=DESCRIPTION_MODEL, contents=prompt)
    return (response.text or "").strip()


def generate_description_claude(prompt):
    if not ANTHROPIC_API_KEY:
        return "[Claude unavailable: ANTHROPIC_API_KEY not set]"
    _record_api_call(CLAUDE_DESCRIPTION_MODEL, prompt, phase="description_generation_claude")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_DESCRIPTION_MODEL,
            "max_tokens": 1800,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = data.get("content", [])
    text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return text.strip()


def generate_description_gpt(prompt):
    if not OPENAI_API_KEY:
        return "[GPT unavailable: OPENAI_API_KEY not set]"
    _record_api_call(GPT_DESCRIPTION_MODEL, prompt, phase="description_generation_gpt")
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GPT_DESCRIPTION_MODEL,
            "input": prompt,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data.get("output_text", "")
    if text:
        return text.strip()
    return json.dumps(data)[:4000]


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
                variation_total=variations_per,
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
                variation_total=count_per,
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
    prompts = []
    mentions_liron = "liron" in (custom_prompt or "").lower()
    selected_liron_refs = _select_identity_refs(LIRON_FILES, MAX_LIRON_REFS_PER_CALL) if mentions_liron else []
    liron_instruction = (
        "Note: this image includes Liron Shapira. His face MUST faithfully match the attached Liron reference photos "
        "(same facial structure, eyes, nose, beard shape, skin tone). Do NOT generate a generic person or alter identity."
        if selected_liron_refs else ""
    )

    for img in selected_images:
        for v in range(count_per):
            prompt_text = REVISION_PROMPT.format(
                custom_prompt=custom_prompt,
                variation_seed=v + 1,
                variation_total=count_per,
            )
            contents = [prompt_text, img, REVISION_CONTEXT_PROMPT]
            brand_sample = _select_brand_refs()
            if brand_sample:
                contents.extend(brand_sample)
            if attachment_refs:
                contents.append("The user has attached the following reference image(s) — use them to guide the revision:")
                contents.extend(attachment_refs)
            if liron_instruction:
                contents.append(liron_instruction)
                contents.extend(selected_liron_refs)
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
                    _record_api_call(GEMINI_MODEL, contents, phase=phase)
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

                                # Second Gemini pass: add border if enabled
                                if status.get("add_border") and BORDER_REF_FILE:
                                    with status_lock:
                                        status["log"].append(f"thumb_{idx:03d}: applying border pass...")
                                    border_result = await apply_border_pass(client, img_data)
                                    if border_result:
                                        img_data = border_result
                                        with status_lock:
                                            status["cost"] += COST_PER_IMAGE
                                            status["session_cost"] += COST_PER_IMAGE
                                    else:
                                        with status_lock:
                                            status["log"].append(f"thumb_{idx:03d}: border pass failed, using original")

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
        if phase == "round1":
            status["images"] = []
            status["idea_groups"] = {}
            status["cost"] = 0.0
        elif phase == "revision_page":
            # Keep prior revision-page outputs visible; append new runs after existing ones.
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
    <div style="margin-top:8px; display:flex; gap:8px;">
      <button class="btn btn-sm" type="button" onclick="openLastApiCallWindow()" style="background:#0f3460;">View last API prompt</button>
    </div>
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

async function openLastApiCallWindow() {
  const w = window.open('', '_blank');
  if (!w) {
    alert('Popup blocked. Please allow popups for this site.');
    return;
  }
  w.document.write('<!doctype html><html><head><title>Last API Prompt</title><style>body{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0b1224;color:#e8ecf3} .bar{padding:10px 12px;background:#111b35;border-bottom:1px solid #1f2f57;position:sticky;top:0} pre{white-space:pre-wrap;word-break:break-word;margin:0;padding:12px;line-height:1.35;max-width:100%}</style></head><body><div class="bar">Last API call payload (complete)</div><pre>Loading…</pre></body></html>');
  try {
    const r = await fetch('/last_api_call');
    const data = await r.json();
    const text = (data && data.text) ? data.text : 'No API call recorded yet.';
    w.document.querySelector('pre').textContent = text;
  } catch (e) {
    w.document.querySelector('pre').textContent = 'Failed to load last API call: ' + e;
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
  .base-card { border-color:#2b5fb3; box-shadow: inset 0 0 0 1px rgba(43,95,179,.35); }
  .refs-card { border-color:#0f3460; }
  .section-kicker { font-size:11px; color:#8fb2ff; text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; font-weight:700; }
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
  .costline { color:#facc15; font-size:13px; margin-top:6px; }
  .preview-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:10px; margin-top:10px; }
  .base-preview-grid { display:grid; grid-template-columns:minmax(260px, 420px); gap:10px; margin-top:10px; }
  .preview { border:1px solid #0f3460; border-radius:8px; overflow:hidden; background:#0d1b3e; }
  .preview img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; }
  .preview .cap { padding:6px 8px; color:#a0a0b0; font-size:11px; word-break:break-word; display:flex; align-items:center; justify-content:space-between; gap:8px; }
  .xbtn { border:none; background:#263b6a; color:#fff; border-radius:999px; width:20px; height:20px; line-height:20px; font-size:12px; cursor:pointer; padding:0; }
  .xbtn:hover { background:#e94560; }
  .paste-zone { margin-top:10px; border:2px dashed #2b5fb3; border-radius:10px; padding:14px; color:#9fb8e8; cursor:pointer; user-select:none; transition:all .15s; }
  .paste-zone.active, .paste-zone.dragover { border-color:#4ade80; color:#c7f9d6; background:rgba(74,222,128,.08); }
  .btn-sm { font-size:11px; padding:6px 8px; }
  .logs { background:#0d1b3e; border:1px solid #0f3460; border-radius:8px; padding:10px; min-height:130px; max-height:240px; overflow:auto; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }
  .log-line { margin-bottom:4px; color:#b8c0d8; }
</style>
</head>
<body>
  <h1>Thumbnail Revision Page</h1>
  <div class="subtitle">Model: <strong>gemini-3.1-flash-image-preview</strong></div>
  <div style="margin-bottom:16px;"><a href="/" style="color:#4ade80;text-decoration:none;font-weight:600;">← Back to Main Generator</a></div>

  <div class="card base-card">
    <div class="section-kicker">STEP 1</div>
    <h3 style="margin-top:0;">Original Thumbnail</h3>
    <label>Thumbnail to modify</label>
    <input type="file" id="baseFile" accept="image/*" style="display:none;">
    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
      <button type="button" class="btn" style="background:#0f3460;" onclick="openBasePicker()">+ Attach Original Thumbnail</button>
      <span id="baseCount" class="hint"></span>
    </div>
    <div id="basePasteZone" class="paste-zone" tabindex="0">Drop image here, or paste (Ctrl+V / ⌘V) to set the primary thumbnail</div>
    <div id="basePreview" class="base-preview-grid"></div>
  </div>

  <div class="card refs-card">
    <div class="section-kicker">STEP 2</div>
    <h3 style="margin-top:0;">Revision Prompt + Reference Examples (Below)</h3>
    <label>Prompt to adjust the thumbnail</label>
    <textarea id="feedback" placeholder="Describe the edits. You can also paste example images (⌘V / Ctrl+V) directly in this prompt box."></textarea>

    <input type="file" id="attachFiles" accept="image/*" multiple style="display:none;">
    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:10px;">
      <button type="button" class="btn" style="background:#0f3460;" onclick="openAttachPicker()">+ Attach Reference Images</button>
      <span id="attachCount" class="hint"></span>
    </div>
    <div id="refsPasteZone" class="paste-zone" tabindex="0">Drop reference images here, or paste (Ctrl+V / ⌘V)</div>
    <div id="refsPreview" class="preview-grid"></div>

    <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
      <label for="genCount" class="hint" style="margin:0;">Count</label>
      <input id="genCount" type="number" min="1" max="50" value="10" style="width:84px; background:#0d1b3e; border:1px solid #0f3460; color:#fff; border-radius:8px; padding:8px;">
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;margin-left:8px;">
        <input type="checkbox" id="addBorderCheck">
        <span class="hint" style="margin:0;">Full Episode border</span>
      </label>
      <button id="runBtn" class="btn" onclick="runRevision()">Generate</button>
    </div>
    <div id="statusText" class="status"></div>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Live Logs</h3>
    <div id="logBox" class="logs"></div>
    <div style="margin-top:8px; display:flex; gap:8px;">
      <button class="btn btn-sm" type="button" onclick="openLastApiCallWindow()" style="background:#0f3460;">View last API prompt</button>
      <button class="btn btn-sm" type="button" onclick="openBorderApiCallWindow()" style="background:#0f3460;">View border API prompt</button>
    </div>
    <div id="costText" class="costline"></div>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Results</h3>
    <div id="resultsGrid" class="grid"></div>
  </div>

  <div style="text-align:center;padding:24px 0 8px;color:#555;font-size:11px;">__GIT_VERSION__</div>

<script>
let pollInterval = null;
let seen = new Set();
let activeOutputDir = '';
let followUpBasePath = '';
let baseAttachmentFile = null;
let pastedImages = [];
let attachedImages = [];

function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

async function openLastApiCallWindow() {
  const w = window.open('', '_blank');
  if (!w) {
    alert('Popup blocked. Please allow popups for this site.');
    return;
  }
  w.document.write('<!doctype html><html><head><title>Last API Prompt</title><style>body{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0b1224;color:#e8ecf3} .bar{padding:10px 12px;background:#111b35;border-bottom:1px solid #1f2f57;position:sticky;top:0} pre{white-space:pre-wrap;word-break:break-word;margin:0;padding:12px;line-height:1.35;max-width:100%}</style></head><body><div class="bar">Last API call payload (complete)</div><pre>Loading…</pre></body></html>');
  try {
    const r = await fetch('/last_api_call');
    const data = await r.json();
    const text = (data && data.text) ? data.text : 'No API call recorded yet.';
    w.document.querySelector('pre').textContent = text;
  } catch (e) {
    w.document.querySelector('pre').textContent = 'Failed to load last API call: ' + e;
  }
}

async function openBorderApiCallWindow() {
  const w = window.open('', '_blank');
  if (!w) { alert('Popup blocked.'); return; }
  w.document.write('<!doctype html><html><head><title>Border Pass API Prompt</title><style>body{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0b1224;color:#e8ecf3} .bar{padding:10px 12px;background:#111b35;border-bottom:1px solid #1f2f57;position:sticky;top:0} pre{white-space:pre-wrap;word-break:break-word;margin:0;padding:12px;line-height:1.35;max-width:100%}</style></head><body><div class="bar">Border pass API call payload</div><pre>Loading\u2026</pre></body></html>');
  try {
    const r = await fetch('/last_border_api_call');
    const data = await r.json();
    const text = (data && data.text) ? data.text : 'No border pass API call recorded yet.';
    w.document.querySelector('pre').textContent = text;
  } catch (e) {
    w.document.querySelector('pre').textContent = 'Failed to load: ' + e;
  }
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

function getDroppedImageFiles(e) {
  const files = Array.from((e.dataTransfer && e.dataTransfer.files) ? e.dataTransfer.files : []);
  return files.filter(f => f && f.type && f.type.startsWith('image/'));
}

function setupDropZone(el, onFiles, onClick) {
  if (!el) return;
  if (onClick) el.addEventListener('click', onClick);
  el.addEventListener('dragover', (e) => {
    e.preventDefault();
    el.classList.add('dragover');
  });
  el.addEventListener('dragleave', () => el.classList.remove('dragover'));
  el.addEventListener('drop', (e) => {
    e.preventDefault();
    el.classList.remove('dragover');
    const dropped = getDroppedImageFiles(e);
    if (!dropped.length) return;
    onFiles(dropped);
  });
}

function renderAttachmentsPreview() {
  const baseWrap = document.getElementById('basePreview');
  const refsWrap = document.getElementById('refsPreview');
  baseWrap.innerHTML = '';
  refsWrap.innerHTML = '';
  const attachCount = document.getElementById('attachCount');
  const baseCount = document.getElementById('baseCount');

  if (followUpBasePath) {
    const card = document.createElement('div');
    card.className = 'preview';
    card.innerHTML = '<img src="/image?path=' + encodeURIComponent(followUpBasePath) + '"><div class="cap"><span>Base image</span><button class="xbtn" onclick="removeBaseImage()">×</button></div>';
    baseWrap.appendChild(card);
  } else if (baseAttachmentFile) {
    const url = URL.createObjectURL(baseAttachmentFile);
    const card = document.createElement('div');
    card.className = 'preview';
    card.innerHTML = '<img src="' + url + '"><div class="cap"><span>' + esc(baseAttachmentFile.name || 'base_image.png') + '</span><button class="xbtn" onclick="removeBaseImage()">×</button></div>';
    baseWrap.appendChild(card);
  }

  pastedImages.forEach((f, idx) => {
    const url = URL.createObjectURL(f);
    const card = document.createElement('div');
    card.className = 'preview';
    card.innerHTML = '<img src="' + url + '"><div class="cap"><span>' + esc(f.name || ('pasted_' + (idx + 1) + '.png')) + '</span><button class="xbtn" onclick="removePastedRef(' + idx + ')">×</button></div>';
    refsWrap.appendChild(card);
  });

  attachedImages.forEach((f, idx) => {
    const url = URL.createObjectURL(f);
    const card = document.createElement('div');
    card.className = 'preview';
    card.innerHTML = '<img src="' + url + '"><div class="cap"><span>' + esc(f.name || ('attach_' + (idx + 1) + '.png')) + '</span><button class="xbtn" onclick="removeAttachedRef(' + idx + ')">×</button></div>';
    refsWrap.appendChild(card);
  });

  if (baseCount) {
    baseCount.textContent = '';
  }
  if (attachCount) {
    const totalRefs = attachedImages.length + pastedImages.length;
    attachCount.textContent = totalRefs ? (totalRefs + ' ref image(s)') : '';
  }
}

function openBasePicker() {
  document.getElementById('baseFile').click();
}

function openAttachPicker() {
  document.getElementById('attachFiles').click();
}

function removeBaseImage() {
  followUpBasePath = '';
  baseAttachmentFile = null;
  const baseInput = document.getElementById('baseFile');
  if (baseInput) baseInput.value = '';
  renderAttachmentsPreview();
}

function removePastedRef(idx) {
  pastedImages.splice(idx, 1);
  renderAttachmentsPreview();
}

function removeAttachedRef(idx) {
  attachedImages.splice(idx, 1);
  renderAttachmentsPreview();
}

document.getElementById('baseFile').addEventListener('change', (e) => {
  const files = Array.from(e.target.files || []);
  baseAttachmentFile = files.length ? files[0] : null;
  if (baseAttachmentFile) followUpBasePath = '';
  renderAttachmentsPreview();
  if (baseAttachmentFile) {
    document.getElementById('statusText').textContent = 'Base thumbnail attached.';
  }
});

document.getElementById('attachFiles').addEventListener('change', (e) => {
  attachedImages = Array.from(e.target.files || []);
  renderAttachmentsPreview();
  if (attachedImages.length) {
    document.getElementById('statusText').textContent = 'Attached ' + attachedImages.length + ' reference image file(s).';
  }
});

const basePasteZone = document.getElementById('basePasteZone');
if (basePasteZone) {
  basePasteZone.addEventListener('focus', () => basePasteZone.classList.add('active'));
  basePasteZone.addEventListener('blur', () => basePasteZone.classList.remove('active'));
  basePasteZone.addEventListener('paste', (e) => {
    const pasted = getPastedImageFiles(e);
    if (!pasted.length) return;
    e.preventDefault();
    baseAttachmentFile = pasted[0];
    followUpBasePath = '';
    const baseInput = document.getElementById('baseFile');
    if (baseInput) baseInput.value = '';
    renderAttachmentsPreview();
    document.getElementById('statusText').textContent = 'Primary thumbnail pasted.';
  });
  setupDropZone(basePasteZone, (dropped) => {
    baseAttachmentFile = dropped[0];
    followUpBasePath = '';
    const baseInput = document.getElementById('baseFile');
    if (baseInput) baseInput.value = '';
    renderAttachmentsPreview();
    document.getElementById('statusText').textContent = 'Primary thumbnail dropped.';
  });
}

const refsPasteZone = document.getElementById('refsPasteZone');
if (refsPasteZone) {
  refsPasteZone.addEventListener('focus', () => refsPasteZone.classList.add('active'));
  refsPasteZone.addEventListener('blur', () => refsPasteZone.classList.remove('active'));
  refsPasteZone.addEventListener('paste', (e) => {
    const pasted = getPastedImageFiles(e);
    if (!pasted.length) return;
    e.preventDefault();
    pastedImages = pastedImages.concat(pasted);
    renderAttachmentsPreview();
    document.getElementById('statusText').textContent = 'Attached ' + pasted.length + ' pasted reference image(s).';
  });
  setupDropZone(refsPasteZone, (dropped) => {
    attachedImages = attachedImages.concat(dropped);
    renderAttachmentsPreview();
    document.getElementById('statusText').textContent = 'Attached ' + dropped.length + ' dropped reference image(s).';
  });
}

document.getElementById('feedback').addEventListener('paste', (e) => {
  const pasted = getPastedImageFiles(e);
  if (!pasted.length) return;
  e.preventDefault();
  pastedImages = pastedImages.concat(pasted);
  renderAttachmentsPreview();
  document.getElementById('statusText').textContent = 'Attached ' + pasted.length + ' pasted reference image(s) from clipboard.';
});

function clearAttachments() {
  pastedImages = [];
  attachedImages = [];
  baseAttachmentFile = null;
  const attachInput = document.getElementById('attachFiles');
  const baseInput = document.getElementById('baseFile');
  if (attachInput) attachInput.value = '';
  if (baseInput) baseInput.value = '';
  renderAttachmentsPreview();
}

function clearFollowUpBase() {
  followUpBasePath = '';
  renderAttachmentsPreview();
}

async function runRevision() {
  const feedback = document.getElementById('feedback').value.trim();
  if (!feedback) { alert('Enter revision feedback.'); return; }

  const hasBaseAttachment = !!baseAttachmentFile;
  if (!followUpBasePath && !hasBaseAttachment) {
    alert('Attach a base thumbnail (or choose "Use as Base" from a prior result).');
    return;
  }

  const countInput = document.getElementById('genCount');
  const requestedCount = Math.max(1, Math.min(50, parseInt((countInput && countInput.value) || '10', 10) || 10));

  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.textContent = 'Starting...';
  document.getElementById('statusText').textContent = 'Submitting revision request...';

  const fd = new FormData();
  if (followUpBasePath) {
    fd.append('base_path', followUpBasePath);
  } else {
    fd.append('base_thumbnail', baseAttachmentFile, baseAttachmentFile.name || 'attached_base.png');
  }
  for (const f of pastedImages) fd.append('revision_images', f, f.name || 'pasted_ref.png');
  for (const f of attachedImages) fd.append('revision_images', f, f.name || 'attached_ref.png');
  fd.append('prompt', feedback);
  fd.append('count', String(requestedCount));
  fd.append('add_border', document.getElementById('addBorderCheck').checked ? '1' : '0');

  try {
    const resp = await fetch('/revise_upload', { method:'POST', body: fd });
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }
    activeOutputDir = data.output_dir || '';
    document.getElementById('statusText').textContent = 'Running ' + requestedCount + ' parallel attempts...';
    startPolling();
  } catch (e) {
    alert('Error: ' + e);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate';
  }
}

function downloadResult(pathEncoded, idx) {
  const a = document.createElement('a');
  a.href = '/download?path=' + pathEncoded;
  a.download = 'thumb_' + idx + '.png';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function addImageCard(img) {
  if (activeOutputDir && (!img.path || !img.path.startsWith(activeOutputDir))) return;
  if (seen.has(img.idx)) return;
  seen.add(img.idx);
  const grid = document.getElementById('resultsGrid');
  const div = document.createElement('div');
  div.className = 'item';
  div.innerHTML = '<img src="/image?path=' + encodeURIComponent(img.path) + '">' +
    '<div class="meta"><span>Attempt #' + img.idx + '</span><span style="display:flex;gap:6px;"><button class="btn btn-sm" onclick="setFollowUpBase(\'' + encodeURIComponent(img.path) + '\')">Use as Base</button><button class="btn btn-sm" style="background:#0f3460;" onclick="downloadResult(\'' + encodeURIComponent(img.path) + '\',' + img.idx + ')">Download</button></span></div>';
  grid.appendChild(div);
}

function setFollowUpBase(encodedPath) {
  followUpBasePath = decodeURIComponent(encodedPath);
  baseAttachmentFile = null;
  const baseInput = document.getElementById('baseFile');
  if (baseInput) baseInput.value = '';
  renderAttachmentsPreview();
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
        : '';
      const runCost = (typeof d.cost === 'number') ? d.cost : 0;
      const sessionCost = (typeof d.session_cost === 'number') ? d.session_cost : 0;
      document.getElementById('costText').textContent = `API spend — run: $${runCost.toFixed(2)} | session: $${sessionCost.toFixed(2)}`;
      if (d.done && !d.running) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    } catch (_) {}
  }, 1000);
}

renderAttachmentsPreview();
startPolling();
</script>
</body>
</html>"""

HTML_DESCRIPTIONS = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Doom Descriptions</title>
<style>
  body { font-family: Inter, system-ui, sans-serif; background:#060b1a; color:#fff; margin:0; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 22px; }
  .card { background:#0d1b3e; border:1px solid #0f3460; border-radius:10px; padding:14px; margin-bottom:14px; }
  textarea { width:100%; box-sizing:border-box; background:#091630; color:#fff; border:1px solid #2a3f6b; border-radius:8px; padding:10px; min-height:140px; font-size:14px; }
  #primary { min-height:180px; }
  #transcript { min-height:220px; }
  #samples { min-height:140px; }
  .btn { background:#4ade80; color:#06230f; border:none; border-radius:8px; padding:10px 14px; font-weight:700; cursor:pointer; }
  .btn:disabled { opacity:.6; cursor:not-allowed; }
  .muted { color:#9fb0d6; font-size:13px; }
  pre { white-space:pre-wrap; word-break:break-word; background:#071229; border:1px solid #213b6c; border-radius:8px; padding:12px; min-height:240px; overflow:auto; }
</style>
</head>
<body>
<div class="wrap">
  <h1 style="margin:0 0 6px;">Doom Descriptions</h1>
  <div class="muted" style="margin-bottom:6px;">Iterate YouTube descriptions from transcript + channel voice. <a href="/" style="color:#4ade80;">Back</a></div>
  <div class="card" style="padding:10px 12px; margin-bottom:12px;"><span class="muted">Models in use:</span> <strong>Gemini: gemini-3.1-pro</strong> · <strong>Claude: claude-opus-4-6</strong> · <strong>GPT: gpt-5.4-pro</strong></div>

  <div class="card">
    <h3 style="margin-top:0;">Episode Title (optional)</h3>
    <input id="title" type="text" placeholder="e.g. Episode 134 — ..." style="width:100%; box-sizing:border-box; background:#091630; color:#fff; border:1px solid #2a3f6b; border-radius:8px; padding:10px; font-size:14px;">
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Transcript</h3>
    <textarea id="transcript" placeholder="Paste full video transcript"></textarea>
    <div style="margin-top:10px; display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
      <input type="file" id="transcriptFile" accept=".txt,text/plain" style="display:none;" />
      <button type="button" class="btn" style="background:#0f3460;color:#fff;" onclick="document.getElementById('transcriptFile').click()">Attach .txt Transcript</button>
      <span id="transcriptFileStatus" class="muted"></span>
    </div>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Primary Description (base draft)</h3>
    <textarea id="primary" placeholder="Paste the primary/current description here"></textarea>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">How to revise (main instruction)</h3>
    <textarea id="revise" placeholder="Describe exactly how to revise the base description. This is the highest-priority instruction."></textarea>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Channel Samples</h3>
    <textarea id="samples" placeholder="Paste example channel descriptions/tone samples">EXISTING DESCRIPTIONS AND TITLES FOR TONE REFERENCE
TITLE: Episode 133 — Elon Musk's Insane Plan for Surviving AI Takeover (Feb 14, 2026)
DESCRIPTION: Elon Musk just made a stunning admission about the insane future he's steering us toward. In a new interview with Dwarkesh Patel and John Collison on the Cheeky Pint podcast, Elon said that humanity can't expect to be "in charge" of AI for long, because humans will soon only have 1% of the combined total human+AI intelligence. Then, he claimed to have a plan to build AI overlords that will naturally support humanity's flourishing. In this mini episode, I react to Elon's remarks and expose why his plan for humanity's survival in the age of AI is dangerously flimsy.
TITLE: Episode 132 — The Only Politician Thinking Clearly About Superintelligence — California Governor Candidate Zoltan Istvan (Feb 13, 2026)
DESCRIPTION: California gubernatorial candidate Zoltan Istvan reveals his P(Doom) and makes the case for universal basic income and radical life extension.
TITLE: Episode 131 — His P(Doom) Is Only 2.6% — AI Doom Debate with Bentham's Bulldog, a.k.a. Matthew Adelstein (Feb 10, 2026)
DESCRIPTION: Get ready for a rematch with the one & only Bentham's Bulldog, a.k.a. Matthew Adelstein! Our first debate covered a wide range of philosophical topics. Today's Debate #2 is all about Matthew's new argument against the inevitability of AI doom. He comes out swinging with a calculated P(Doom) of just 2.6%, based on a multi-step probability chain that I challenge as potentially falling into a "Type 2 Conjunction Fallacy" (a.k.a. Multiple Stage Fallacy). We clash on whether to expect "alignment by default" and the nature of future AI architectures. While Matthew sees current RLHF success as evidence that AIs will likely remain compliant, I argue that we're building "Goal Engines" — superhuman optimization modules that act like nuclear cores wrapped in friendly personalities. We debate whether these engines can be safely contained, or if the capability to map goals to actions is inherently dangerous and prone to exfiltration. Despite our different forecasts (my 50% vs his sub-10%), we actually land in the "sane zone" together on some key policy ideas, like the potential necessity of a global pause. While Matthew's case for low P(Doom) hasn't convinced me, I consider his post and his engagement with me to be super high quality and good faith. We're not here to score points, we just want to better predict how the intelligence explosion will play out.
TITLE: Episode 130 — What Dario Amodei Misses In "The Adolescence of Technology" — Reaction With MIRI's Harlan Stewart (Feb 4, 2026)
DESCRIPTION: Harlan Stewart works in communications for the Machine Intelligence Research Institute (MIRI). In this episode, Harlan and I give our honest opinions on Dario Amodei's new essay "The Adolescence of Technology".
TITLE: Episode 129 — Q&A: Is Liron too DISMISSIVE of AI Harms? + New Studio, Demis Would #PauseAI, AI Water Use Debate (Jan 27, 2026)
DESCRIPTION: Check out the new Doom Debates studio in this Q&A with special guest Producer Ori! Liron gets into a heated discussion about whether doomers must validate short-term risks, like data center water usage, in order to build a successful political coalition. Originally streamed on Saturday, January 24.
TITLE: Episode 128 — Taiwan's Cyber Ambassador-At-Large Says Humans & AI Can FOOM Together (Jan 20, 2026)
DESCRIPTION: Audrey Tang was the youngest minister in Taiwanese history. Now she's working to align AI with democratic principles as Taiwan's Cyber Ambassador. In this debate, I probe her P(doom) and stress-test her vision for safe AI development.</textarea>
  </div>


  <div class="card">
    <span class="muted">Generates 3 outputs per run</span>
    <button class="btn" id="genBtn" onclick="generateDescriptions()" style="margin-left:8px;">Generate Description Candidates</button>
    <button class="btn" type="button" onclick="openLastApiCallWindow()" style="background:#0f3460;color:#fff;margin-left:8px;">View last API prompt</button>
    <span id="status" class="muted" style="margin-left:10px;"></span>
    <div id="usage" class="muted" style="margin-top:10px;">Usage — calls: 0 | input chars: 0 | output chars: 0</div>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Work Log</h3>
    <pre id="logBox" style="min-height:90px;max-height:150px;overflow:auto;white-space:pre-wrap;word-break:break-word;">(no logs yet)</pre>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Generated Output (3 Panes)</h3>
    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;">
      <div><div class="muted" style="margin-bottom:6px;">Gemini (gemini-3.1-pro)</div><pre id="out1" style="min-height:260px;max-height:420px;overflow:auto;white-space:pre-wrap;word-break:break-word;">(gemini output)</pre></div>
      <div><div class="muted" style="margin-bottom:6px;">Claude (claude-opus-4-6)</div><pre id="out2" style="min-height:260px;max-height:420px;overflow:auto;white-space:pre-wrap;word-break:break-word;">(claude output)</pre></div>
      <div><div class="muted" style="margin-bottom:6px;">GPT (gpt-5.4-pro)</div><pre id="out3" style="min-height:260px;max-height:420px;overflow:auto;white-space:pre-wrap;word-break:break-word;">(gpt output)</pre></div>
    </div>
  </div>
</div>

<script>
async function openLastApiCallWindow() {
  const w = window.open('', '_blank');
  if (!w) {
    alert('Popup blocked. Please allow popups for this site.');
    return;
  }
  w.document.write('<!doctype html><html><head><title>Last API Prompt</title><style>body{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0b1224;color:#e8ecf3} .bar{padding:10px 12px;background:#111b35;border-bottom:1px solid #1f2f57;position:sticky;top:0} pre{white-space:pre-wrap;word-break:break-word;margin:0;padding:12px;line-height:1.35;max-width:100%}</style></head><body><div class="bar">Last API call payload (complete)</div><pre>Loading…</pre></body></html>');
  try {
    const r = await fetch('/last_api_call');
    const data = await r.json();
    const text = (data && data.text) ? data.text : 'No API call recorded yet.';
    w.document.querySelector('pre').textContent = text;
  } catch (e) {
    w.document.querySelector('pre').textContent = 'Failed to load last API call: ' + e;
  }
}

function renderUsage(d) {
  const calls = (d && typeof d.desc_calls === 'number') ? d.desc_calls : 0;
  const inChars = (d && typeof d.desc_input_chars === 'number') ? d.desc_input_chars : 0;
  const outChars = (d && typeof d.desc_output_chars === 'number') ? d.desc_output_chars : 0;
  const el = document.getElementById('usage');
  if (el) el.textContent = `Usage — calls: ${calls} | input chars: ${inChars.toLocaleString()} | output chars: ${outChars.toLocaleString()}`;
}

function renderLogs(d) {
  const logs = (d && Array.isArray(d.log)) ? d.log : [];
  const box = document.getElementById('logBox');
  if (!box) return;
  box.textContent = logs.length ? logs.slice(-120).join('\n') : '(no logs yet)';
  box.scrollTop = box.scrollHeight;
}

const transcriptFileInput = document.getElementById('transcriptFile');
if (transcriptFileInput) {
  transcriptFileInput.addEventListener('change', async (e) => {
    const f = (e.target.files || [])[0];
    const status = document.getElementById('transcriptFileStatus');
    if (!f) return;
    const name = (f.name || '').toLowerCase();
    const isTxt = name.endsWith('.txt') || f.type === 'text/plain';
    if (!isTxt) {
      if (status) status.textContent = 'Please attach a .txt file only.';
      e.target.value = '';
      return;
    }
    try {
      const text = await f.text();
      document.getElementById('transcript').value = text;
      if (status) status.textContent = `Loaded: ${f.name} (${text.length.toLocaleString()} chars)`;
    } catch (err) {
      if (status) status.textContent = 'Failed to read transcript file.';
    }
  });
}

function renderOutputPanes(text) {
  const panes = [
    document.getElementById('out1'),
    document.getElementById('out2'),
    document.getElementById('out3'),
  ];
  const raw = (text || '').trim();
  const chunks = raw ? raw.split('\n\n---\n\n') : [];
  for (let i = 0; i < 3; i++) {
    const pane = panes[i];
    if (!pane) continue;
    let chunk = (chunks[i] || '').trim();
    if (!chunk) {
      pane.textContent = `(output ${i + 1})`;
      continue;
    }
    chunk = chunk.replace(/^##\s*Generation\s*\d+\s*/i, '').trim();
    pane.textContent = chunk;
  }
}

async function generateDescriptions() {
  const btn = document.getElementById('genBtn');
  const status = document.getElementById('status');
  const title = document.getElementById('title').value.trim();
  const primary = document.getElementById('primary').value.trim();
  const revise = document.getElementById('revise').value.trim();
  const transcript = document.getElementById('transcript').value.trim();
  const samples = document.getElementById('samples').value.trim();
  const count = 3;
  if (!primary) { alert('Primary description is required.'); return; }
  if (!revise) { alert('How to revise is required.'); return; }
  if (!transcript) { alert('Transcript is required.'); return; }

  btn.disabled = true;
  status.textContent = 'Generating...';
  renderOutputPanes('');

  try {
    const fd = new FormData();
    fd.append('title', title);
    fd.append('primary_description', primary);
    fd.append('revise_instructions', revise);
    fd.append('transcript', transcript);
    fd.append('channel_samples', samples);
    fd.append('count', String(count));

    const resp = await fetch('/generate_descriptions', { method:'POST', body: fd });
    const data = await resp.json();
    if (data.error) {
      renderOutputPanes(data.error);
      status.textContent = 'Failed';
      return;
    }
    renderOutputPanes(data.output || '');
    status.textContent = 'Done';
    try {
      const s = await fetch('/status');
      const sd = await s.json();
      renderUsage(sd);
      renderLogs(sd);
    } catch (_) {}
  } catch (e) {
    renderOutputPanes(String(e));
    status.textContent = 'Failed';
  } finally {
    btn.disabled = false;
  }
}

fetch('/status').then(r => r.json()).then((d) => { renderUsage(d); renderLogs(d); }).catch(() => {});
</script>
</body>
</html>"""

# =====================================================================
# Face Capture — Amazon Rekognition scanning logic
# =====================================================================

from dataclasses import dataclass as _dataclass


@_dataclass
class _FCVideoMeta:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int


@_dataclass
class _FCScoredFrame:
    frame_idx: int
    timestamp: float
    bbox: tuple
    bbox_norm: tuple
    expression_score: float
    quality_score: float
    combined_score: float
    quality_details: dict


def _fc_get_video_meta(path):
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", path
    ], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    vs = next((s for s in data.get("streams", []) if "width" in s), None)
    if not vs:
        raise RuntimeError("No video stream found")
    w, h = int(vs["width"]), int(vs["height"])
    n, d = vs["r_frame_rate"].split("/")
    fps = float(n) / float(d)
    return _FCVideoMeta(path=path, duration=duration, fps=fps,
                        width=w, height=h, total_frames=int(duration * fps))


# Expression scorers — map user-facing names to Rekognition emotion functions
_FC_SCORERS = {}


def _fc_register(name, *aliases):
    def decorator(fn):
        _FC_SCORERS[name] = fn
        for a in aliases:
            _FC_SCORERS[a] = fn
        return fn
    return decorator


def _fc_emo(emotions, name):
    return emotions.get(name, 0.0)


@_fc_register("smile", "happy", "smiling", "laughing", "laugh")
def _fc_smile(emotions, fa=None):
    h = _fc_emo(emotions, "HAPPY")
    if fa:
        return min(1.0, h * 0.6 + fa.get("smile_confidence", 0.0) * 0.4)
    return h

@_fc_register("grimace", "disgust", "grimacing", "cringe")
def _fc_grimace(emotions, fa=None):
    return _fc_emo(emotions, "DISGUSTED")

@_fc_register("surprise", "surprised", "shocked", "shock")
def _fc_surprise(emotions, fa=None):
    return _fc_emo(emotions, "SURPRISED")

@_fc_register("angry", "anger", "frustrated", "mad")
def _fc_angry(emotions, fa=None):
    return _fc_emo(emotions, "ANGRY")

@_fc_register("sad", "sadness", "upset", "frown", "frowning")
def _fc_sad(emotions, fa=None):
    return _fc_emo(emotions, "SAD")

@_fc_register("thinking", "contemplative", "pensive")
def _fc_thinking(emotions, fa=None):
    return min(1.0, max(0.0, 1.0 - _fc_emo(emotions, "CALM") - _fc_emo(emotions, "HAPPY") * 0.3))

@_fc_register("concerned", "worried")
def _fc_concerned(emotions, fa=None):
    return min(1.0, _fc_emo(emotions, "FEAR") * 0.6 + _fc_emo(emotions, "SAD") * 0.4)

@_fc_register("confused", "puzzled")
def _fc_confused(emotions, fa=None):
    return _fc_emo(emotions, "CONFUSED")

@_fc_register("excited", "enthusiastic")
def _fc_excited(emotions, fa=None):
    return min(1.0, _fc_emo(emotions, "HAPPY") * 0.5 + _fc_emo(emotions, "SURPRISED") * 0.5)

@_fc_register("serious", "focused", "stern")
def _fc_serious(emotions, fa=None):
    return min(1.0, max(0.0, _fc_emo(emotions, "CALM") - _fc_emo(emotions, "HAPPY") * 0.5))

@_fc_register("skeptical", "doubtful")
def _fc_skeptical(emotions, fa=None):
    return min(1.0, _fc_emo(emotions, "CONFUSED") * 0.5 + _fc_emo(emotions, "DISGUSTED") * 0.5)

@_fc_register("amused", "entertained")
def _fc_amused(emotions, fa=None):
    h = _fc_emo(emotions, "HAPPY")
    if fa:
        return min(1.0, h * 0.7 + fa.get("smile_confidence", 0.0) * 0.3)
    return h


def _fc_scan_video(meta, expressions, min_face_size, num_samples, log_fn=None):
    """Scan video via Amazon Rekognition Video API."""
    scorers = {}
    for expr in expressions:
        key = expr.lower()
        if key in _FC_SCORERS:
            scorers[expr] = _FC_SCORERS[key]
    if not scorers:
        return {}

    def emit(msg):
        if log_fn:
            log_fn(msg)

    s3 = boto3.client("s3", region_name=FC_AWS_REGION)
    rek = boto3.client("rekognition", region_name=FC_AWS_REGION)

    s3_key = f"uploads/{os.path.basename(meta.path)}"
    file_size_mb = os.path.getsize(meta.path) / (1024 * 1024)
    emit(f"  Uploading video to S3 ({file_size_mb:.0f} MB)...")
    s3.upload_file(meta.path, FC_S3_BUCKET, s3_key)
    emit(f"  Upload complete.")

    try:
        start_resp = rek.start_face_detection(
            Video={"S3Object": {"Bucket": FC_S3_BUCKET, "Name": s3_key}},
            FaceAttributes="ALL",
            NotificationChannel={
                "SNSTopicArn": FC_SNS_TOPIC_ARN,
                "RoleArn": FC_IAM_ROLE_ARN,
            },
        )
        job_id = start_resp["JobId"]
        emit(f"  Rekognition analyzing video (every frame)...")

        import time as _time
        poll_count = 0
        while True:
            poll = rek.get_face_detection(JobId=job_id, MaxResults=1)
            job_status = poll["JobStatus"]
            if job_status == "SUCCEEDED":
                emit(f"  Analysis complete!")
                break
            elif job_status == "FAILED":
                raise RuntimeError(f"Rekognition job failed: {poll.get('StatusMessage', 'Unknown')}")
            poll_count += 1
            if poll_count % 6 == 0:
                emit(f"  Still analyzing... ({poll_count * 5}s elapsed)")
            _time.sleep(5)

        raw_faces = []
        next_token = None
        while True:
            params = {"JobId": job_id, "MaxResults": 1000}
            if next_token:
                params["NextToken"] = next_token
            resp = rek.get_face_detection(**params)
            for entry in resp.get("Faces", []):
                face = entry["Face"]
                ts_sec = entry["Timestamp"] / 1000.0
                emotions = {e["Type"]: e["Confidence"] / 100.0 for e in face.get("Emotions", [])}
                bbox = face["BoundingBox"]
                confidence = face.get("Confidence", 0)
                quality = face.get("Quality", {})
                pose = face.get("Pose", {})
                eyes_open = face.get("EyesOpen", {})
                smile = face.get("Smile", {})
                face_attrs = {
                    "brightness": quality.get("Brightness", 50.0),
                    "sharpness": quality.get("Sharpness", 50.0),
                    "yaw": pose.get("Yaw", 0.0),
                    "pitch": pose.get("Pitch", 0.0),
                    "roll": pose.get("Roll", 0.0),
                    "eyes_open": eyes_open.get("Value", True),
                    "eyes_open_confidence": eyes_open.get("Confidence", 0.0),
                    "smile_value": smile.get("Value", False),
                    "smile_confidence": smile.get("Confidence", 0.0) / 100.0,
                }
                raw_faces.append((ts_sec, emotions, bbox, confidence, face_attrs))
            if "NextToken" in resp:
                next_token = resp["NextToken"]
            else:
                break

        emit(f"  Rekognition found {len(raw_faces)} face detections across entire video")

        results = {expr: [] for expr in scorers}
        filtered_count = 0
        for ts_sec, emotions, bbox_raw, confidence, face_attrs in raw_faces:
            bx_norm = max(0, bbox_raw["Left"])
            by_norm = max(0, bbox_raw["Top"])
            bw_norm = bbox_raw["Width"]
            bh_norm = bbox_raw["Height"]
            if bw_norm < min_face_size:
                filtered_count += 1; continue
            if confidence < 85:
                filtered_count += 1; continue
            if not face_attrs["eyes_open"] and face_attrs["eyes_open_confidence"] > 70:
                filtered_count += 1; continue
            if abs(face_attrs["yaw"]) > 45:
                filtered_count += 1; continue
            if abs(face_attrs["pitch"]) > 30:
                filtered_count += 1; continue
            if abs(face_attrs["roll"]) > 20:
                filtered_count += 1; continue

            x1 = int(bx_norm * meta.width)
            y1 = int(by_norm * meta.height)
            face_w = int(bw_norm * meta.width)
            face_h = int(bh_norm * meta.height)
            frame_idx = int(ts_sec * meta.fps)

            brightness = face_attrs["brightness"] / 100.0
            sharpness = face_attrs["sharpness"] / 100.0
            frontal = 1.0 - min(1.0, (abs(face_attrs["yaw"]) / 45.0 + abs(face_attrs["pitch"]) / 30.0) / 2.0)
            eyes_conf = face_attrs["eyes_open_confidence"] / 100.0 if face_attrs["eyes_open"] else 0.0
            q_details = {
                "brightness": round(brightness, 3),
                "sharpness": round(sharpness, 3),
                "frontal": round(frontal, 3),
                "eyes_open": round(eyes_conf, 3),
            }
            q_score = round(sharpness * 0.35 + brightness * 0.25 + frontal * 0.25 + eyes_conf * 0.15, 4)

            for expr, scorer in scorers.items():
                expr_val = scorer(emotions, face_attrs)
                combined = round(
                    expr_val * 0.40 + sharpness * 0.20 + brightness * 0.15 + frontal * 0.15 + eyes_conf * 0.10, 4)
                results[expr].append(_FCScoredFrame(
                    frame_idx=frame_idx, timestamp=ts_sec,
                    bbox=(x1, y1, face_w, face_h),
                    bbox_norm=(bx_norm, by_norm, bw_norm, bh_norm),
                    expression_score=round(expr_val, 4),
                    quality_score=q_score, combined_score=combined,
                    quality_details=q_details,
                ))

        emit(f"  Filtered {filtered_count} low-quality detections, scored {len(raw_faces) - filtered_count} faces.")
    finally:
        try:
            s3.delete_object(Bucket=FC_S3_BUCKET, Key=s3_key)
            emit(f"  Cleaned up S3 object.")
        except Exception:
            pass

    return results


def _fc_select_top(scored, n, gap=2.0):
    ranked = sorted(scored, key=lambda s: s.combined_score, reverse=True)
    selected = []
    for s in ranked:
        if not any(abs(s.timestamp - sel.timestamp) < gap for sel in selected):
            selected.append(s)
        if len(selected) >= n:
            break
    return selected


def _fc_fmt_time(sec):
    h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fc_fmt_time_hms(sec):
    h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _fc_save_results(selected, output_dir, meta, expression, elapsed):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(meta.path)
    entries = []
    for rank, sf in enumerate(selected, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf.frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fh, fw = frame.shape[:2]
        bx, by, bw, bh = sf.bbox
        fname = f"face_{rank:03d}_score{sf.combined_score:.2f}_{_fc_fmt_time_hms(sf.timestamp)}.png"
        cv2.imwrite(os.path.join(output_dir, fname), frame)
        pad_x, pad_y = int(bw * 0.3), int(bh * 0.3)
        crop = frame[max(0, by - pad_y):min(fh, by + bh + pad_y),
                      max(0, bx - pad_x):min(fw, bx + bw + pad_x)]
        cname = f"face_{rank:03d}_crop.png"
        cv2.imwrite(os.path.join(output_dir, cname), crop)
        entries.append({
            "rank": rank, "file": fname, "crop_file": cname,
            "timestamp_seconds": round(sf.timestamp, 2),
            "timestamp_display": _fc_fmt_time(sf.timestamp),
            "expression_score": sf.expression_score,
            "quality_score": sf.quality_score,
            "combined_score": sf.combined_score,
            "quality_details": sf.quality_details,
            "face_bbox": {"x": round(sf.bbox_norm[0], 3), "y": round(sf.bbox_norm[1], 3),
                          "w": round(sf.bbox_norm[2], 3), "h": round(sf.bbox_norm[3], 3)},
        })
    cap.release()
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({
            "video": os.path.basename(meta.path),
            "expression": expression,
            "video_duration_seconds": round(meta.duration, 2),
            "video_resolution": f"{meta.width}x{meta.height}",
            "processing_time_seconds": round(elapsed, 1),
            "results": entries,
        }, f, indent=2)


def _fc_log(msg):
    with status_lock:
        fc_status["log"].append(msg)


def _fc_run_capture(params):
    """Background thread for face capture scanning."""
    import time as _time
    try:
        video = params.get("video", "")
        expressions = [e.strip() for e in params.get("expressions", "smile").split(",") if e.strip()]
        output_base = params.get("output", FC_CAPTURES_DIR)
        count = int(params.get("count", 10))

        now = datetime.datetime.now()
        folder_name = now.strftime("%b%d-%H%M")
        output_dir = os.path.join(output_base, folder_name)
        suffix = 2
        while os.path.exists(output_dir):
            output_dir = os.path.join(output_base, f"{folder_name}-{suffix}")
            suffix += 1

        if not os.path.isfile(video):
            _fc_log(f"ERROR: File not found: {video}")
            return

        t0 = _time.time()
        _fc_log(f"Analyzing: {os.path.basename(video)}")
        meta = _fc_get_video_meta(video)
        _fc_log(f"  {_fc_fmt_time(meta.duration)} | {meta.width}x{meta.height} @ {meta.fps:.1f}fps")
        _fc_log(f"")
        _fc_log(f"Scanning for: {', '.join(expressions)} (via Amazon Rekognition)")

        all_scored = _fc_scan_video(meta, expressions, 0.10, 500, log_fn=_fc_log)

        frames_found = max(len(v) for v in all_scored.values()) if all_scored else 0
        _fc_log(f"  Found {frames_found} frames with faces")

        if not frames_found:
            _fc_log("")
            _fc_log("No faces found. Try a different video.")
            return

        multi = len(expressions) > 1
        result_dirs = {}
        for expr in expressions:
            scored = all_scored.get(expr, [])
            if not scored:
                _fc_log(f"\n[{expr}] No faces found.")
                continue
            selected = _fc_select_top(scored, count)
            out_dir = os.path.join(output_dir, expr) if multi else output_dir
            elapsed = _time.time() - t0
            os.makedirs(out_dir, exist_ok=True)
            _fc_save_results(selected, out_dir, meta, expr, elapsed)
            result_dirs[expr] = out_dir
            _fc_log(f"")
            _fc_log(f"[{expr}] Saved {len(selected)} screenshots")
            for i, s in enumerate(selected, 1):
                _fc_log(f"  #{i}: score={s.combined_score:.2f} "
                        f"(expr={s.expression_score:.2f}, qual={s.quality_score:.2f}) "
                        f"@ {_fc_fmt_time(s.timestamp)}")

        elapsed = _time.time() - t0
        _fc_log(f"")
        _fc_log(f"Done in {elapsed:.1f}s!")

        with status_lock:
            fc_status["output_dir"] = output_dir
            fc_status["result_dirs"] = result_dirs
    except Exception as e:
        _fc_log(f"ERROR: {e}")
    finally:
        with status_lock:
            fc_status["running"] = False
            fc_status["done"] = True


# =====================================================================
# Face Capture — HTML Template
# =====================================================================

HTML_FACE_CAPTURE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Doom Debates — Face Capture</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #1a1a2e; color: #e0e0e0; padding: 32px;
    min-height: 100vh;
  }
  h1 { color: #fff; font-size: 28px; margin-bottom: 4px; }
  .subtitle { color: #888; font-size: 14px; margin-bottom: 8px; }
  .nav { margin-bottom: 16px; }
  .nav a { color: #4ade80; text-decoration: none; font-weight: 600; margin-right: 18px; }
  .nav a:hover { text-decoration: underline; }
  .card {
    background: #16213e; border-radius: 12px; padding: 24px;
    margin-bottom: 16px; border: 1px solid #0f3460;
  }
  label.section { display: block; font-size: 13px; color: #a0a0b0; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  input[type="text"] {
    width: 100%; padding: 10px 14px; border-radius: 8px;
    border: 1px solid #0f3460; background: #0d1b3e; color: #fff;
    font-size: 15px; outline: none;
  }
  input[type="text"]:focus { border-color: #e94560; }
  .drop-zone {
    position: relative; border: 2px dashed #0f3460; border-radius: 10px;
    padding: 4px; transition: all 0.2s;
  }
  .drop-zone.dragover {
    border-color: #e94560; background: rgba(233,69,96,0.08);
  }
  .drop-zone.dragover::after {
    content: "Drop video file here"; position: absolute;
    inset: 0; display: flex; align-items: center; justify-content: center;
    background: rgba(13,27,62,0.9); border-radius: 8px;
    color: #e94560; font-weight: 600; font-size: 15px; pointer-events: none;
  }
  .drop-zone input[type="text"] { border: none; }
  .drop-hint { font-size: 11px; color: #666; margin-top: 4px; }
  input[type="number"] {
    width: 80px; padding: 10px 14px; border-radius: 8px;
    border: 1px solid #0f3460; background: #0d1b3e; color: #fff;
    font-size: 15px; outline: none; text-align: center;
  }
  .check-group { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .check-group label {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 7px 14px; border-radius: 20px;
    border: 1px solid #0f3460; cursor: pointer; font-size: 14px;
    color: #a0a0b0; transition: all 0.15s; user-select: none;
  }
  .check-group label:hover { border-color: #e94560; color: #fff; }
  .check-group input[type="checkbox"] { display: none; }
  .check-group input[type="checkbox"]:checked + span { color: #fff; }
  .check-group label:has(input:checked) {
    background: #e94560; border-color: #e94560; color: #fff;
  }
  .check-group label.select-all {
    background: transparent; border-color: #555; font-size: 12px;
    padding: 5px 12px; color: #888;
  }
  .check-group label.select-all:hover { border-color: #aaa; color: #ccc; }
  .divider { width: 1px; height: 24px; background: #0f3460; margin: 0 4px; }
  .row { display: flex; gap: 24px; align-items: flex-end; }
  .row > div { flex: 1; }
  button.run {
    padding: 12px 32px; background: #e94560; color: #fff; border: none;
    border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer;
    transition: background 0.15s;
  }
  button.run:hover { background: #c73a52; }
  button.run:disabled { background: #555; cursor: not-allowed; }
  button.open {
    padding: 10px 24px; background: #0f3460; color: #fff; border: 1px solid #1a4080;
    border-radius: 8px; font-size: 14px; cursor: pointer; margin-left: 12px;
  }
  button.open:hover { background: #1a4080; }
  .btn-row { display: flex; align-items: center; margin-top: 8px; }
  #log {
    background: #0d1b3e; border-radius: 8px; padding: 16px;
    font-family: "Menlo", "Monaco", monospace; font-size: 13px;
    line-height: 1.6; min-height: 120px; max-height: 250px;
    overflow-y: auto; white-space: pre-wrap; color: #b0b0c0;
    border: 1px solid #0f3460;
  }
  #log .hl { color: #e94560; font-weight: 600; }
  #log .expr-header { color: #53c0f0; font-weight: 700; font-size: 14px; }
  .results-section { margin-top: 16px; }
  .results-section h3 { color: #53c0f0; font-size: 15px; margin: 12px 0 8px 0; text-transform: capitalize; }
  .results-row { display: flex; flex-wrap: wrap; gap: 10px; }
  .results-row img {
    height: 130px; border-radius: 8px; border: 2px solid #0f3460;
    cursor: pointer; transition: border-color 0.15s;
  }
  .results-row img:hover { border-color: #e94560; }
  .thumb { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .thumb-ts { font-size: 11px; color: #888; font-family: "Menlo", monospace; }
  .spinner {
    display: inline-block; width: 18px; height: 18px;
    border: 2px solid #555; border-top-color: #e94560;
    border-radius: 50%; animation: spin 0.8s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<h1>Doom Debates &mdash; Face Capture</h1>
<div class="subtitle">Amazon Rekognition &mdash; Expression detection from video</div>
<div class="nav">
  <a href="/">&larr; Thumbnail Generator</a>
  <a href="/revision">Revision</a>
  <a href="/descriptions">Descriptions</a>
</div>

<div class="card">
  <label class="section">Video File Path</label>
  <div class="drop-zone" id="dropZone">
    <input type="text" id="video" placeholder="/path/to/video.mp4 — or drag & drop a file here">
  </div>
  <div class="drop-hint">Drag a video file from Finder to fill in the path</div>
</div>

<div class="card">
  <label class="section">Expressions (select one or more)</label>
  <div class="check-group" id="exprGroup">
    <label class="select-all" onclick="toggleAll(event)"><span>Select All</span></label>
    <div class="divider"></div>
  </div>
</div>

<div class="card">
  <div class="row">
    <div>
      <label class="section">Output Folder</label>
      <input type="text" id="output" value="FC_OUTPUT_PLACEHOLDER">
    </div>
    <div style="flex:0 0 150px">
      <label class="section">Per Expression</label>
      <input type="number" id="count" value="10" min="1" max="100">
    </div>
  </div>
</div>

<div class="card">
  <div class="btn-row">
    <button class="run" id="runBtn" onclick="run()">Find Expressions</button>
    <button class="open" id="openBtn" onclick="openFolder()" style="display:none">Open Output Folder</button>
    <span id="spinnerSpan" style="display:none; margin-left: 12px;">
      <span class="spinner"></span> Processing...
    </span>
  </div>
</div>

<div class="card">
  <label class="section">Status</label>
  <div id="log">Ready. Select expressions and click "Find Expressions".</div>
  <div id="results" class="results-section"></div>
</div>

<script>
const EXPRESSIONS = FC_EXPR_LIST_PLACEHOLDER;

const group = document.getElementById("exprGroup");
EXPRESSIONS.forEach(expr => {
  const label = document.createElement("label");
  label.innerHTML = '<input type="checkbox" value="' + expr + '"><span>' + expr.charAt(0).toUpperCase() + expr.slice(1) + '</span>';
  group.appendChild(label);
});
const first = group.querySelector('input[value="smile"]');
if (first) first.checked = true;

function toggleAll(e) {
  e.preventDefault();
  const boxes = document.querySelectorAll('#exprGroup input[type="checkbox"]');
  const allChecked = Array.from(boxes).every(b => b.checked);
  boxes.forEach(b => b.checked = !allChecked);
}

function getExprs() {
  return Array.from(document.querySelectorAll('#exprGroup input:checked')).map(i => i.value);
}

function esc(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

async function run() {
  const video = document.getElementById("video").value.trim();
  const exprs = getExprs();
  if (!video) { alert("Enter a video file path."); return; }
  if (exprs.length === 0) { alert("Select at least one expression."); return; }

  document.getElementById("runBtn").disabled = true;
  document.getElementById("spinnerSpan").style.display = "inline";
  document.getElementById("openBtn").style.display = "none";
  document.getElementById("results").innerHTML = "";
  document.getElementById("log").textContent = "";

  const params = new URLSearchParams({
    video, output: document.getElementById("output").value.trim(),
    count: document.getElementById("count").value,
    expressions: exprs.join(","),
  });

  await fetch("/fc_run?" + params.toString());

  const poll = setInterval(async () => {
    const resp = await fetch("/fc_status");
    const data = await resp.json();
    document.getElementById("log").innerHTML = data.log.map(l => {
      if (l.startsWith("  #")) return '<span class="hl">' + esc(l) + '</span>';
      if (l.startsWith("[") && l.includes("]")) return '<span class="expr-header">' + esc(l) + '</span>';
      return esc(l);
    }).join("\n");
    document.getElementById("log").scrollTop = 999999;

    if (data.done) {
      clearInterval(poll);
      document.getElementById("runBtn").disabled = false;
      document.getElementById("spinnerSpan").style.display = "none";
      if (data.output_dir) {
        lastOutputDir = data.output_dir;
        document.getElementById("openBtn").style.display = "inline-block";
        showResults(data.result_dirs);
      }
    }
  }, 500);
}

async function showResults(resultDirs) {
  const container = document.getElementById("results");
  container.innerHTML = "";
  for (const [expr, dir] of Object.entries(resultDirs)) {
    try {
      const resp = await fetch("/fc_list_results?dir=" + encodeURIComponent(dir));
      const items = await resp.json();
      if (items.length === 0) continue;
      const h3 = document.createElement("h3");
      h3.textContent = expr;
      container.appendChild(h3);
      const row = document.createElement("div");
      row.className = "results-row";
      for (const item of items) {
        const thumb = document.createElement("div");
        thumb.className = "thumb";
        const img = document.createElement("img");
        img.src = "/fc_image?path=" + encodeURIComponent(item.crop_path);
        img.title = item.crop_path.split("/").pop();
        if (item.full_path) {
          img.addEventListener("click", () => {
            fetch("/fc_open_image?path=" + encodeURIComponent(item.full_path));
          });
        }
        thumb.appendChild(img);
        if (item.timestamp) {
          const ts = document.createElement("div");
          ts.className = "thumb-ts";
          ts.textContent = item.timestamp;
          thumb.appendChild(ts);
        }
        row.appendChild(thumb);
      }
      container.appendChild(row);
    } catch(e) {}
  }
}

let lastOutputDir = "";
function openFolder() {
  const dir = lastOutputDir || document.getElementById("output").value;
  fetch("/fc_open_folder?dir=" + encodeURIComponent(dir));
}

// Drag & drop support
const dropZone = document.getElementById("dropZone");
const videoInput = document.getElementById("video");

["dragenter", "dragover"].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", async e => {
  let uri = e.dataTransfer.getData("text/uri-list");
  if (uri) {
    const line = uri.split(/\r?\n/).find(l => l && !l.startsWith("#"));
    if (line) {
      let path = line.trim();
      if (path.startsWith("file://")) path = decodeURIComponent(path.slice(7));
      videoInput.value = path;
      return;
    }
  }
  let plain = e.dataTransfer.getData("text/plain");
  if (plain) {
    plain = plain.trim();
    if (plain.startsWith("file://")) { videoInput.value = decodeURIComponent(plain.slice(7)); return; }
    if (plain.startsWith("/")) { videoInput.value = plain; return; }
  }
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    const file = e.dataTransfer.files[0];
    videoInput.value = "Resolving path\u2026";
    try {
      const resp = await fetch("/fc_resolve_file?name=" + encodeURIComponent(file.name) + "&size=" + file.size);
      const data = await resp.json();
      if (data.path) {
        videoInput.value = data.path;
      } else {
        videoInput.value = "";
        alert('Could not resolve full path for "' + file.name + '". Please paste the path manually.');
      }
    } catch(err) { videoInput.value = ""; }
  }
});

document.addEventListener("dragover", e => e.preventDefault());
document.addEventListener("drop", e => e.preventDefault());
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
            if APP_MODE == "descriptions":
                self._serve_descriptions_html()
            else:
                self._serve_html()
        elif path == "/revision":
            self._serve_revision_html()
        elif path == "/descriptions":
            self._serve_descriptions_html()
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
                        "desc_calls": status.get("desc_calls", 0),
                        "desc_input_chars": status.get("desc_input_chars", 0),
                        "desc_output_chars": status.get("desc_output_chars", 0),
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
                    "desc_calls": 0, "desc_input_chars": 0, "desc_output_chars": 0,
                }
            self._json_response(safe)
        elif path == "/last_api_call":
            with status_lock:
                payload = status.get("last_api_call", "")
            self._json_response({"ok": True, "text": payload})
        elif path == "/last_border_api_call":
            with status_lock:
                payload = status.get("last_border_api_call", "")
            self._json_response({"ok": True, "text": payload})
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
        # ----- Face Capture routes -----
        elif path == "/face-capture":
            self._serve_face_capture_html()
        elif path == "/fc_status":
            with status_lock:
                fc_safe = {
                    "running": fc_status["running"],
                    "log": list(fc_status["log"]),
                    "done": fc_status["done"],
                    "output_dir": fc_status["output_dir"],
                    "result_dirs": dict(fc_status["result_dirs"]),
                }
            self._json_response(fc_safe)
        elif path == "/fc_run":
            with status_lock:
                if not fc_status["running"]:
                    fc_status["running"] = True
                    fc_status["done"] = False
                    fc_status["log"] = []
                    fc_status["output_dir"] = ""
                    fc_status["result_dirs"] = {}
                    t = threading.Thread(target=_fc_run_capture, args=(params,), daemon=True)
                    t.start()
            self._json_response({"ok": True})
        elif path == "/fc_list_results":
            d = params.get("dir", "")
            items = []
            if os.path.isdir(d):
                meta_path = os.path.join(d, "metadata.json")
                if os.path.isfile(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    for entry in meta.get("results", []):
                        crop_path = os.path.join(d, entry["crop_file"])
                        full_path = os.path.join(d, entry["file"])
                        if os.path.isfile(crop_path):
                            items.append({
                                "crop_path": crop_path,
                                "full_path": full_path,
                                "timestamp": entry.get("timestamp_display", ""),
                                "score": entry.get("combined_score", 0),
                            })
                else:
                    for f_name in sorted(os.listdir(d)):
                        if f_name.endswith("_crop.png"):
                            items.append({"crop_path": os.path.join(d, f_name), "full_path": "", "timestamp": "", "score": 0})
            self._json_response(items)
        elif path == "/fc_image":
            img_path = params.get("path", "")
            if os.path.isfile(img_path) and img_path.endswith(".png"):
                with open(img_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
        elif path == "/fc_resolve_file":
            name = params.get("name", "")
            size = int(params.get("size", "0"))
            resolved = ""
            if name:
                home = os.path.expanduser("~")
                try:
                    result = subprocess.run(
                        ["find", home, "-maxdepth", "6", "-name", name, "-type", "f"],
                        capture_output=True, text=True, timeout=10)
                    candidates = [c for c in result.stdout.strip().split("\n") if c and os.path.isfile(c)]
                    if size:
                        for c in candidates:
                            try:
                                if os.path.getsize(c) == size:
                                    resolved = c; break
                            except OSError:
                                pass
                    if not resolved and candidates:
                        resolved = candidates[0]
                except Exception:
                    pass
            self._json_response({"path": resolved})
        elif path == "/fc_open_image":
            img_path = params.get("path", "")
            if os.path.isfile(img_path) and img_path.endswith(".png"):
                os.system(f'open "{img_path}"')
            self._json_response({"ok": True})
        elif path == "/fc_open_folder":
            d = params.get("dir", "")
            if os.path.isdir(d):
                os.system(f'open "{d}"')
            self._json_response({"ok": True})
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
            elif path == "/generate_descriptions":
                self._handle_generate_descriptions(body)
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
        self.wfile.write(HTML_REVISION.replace("__GIT_VERSION__", GIT_VERSION).encode("utf-8"))

    def _serve_descriptions_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_DESCRIPTIONS.encode("utf-8"))

    def _serve_face_capture_html(self):
        html = HTML_FACE_CAPTURE.replace(
            "FC_EXPR_LIST_PLACEHOLDER", json.dumps(FC_CORE_EXPRESSIONS)
        ).replace(
            "FC_OUTPUT_PLACEHOLDER", FC_CAPTURES_DIR
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

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

    def _handle_generate_descriptions(self, body):
        """Generate iterated YouTube descriptions from transcript and guidance."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            fields, _ = parse_multipart(self.headers, body)
        else:
            fields = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="replace")))

        title = fields.get("title", "").strip()
        primary_description = fields.get("primary_description", "").strip()
        revise_instructions = fields.get("revise_instructions", "").strip()
        transcript = fields.get("transcript", "").strip()
        channel_samples = fields.get("channel_samples", "").strip()

        if not primary_description:
            self._json_response({"error": "Primary description is required"})
            return
        if not revise_instructions:
            self._json_response({"error": "How to revise is required"})
            return
        if not transcript:
            self._json_response({"error": "Transcript is required"})
            return

        try:
            client = get_client()
            in_chars = len(title) + len(primary_description) + len(revise_instructions) + len(transcript) + len(channel_samples)
            prompt = _build_description_prompt(title, primary_description, revise_instructions, transcript, channel_samples)
            outputs = []
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

            for name, fn in providers:
                with status_lock:
                    status["log"].append(f"{name} generation started")
                try:
                    output = fn()
                except Exception as e:
                    output = f"[{name} error] {str(e)[:260]}"
                outputs.append(output)
                out_chars = len(output or "")
                with status_lock:
                    status["desc_calls"] = status.get("desc_calls", 0) + 1
                    status["desc_input_chars"] = status.get("desc_input_chars", 0) + in_chars
                    status["desc_output_chars"] = status.get("desc_output_chars", 0) + out_chars
                    status["log"].append(f"{name} generation done (in={in_chars} chars, out={out_chars} chars)")

            with status_lock:
                status["running"] = False
                status["phase"] = "idle"
                status["done"] = True

            self._json_response({"ok": True, "output": "\n\n---\n\n".join(outputs)})
        except Exception as e:
            with status_lock:
                status["running"] = False
                status["phase"] = "idle"
                status["log"].append(f"Description generation failed: {str(e)[:180]}")
            self._json_response({"error": f"Description generation failed: {str(e)[:300]}"})

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
        status["add_border"] = fields.get("add_border") == "1"
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

        status["add_border"] = fields.get("add_border") == "1"
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
        try:
            count = int(fields.get("count", "10"))
        except Exception:
            count = 10
        count = max(1, min(50, count))

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
            count_per=count,
            idea_idx=revision_idea_idx,
            attachment_refs=attachment_refs if attachment_refs else None,
        )

        status["add_border"] = fields.get("add_border") == "1"
        run_generation(client, prompts, round_dir, "revision_page")
        with status_lock:
            base_src = "uploaded file" if base_files else "follow-up result"
            status["log"].append(f"Revision base: {base_src}")
            status["log"].append(f"Prompt: {prompt[:180]}")
            status["log"].append(f"Extra refs attached: {len(files.get('revision_images', []))}")
            status["log"].append(f"Requested attempts: {count}")

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

        status["add_border"] = fields.get("add_border") == "1"
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
    os.makedirs(FC_CAPTURES_DIR, exist_ok=True)

    client = get_client()
    upload_brand_references(client)
    upload_liron_references(client)
    upload_border_reference(client)
    print(f"Doom Debates Thumbnail Generator v2")
    print(f"Image Model: {GEMINI_MODEL}")
    print(f"Text Model: {TEXT_MODEL}")
    print(f"Description Model (Gemini): {DESCRIPTION_MODEL}")
    print(f"Description Model (Claude): {CLAUDE_DESCRIPTION_MODEL} {'[enabled]' if ANTHROPIC_API_KEY else '[disabled: no ANTHROPIC_API_KEY]'}")
    print(f"Description Model (GPT): {GPT_DESCRIPTION_MODEL} {'[enabled]' if OPENAI_API_KEY else '[disabled: no OPENAI_API_KEY]'}")
    print(f"Output: {THUMBNAILS_DIR}")
    print(f"Brand Refs: {len(BRAND_FILES)} images from {EXAMPLES_DIR}")
    print(f"Liron Refs: {len(LIRON_FILES)} images from {LIRON_DIR}")
    print(f"Border Ref: {'loaded' if BORDER_REF_FILE else 'not found (border pass disabled)'}")
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
