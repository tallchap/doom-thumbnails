"""Async thumbnail generation via Gemini image API."""

import asyncio
import datetime
import io
import json
import os
import random
import threading

from google.genai import types
from PIL import Image

from config import (
    GEMINI_MODEL, TEXT_MODEL, THUMBNAILS_DIR, MAX_CONCURRENT, COST_PER_IMAGE,
    MAX_BRAND_REFS_PER_CALL, MAX_SPEAKER_REFS_PER_CALL, MAX_LIRON_REFS_PER_CALL,
)
from shared.gemini_client import BRAND_FILES, LIRON_FILES, BORDER_REF_FILES
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


def _select_identity_refs(refs, max_refs):
    """Choose a small, deterministic subset of identity refs."""
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

            brand_sample = _select_brand_refs()
            if brand_sample:
                contents.append("=== DOOM DEBATES BRAND STYLE ONLY — match colors, layout, typography, energy. WARNING: These images contain people — COMPLETELY IGNORE all faces/people in these images. Do NOT reproduce any human likeness from these references. ===")
                contents.extend(brand_sample)

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


def build_revision_prompts(selected_images, speaker_refs, custom_prompt, count_per=3, idea_idx=-1, attachment_refs=None, context_prompt=None):
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
            contents = [prompt_text, img]
            if selected_liron_refs:
                contents.append(
                    "In this image, you must make one of the faces look like Liron Shapira. "
                    "Liron Shapira looks like the following enclosed images:"
                )
                contents.extend(selected_liron_refs)
                contents.append(
                    "The face of Liron in your output MUST be a faithful reproduction of the person in the above photos — "
                    "same facial structure, same nose, same eyes, same beard shape, same skin tone. "
                    "Do NOT generate a generic man's face. Do NOT invent features. Copy Liron's exact likeness."
                )
            contents.append(context_prompt or REVISION_CONTEXT_PROMPT)
            brand_sample = _select_brand_refs()
            if brand_sample:
                contents.extend(brand_sample)
            if attachment_refs:
                contents.append("The user has attached the following reference image(s) — use them to guide the revision:")
                contents.extend(attachment_refs)
            prompts.append((idea_idx, v, contents))
    return prompts


# ----- Async Generation -----


async def apply_border_pass(client, img_data, prompt_override=None, target_status=None, target_lock=None):
    """Second Gemini pass: add red border + DOOM DEBATES wordmark."""
    _st = target_status if target_status is not None else main_status
    _lk = target_lock if target_lock is not None else main_status_lock
    if not BORDER_REF_FILES:
        return None
    try:
        prompt_text = prompt_override or BORDER_PASS_PROMPT
        source_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        border_contents = list(BORDER_REF_FILES) + [source_img, prompt_text]
        _record_api_call(GEMINI_MODEL, border_contents, phase="border_pass", key="last_border_api_call", target_status=_st, target_lock=_lk)
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
        with _lk:
            _st["log"].append(f"Border pass: no image in response")
    except Exception as e:
        with _lk:
            _st["log"].append(f"Border pass error: {str(e)[:150]}")
        print(f"Border pass failed: {e}")
    return None


async def generate_batch(client, prompts, output_dir, phase="round1", target_status=None, target_lock=None):
    """Fire parallel Gemini API calls and save results.
    prompts is a list of (idea_idx, variation_num, contents)."""
    _st = target_status if target_status is not None else main_status
    _lk = target_lock if target_lock is not None else main_status_lock
    os.makedirs(output_dir, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    with _lk:
        start_idx = len(_st["images"])

    async def generate_one(idx, idea_idx, contents):
        async with sem:
            with _lk:
                if _st.get("cancel_requested"):
                    _st["log"].append(f"thumb_{idx:03d}: cancelled")
                    return None
            for attempt in range(3):
                try:
                    _record_api_call(GEMINI_MODEL, contents, phase=phase, target_status=_st, target_lock=_lk)
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

                                if _st.get("add_border") and BORDER_REF_FILES:
                                    with _lk:
                                        _st["log"].append(f"thumb_{idx:03d}: applying border pass...")
                                    border_result = await apply_border_pass(client, img_data, _st.get("border_prompt"), target_status=_st, target_lock=_lk)
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
                                        f"[{_st['completed']}/{_st['total']}] thumb_{idx:03d}.png OK"
                                    )
                                return filepath

                    with _lk:
                        _st["errors"] += 1
                        _st["log"].append(f"thumb_{idx:03d}: no image in response")
                    return None

                except asyncio.TimeoutError:
                    with _lk:
                        _st["errors"] += 1
                        _st["log"].append(f"thumb_{idx:03d}: timed out after 120s")
                    return None

                except Exception as e:
                    err = str(e)
                    if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < 2:
                        wait = (2 ** attempt) + random.random()
                        with _lk:
                            if _st.get("cancel_requested"):
                                _st["log"].append(f"thumb_{idx:03d}: cancelled")
                                return None
                            _st["log"].append(
                                f"thumb_{idx:03d}: rate limited, retrying in {wait:.1f}s..."
                            )
                        await asyncio.sleep(wait)
                        continue
                    with _lk:
                        _st["errors"] += 1
                        _st["log"].append(f"thumb_{idx:03d}: ERROR — {err[:120]}")
                    return None

    tasks = [
        generate_one(start_idx + i + 1, idea_idx, contents)
        for i, (idea_idx, _var, contents) in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]


def run_generation(client, prompts, output_dir, phase="round1", target_status=None, target_lock=None):
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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                generate_batch(client, prompts, output_dir, phase, target_status=_st, target_lock=_lk)
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
