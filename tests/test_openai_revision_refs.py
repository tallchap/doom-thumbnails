"""Regression guard: the OpenAI (gpt-image-2) revision path must send ONLY the user's base
image and the user's own attachments — never a published episode thumbnail.

Bug this locks down: `_openai_brand_ref_bytes()` used to staple
`sorted(os.listdir(doom_debates_thumbnails/))[:3]` onto every `images.edit` call. Those three
files are always the same, and their pixels carry baked-in headlines ("ARE WE COOKED?",
"OPENAI IS TOO BIG TO FAIL?", "GLOBAL WARMING SOLVED!" + an SO2 balloon). images.edit
composites every image it is handed, so gpt-image transcribed them into unrelated episodes.
"""

import asyncio
import io
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thumbnails.generator import (  # noqa: E402
    build_openai_revision_prompt,
    run_openai_revision,
)
from thumbnails.prompts import (  # noqa: E402
    OPENAI_BRAND_STYLE_TEXT,
    OPENAI_NO_INVENTED_CONTENT_RULE,
    OPENAI_REVISION_CONTEXT_PROMPT,
    REVISION_CONTEXT_PROMPT,
)


PNG_1PX = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00"
    b"\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _FakeResponse:
    def __init__(self):
        # 1x1 transparent PNG, base64
        self.data = [type("D", (), {"b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="})()]


class _CapturingImages:
    def __init__(self, sink):
        self._sink = sink

    async def edit(self, **kwargs):
        self._sink.append(kwargs)
        return _FakeResponse()


class _FakeAsyncOpenAI:
    """Stand-in for openai.AsyncOpenAI that records every images.edit payload."""

    calls = []

    def __init__(self, *a, **kw):
        self.images = _CapturingImages(_FakeAsyncOpenAI.calls)

    async def close(self):
        return None


@pytest.fixture
def captured_calls(monkeypatch):
    import openai

    _FakeAsyncOpenAI.calls = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr("config.OPENAI_API_KEY", "test-key", raising=False)
    return _FakeAsyncOpenAI.calls


def _run_and_wait(tmp_path, attachments, count=2):
    status = {}
    lock = threading.Lock()
    run_openai_revision(
        [PNG_1PX], "make the sky bluer", count, str(tmp_path),
        attachment_bytes_list=attachments,
        target_status=status, target_lock=lock,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        with lock:
            if status.get("done"):
                break
        time.sleep(0.05)
    with lock:
        assert status.get("done"), f"generation never finished; log={status.get('log')}"
    return status


# ----- image manifest -----


def test_no_brand_reference_images_are_sent(captured_calls, tmp_path):
    """Base image only — nothing from doom_debates_thumbnails/ reaches images.edit."""
    _run_and_wait(tmp_path, attachments=None, count=3)

    assert len(captured_calls) == 3
    for kwargs in captured_calls:
        images = kwargs["image"]
        assert len(images) == 1, f"expected base image only, got {[i[0] for i in images]}"
        assert images[0][0] == "base_1.png"
        assert kwargs["model"] == "gpt-image-2"
        assert kwargs["size"] == "1280x720"


def test_user_attachments_are_sent_and_brand_refs_are_not(captured_calls, tmp_path):
    _run_and_wait(tmp_path, attachments=[PNG_1PX, PNG_1PX], count=2)

    assert len(captured_calls) == 2
    for kwargs in captured_calls:
        names = [i[0] for i in kwargs["image"]]
        assert names == ["base_1.png", "user_ref_1.png", "user_ref_2.png"]


def test_no_filename_resolves_into_the_brand_corpus(captured_calls, tmp_path):
    """Hard guard: the published-thumbnail dir must never appear in a payload filename."""
    from config import EXAMPLES_DIR

    brand_names = {
        n for n in os.listdir(EXAMPLES_DIR)
        if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    }
    assert brand_names, "brand corpus is empty — this test would pass vacuously"

    _run_and_wait(tmp_path, attachments=[PNG_1PX], count=1)

    for kwargs in captured_calls:
        for name, _stream, *_rest in kwargs["image"]:
            assert name not in brand_names
            assert not os.path.exists(os.path.join(EXAMPLES_DIR, name))


def test_manifest_and_prompt_are_logged(captured_calls, tmp_path):
    """The /logs tab must show exactly what was sent — no sampling, no filtering."""
    status = _run_and_wait(tmp_path, attachments=[PNG_1PX], count=1)
    log = "\n".join(status["log"])

    assert "Brand reference images attached: 0 (text style guide only)" in log
    assert "base_1.png" in log and "user_ref_1.png" in log
    assert "Prompt sent to OpenAI:" in log
    assert "NOTHING-NEW RULE" in log

    api = status.get("last_api_call", "")
    assert "BRAND REFERENCE IMAGES: none" in api
    assert "base_1.png" in api


# ----- prompt text -----


def test_prompt_has_no_brand_reference_legend():
    prompt = build_openai_revision_prompt("bluer sky", 1, 5, attachment_count=0)

    assert "brand-style references" not in prompt
    assert "Image 1 is the base thumbnail to revise" in prompt
    # With no attachments there is no Image 2 to describe.
    assert "Image 2" not in prompt
    assert OPENAI_BRAND_STYLE_TEXT in prompt
    assert OPENAI_NO_INVENTED_CONTENT_RULE in prompt


def test_attachment_legend_starts_at_image_two():
    one = build_openai_revision_prompt("x", 1, 1, attachment_count=1)
    assert "Image 2 is user-attached reference images" in one

    three = build_openai_revision_prompt("x", 1, 1, attachment_count=3)
    assert "Images 2-4 are user-attached reference images" in three


def test_default_context_does_not_reference_enclosed_images():
    """The Gemini context prompt talks about style images that this path no longer sends."""
    assert "enclosed as separate images" in REVISION_CONTEXT_PROMPT
    assert "enclosed as separate images" not in OPENAI_REVISION_CONTEXT_PROMPT

    prompt = build_openai_revision_prompt("x", 1, 1)
    assert "enclosed as separate images" not in prompt
    assert "COMPLETELY IGNORE all faces" not in prompt
    # The quoted-text fidelity rule must survive the fork.
    assert "TEXT FIDELITY (CRITICAL)" in prompt


def test_custom_context_prompt_still_wins():
    prompt = build_openai_revision_prompt("x", 1, 1, context_prompt="MY OWN CONTEXT")
    assert "MY OWN CONTEXT" in prompt
    assert OPENAI_REVISION_CONTEXT_PROMPT not in prompt


def test_brand_style_text_names_no_real_episode_headline():
    """Describing past headlines in prose would reintroduce the leak through the back door."""
    lowered = OPENAI_BRAND_STYLE_TEXT.lower()
    for banned in ("global warming", "so2", "are we cooked", "p(doom)", "too big to fail"):
        assert banned not in lowered, f"{banned!r} must not appear in the text style guide"
