"""Gemini API client setup and File API reference uploads (lazy init).

Supports multiple backends for rate-limit fallback. Each backend wraps one
API key and holds its own File API upload handles (which are per-key scoped).
"""

import os
import sys
import threading
from dataclasses import dataclass, field
from typing import List, Optional

from google import genai
from google.genai import types

from config import EXAMPLES_DIR, LIRON_DIR, SCRIPT_DIR


@dataclass
class GeminiBackend:
    name: str  # "primary" | "secondary"
    client: genai.Client
    brand_files: list = field(default_factory=list)
    liron_files: list = field(default_factory=list)
    border_ref_files: list = field(default_factory=list)


BACKENDS: List[GeminiBackend] = []

_init_done = threading.Event()
_init_started = False
_init_lock = threading.Lock()


def _build_backends() -> List[GeminiBackend]:
    primary_key = os.environ.get("GEMINI_API_KEY", "")
    secondary_key = os.environ.get("GEMINI_API_KEY_2", "")
    if not primary_key:
        print("ERROR: GEMINI_API_KEY not set.")
        print("Add it to .env file or export GEMINI_API_KEY='your-key'")
        sys.exit(1)
    if not secondary_key:
        print("ERROR: GEMINI_API_KEY_2 not set (required secondary key for rate-limit fallback).")
        print("Add it to .env file or export GEMINI_API_KEY_2='your-key'")
        sys.exit(1)
    return [
        GeminiBackend(name="primary", client=genai.Client(api_key=primary_key)),
        GeminiBackend(name="secondary", client=genai.Client(api_key=secondary_key)),
    ]


def get_primary_backend() -> GeminiBackend:
    """Get primary backend. Triggers lazy ref upload on first call."""
    ensure_gemini_ready()
    return BACKENDS[0]


def get_fallback_backend(current: GeminiBackend) -> Optional[GeminiBackend]:
    """Return the next backend to try after `current` hit a rate limit."""
    ensure_gemini_ready()
    for b in BACKENDS:
        if b is not current:
            return b
    return None


def get_all_backends() -> List[GeminiBackend]:
    """Return all initialized backends (primary + secondary). Triggers lazy init."""
    ensure_gemini_ready()
    return list(BACKENDS)


def get_client(skip_init=False):
    """Legacy helper — returns the primary backend's client.

    New code should use get_primary_backend() to access the full backend.
    """
    if skip_init:
        # Used during init_gemini_files to avoid recursion. Build a throwaway primary client.
        primary_key = os.environ.get("GEMINI_API_KEY", "")
        if not primary_key:
            print("ERROR: GEMINI_API_KEY not set.")
            sys.exit(1)
        return genai.Client(api_key=primary_key)
    return get_primary_backend().client


def upload_brand_references(backend: GeminiBackend):
    """Upload all brand thumbnails to one backend's File API."""
    if not os.path.isdir(EXAMPLES_DIR):
        print(f"[{backend.name}] Brand directory not found: {EXAMPLES_DIR}")
        return
    files = sorted(
        f for f in os.listdir(EXAMPLES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if not files:
        print(f"[{backend.name}] No brand reference images found.")
        return
    print(f"[{backend.name}] Uploading {len(files)} brand references to File API...")
    for i, f in enumerate(files):
        filepath = os.path.join(EXAMPLES_DIR, f)
        try:
            uploaded = backend.client.files.upload(
                file=filepath,
                config=types.UploadFileConfig(display_name=f"brand_doom_debates_{i+1:02d}"),
            )
            backend.brand_files.append(uploaded)
        except Exception as e:
            print(f"  [{backend.name}] Failed to upload {f}: {e}")
    print(f"[{backend.name}] Uploaded {len(backend.brand_files)} brand references.")


def upload_liron_references(backend: GeminiBackend):
    """Upload Liron reaction photos to one backend's File API."""
    if not os.path.isdir(LIRON_DIR):
        print(f"[{backend.name}] Liron reactions directory not found: {LIRON_DIR}")
        return
    files = sorted(
        f for f in os.listdir(LIRON_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if not files:
        print(f"[{backend.name}] No Liron reaction images found.")
        return
    print(f"[{backend.name}] Uploading {len(files)} Liron reaction photos to File API...")
    for i, f in enumerate(files):
        filepath = os.path.join(LIRON_DIR, f)
        try:
            uploaded = backend.client.files.upload(
                file=filepath,
                config=types.UploadFileConfig(display_name=f"liron_{os.path.splitext(f)[0]}"),
            )
            backend.liron_files.append(uploaded)
        except Exception as e:
            print(f"  [{backend.name}] Failed to upload {f}: {e}")
    print(f"[{backend.name}] Uploaded {len(backend.liron_files)} Liron reaction photos.")


def upload_border_reference(backend: GeminiBackend):
    """Upload border reference images from assets/ to one backend's File API."""
    assets_dir = os.path.join(SCRIPT_DIR, "assets")
    if not os.path.isdir(assets_dir):
        print(f"[{backend.name}] Assets directory not found: {assets_dir} (border pass disabled)")
        return
    files = sorted(
        f for f in os.listdir(assets_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if not files:
        print(f"[{backend.name}] No border reference images found in assets/.")
        return
    print(f"[{backend.name}] Uploading {len(files)} border reference images...")
    for i, f in enumerate(files):
        filepath = os.path.join(assets_dir, f)
        try:
            uploaded = backend.client.files.upload(
                file=filepath,
                config=types.UploadFileConfig(display_name=f"border_ref_{i+1:02d}"),
            )
            backend.border_ref_files.append(uploaded)
        except Exception as e:
            print(f"  [{backend.name}] Failed to upload {f}: {e}")
    print(f"[{backend.name}] Uploaded {len(backend.border_ref_files)} border reference images.")


def upload_files_from_bytes(file_bytes_list, name_prefix):
    """Upload raw bytes to every backend's File API, return a per-backend dict of refs.

    Returns {"primary": [refs...], "secondary": [refs...]} so callers can select
    the right refs for whichever backend they're currently using.
    """
    from config import THUMBNAILS_DIR
    backends = get_all_backends()
    by_backend: dict = {b.name: [] for b in backends}
    for i, data in enumerate(file_bytes_list):
        tmp_path = os.path.join(THUMBNAILS_DIR, f"_tmp_{name_prefix}_{i+1}.jpg")
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
            for backend in backends:
                try:
                    uploaded = backend.client.files.upload(
                        file=tmp_path,
                        config=types.UploadFileConfig(display_name=f"{name_prefix}_{i+1}"),
                    )
                    by_backend[backend.name].append(uploaded)
                except Exception as e:
                    print(f"  [{backend.name}] Failed to upload {name_prefix}_{i+1}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return by_backend


def init_gemini_files():
    """Initialize all backends and upload all reference image sets to each."""
    global BACKENDS
    BACKENDS = _build_backends()
    for backend in BACKENDS:
        upload_brand_references(backend)
        upload_liron_references(backend)
        upload_border_reference(backend)


def ensure_gemini_ready():
    """Lazy init: upload refs on first call, block concurrent callers until done."""
    global _init_started
    if _init_done.is_set():
        return
    should_init = False
    with _init_lock:
        if not _init_started:
            _init_started = True
            should_init = True
    if should_init:
        try:
            init_gemini_files()
        finally:
            _init_done.set()
    else:
        _init_done.wait(timeout=120)
