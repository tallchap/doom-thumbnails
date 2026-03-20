"""Gemini API client setup and File API reference uploads (lazy init)."""

import os
import sys
import threading

from google import genai
from google.genai import types

from config import EXAMPLES_DIR, LIRON_DIR, SCRIPT_DIR

# Gemini File API refs — populated lazily on first use
BRAND_FILES = []
LIRON_FILES = []
BORDER_REF_FILES = []

_init_done = threading.Event()
_init_started = False
_init_lock = threading.Lock()


def get_client(skip_init=False):
    """Get Gemini client. Also triggers lazy ref upload on first call."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        print("Add it to .env file or export GEMINI_API_KEY='your-key'")
        sys.exit(1)
    if not skip_init:
        ensure_gemini_ready()
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
    """Upload border reference images from assets/ to Gemini File API on startup."""
    global BORDER_REF_FILES
    assets_dir = os.path.join(SCRIPT_DIR, "assets")
    if not os.path.isdir(assets_dir):
        print(f"Assets directory not found: {assets_dir} (border pass disabled)")
        return
    files = sorted(
        f for f in os.listdir(assets_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if not files:
        print("No border reference images found in assets/.")
        return
    print(f"Uploading {len(files)} border reference images...")
    for i, f in enumerate(files):
        filepath = os.path.join(assets_dir, f)
        try:
            uploaded = client.files.upload(
                file=filepath,
                config=types.UploadFileConfig(display_name=f"border_ref_{i+1:02d}"),
            )
            BORDER_REF_FILES.append(uploaded)
        except Exception as e:
            print(f"  Failed to upload {f}: {e}")
    print(f"Uploaded {len(BORDER_REF_FILES)} border reference images.")


def upload_files_from_bytes(client, file_bytes_list, name_prefix):
    """Upload raw bytes to Gemini File API, return file reference objects."""
    from config import THUMBNAILS_DIR
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


def init_gemini_files():
    """Upload all reference images. Called lazily on first generation request."""
    client = get_client(skip_init=True)
    upload_brand_references(client)
    upload_liron_references(client)
    upload_border_reference(client)
    return client


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
        # First caller does the work (outside the lock so others can reach wait())
        try:
            init_gemini_files()
        finally:
            _init_done.set()
    else:
        # Another thread is doing init — wait for it
        _init_done.wait(timeout=120)
