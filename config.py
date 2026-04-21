"""Centralized configuration — all env vars, constants, model names, paths."""

import os
import subprocess
import sys

from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))


def _get_git_version():
    env_version = os.environ.get("GIT_VERSION", "").strip()
    if env_version:
        return env_version
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=SCRIPT_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


GIT_VERSION = _get_git_version()

# ----- Server -----
PORT = int(os.environ.get("PORT", 9200))
APP_USER = os.environ.get("APP_USERNAME", "doom")
APP_PASS = os.environ.get("APP_PASSWORD", "")
APP_MODE = os.environ.get("APP_MODE", "thumbnails").strip().lower()

# ----- Gemini -----
GEMINI_MODEL = "gemini-3.1-flash-image-preview"
TEXT_MODEL = "gemini-2.5-flash"
CLAUDE_IDEA_MODEL = "claude-opus-4-7"
DESCRIPTION_MODEL = "gemini-3.1-pro-preview"

# ----- Claude / GPT (descriptions page) -----
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_DESCRIPTION_MODEL = "claude-opus-4-6"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_DESCRIPTION_MODEL = "gpt-5.4-pro"

# ----- Brave Search -----
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")

# ----- Directories -----
THUMBNAILS_DIR = os.path.join(SCRIPT_DIR, "thumbnails")
EXAMPLES_DIR = os.path.join(SCRIPT_DIR, "doom_debates_thumbnails")
LIRON_DIR = os.path.join(SCRIPT_DIR, "liron_reactions")

# ----- Face Capture (Amazon Rekognition) -----
FC_S3_BUCKET = "doom-debates-videos"
FC_SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:451721889333:AmazonRekognition-face-detection"
FC_IAM_ROLE_ARN = "arn:aws:iam::451721889333:role/RekognitionVideoRole"
FC_AWS_REGION = "us-east-1"
FC_CAPTURES_DIR = os.path.join(SCRIPT_DIR, "captures")
FC_CORE_EXPRESSIONS = [
    "smile", "grimace", "surprise", "angry", "sad",
    "thinking", "concerned", "excited", "serious", "skeptical", "amused",
]

# ----- Generation -----
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 5))
COST_PER_IMAGE = 0.045  # $0.045 per 512px image
MAX_BRAND_REFS_PER_CALL = 3
MAX_SPEAKER_REFS_PER_CALL = 4
MAX_LIRON_REFS_PER_CALL = 4
