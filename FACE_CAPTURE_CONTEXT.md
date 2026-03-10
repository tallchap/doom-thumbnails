# Face Capture Tool -- Developer Context

A macOS tool that scans video files for facial expressions using **Amazon Rekognition** and saves the best screenshots. Built as a web GUI served locally from Python.

**Code now lives in the doom-thumbnails repo** as an orphan page at `/face-capture`. This local project is the development/testing copy. For production updates, edit `doom-thumbnails/thumbnail_gen.py`.

---

## Quick Start

```bash
# 1. Activate the existing venv
source venv/bin/activate

# 2. Run the GUI (opens browser to http://127.0.0.1:9123)
python face_capture_gui.py

# 3. Or run CLI directly
python face_capture.py /path/to/video.mp4 --expression smile --count 10

# 4. Or double-click "Face Capture.app" (must be in same dir as venv and scripts)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| Python | 3.13+ Runtime |
| boto3 | Amazon Rekognition + S3 API |
| opencv-contrib-python | Video reading, frame extraction, image processing |
| numpy | Array operations |
| ffprobe (system) | Video metadata extraction |

**AWS Requirements:** Configured credentials (`aws configure`) with access to:
- **S3 bucket:** `doom-debates-videos` (for temporary video upload)
- **Rekognition Video:** `start_face_detection` / `get_face_detection`
- **SNS topic:** `arn:aws:sns:us-east-1:451721889333:AmazonRekognition-face-detection`
- **IAM role:** `arn:aws:iam::451721889333:role/RekognitionVideoRole`

## Architecture

```
face-capture-project/
  face_capture.py        # Core library + CLI -- Rekognition scanning/scoring/saving
  face_capture_gui.py    # Web GUI -- serves HTML on localhost:9123
  venv/                  # Python virtual environment
  captures/              # Output directory for screenshots
  models/                # Legacy MediaPipe models (no longer used)
  Face Capture.app/      # macOS .app bundle for double-click launch
  CONTEXT.md             # This file
```

## How It Works

### Pipeline

1. **Get video metadata** via ffprobe (duration, fps, resolution)
2. **Upload video to S3** (`doom-debates-videos` bucket)
3. **Start Rekognition face detection job** with `FaceAttributes="ALL"`
4. **Poll for completion** (every 5 seconds)
5. **Paginate through all face results** — Rekognition analyzes every frame and returns:
   - Emotions: HAPPY, SAD, ANGRY, SURPRISED, DISGUSTED, CALM, CONFUSED, FEAR (confidence 0-100)
   - Face quality: brightness, sharpness
   - Pose: yaw, pitch, roll
   - Attributes: eyes open, smile
6. **Filter low-quality detections**: confidence < 85, eyes closed, extreme pose angles
7. **Score each detection** on two axes:
   - **Expression score** (0-1): Mapped from Rekognition emotions via scorer functions
   - **Quality score** (0-1): sharpness (35%), brightness (25%), frontal pose (25%), eyes open (15%)
   - **Combined**: `expression * 0.40 + sharpness * 0.20 + brightness * 0.15 + frontal * 0.15 + eyes * 0.10`
8. **Deduplicate**: Select top N with minimum 2-second gap between picks
9. **Extract frames** via OpenCV at the exact timestamps
10. **Save**: Full frame PNG + padded face crop PNG + metadata.json
11. **Cleanup**: Delete video from S3

### Expression Scorers

Each expression maps Rekognition emotion types to a 0-1 score:

| Expression | Rekognition Emotions Used |
|-----------|--------------------------|
| smile | HAPPY (+ smile attribute) |
| grimace | DISGUSTED |
| surprise | SURPRISED |
| angry | ANGRY |
| sad | SAD |
| thinking | inverse of CALM and HAPPY |
| concerned | FEAR (60%) + SAD (40%) |
| confused | CONFUSED |
| excited | HAPPY (50%) + SURPRISED (50%) |
| serious | CALM minus HAPPY |
| skeptical | CONFUSED (50%) + DISGUSTED (50%) |
| amused | HAPPY (+ smile attribute) |

Aliases: `frown` -> sad, `shocked` -> surprise, `mad` -> angry, `worried` -> concerned, `focused` -> serious

### Multi-Expression Mode

When multiple expressions are selected (or `--expression all`), Rekognition runs once (it always returns all emotions per face). Each detection is scored against all requested expressions. Results are saved to per-expression subdirectories.

## Deployment

The face capture page is deployed as part of **doom-thumbnails** on Render:
- **Repo:** `tallchap/doom-thumbnails` on GitHub
- **Page URL:** `yourdomain.com/face-capture` (orphan page, no links from other pages)
- **Code location:** Inlined in `thumbnail_gen.py` (search for "Face Capture" section)
- **Env vars on Render:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

## File Details

### face_capture.py (Core Library + CLI)

Key exports used by the GUI:
- `CORE_EXPRESSIONS` -- list of all expression names
- `get_video_meta(path) -> VideoMeta` -- ffprobe wrapper
- `scan_video_multi(meta, expressions, min_face_size, num_samples, verbose, log_fn) -> dict[str, list[ScoredFrame]]`
- `select_top(scored, n, gap=2.0) -> list[ScoredFrame]` -- deduplication
- `save_results(selected, output_dir, meta, expression, elapsed)` -- writes PNGs + metadata.json
- `fmt_time(sec) -> str` -- "1:23:45" or "3:45" format
- `fmt_time_hms(sec) -> str` -- "0h03m45s" format (used in filenames)

CLI usage:
```bash
python face_capture.py video.mp4 --expression smile --count 10
python face_capture.py video.mp4 --expression smile,surprise,sad --count 5
python face_capture.py video.mp4 --expression all --verbose
```

### face_capture_gui.py (Web GUI)

A single-file web app. The entire HTML/CSS/JS is embedded as a Python string. Served via `http.server` on `127.0.0.1:9123`.

**Server Endpoints (all GET):**
| Endpoint | Purpose |
|----------|---------|
| `/` | Serve the HTML page |
| `/run?video=...&expressions=...&count=...&output=...` | Start a capture run (background thread) |
| `/status` | Poll for progress (JSON: log, done, result_dirs) |
| `/list_results?dir=...` | List results from a directory (reads metadata.json) |
| `/image?path=...` | Serve a PNG file |
| `/open_image?path=...` | Open a PNG in macOS Preview |
| `/open_folder?dir=...` | Open a folder in Finder |
| `/resolve_file?name=...&size=...` | Resolve full path from filename via `find` |

**Output Folder Structure:**
```
captures/
  Mar10-0204/          # Single expression run
    face_001_score0.87_0h03m18s.png    # Full frame
    face_001_crop.png                   # Cropped face
    metadata.json
  Mar10-1530/          # Multi-expression run
    smile/
      face_001_score0.92_1h02m15s.png
      face_001_crop.png
      metadata.json
    surprise/
      face_001_score0.78_0h45m33s.png
      face_001_crop.png
      metadata.json
```

## Known Issues & Gotchas

1. **AWS credentials**: Must be configured via `aws configure` or environment variables. The tool uploads video to S3, so the IAM user needs `s3:PutObject`, `s3:DeleteObject`, and Rekognition permissions.

2. **Processing time**: Rekognition analyzes the full video (every frame), which takes 1-5 minutes depending on video length. The S3 upload can also take time for large files.

3. **Drag & drop**: Browser security prevents direct file path access. The tool uses a server-side `find` command to resolve filenames.

4. **The `samples` parameter**: Unlike the old MediaPipe version which sampled N frames locally, Rekognition always analyzes every frame. The samples parameter is kept in the CLI for compatibility but is not used by the scanning logic.
