"""Flask blueprint for the face capture page (/face-capture)."""

import io
import json
import os
import subprocess
import tempfile
import threading
import zipfile

import boto3
from flask import Blueprint, jsonify, render_template, request, Response

from auth import require_auth
from config import (
    GIT_VERSION, FC_S3_BUCKET, FC_AWS_REGION, FC_CAPTURES_DIR, FC_CORE_EXPRESSIONS,
)
from shared.state import status_lock, fc_status
from face_capture.scanner import _fc_run_capture

face_capture_bp = Blueprint("face_capture", __name__, template_folder="templates")


@face_capture_bp.route("/face-capture")
@require_auth
def face_capture_index():
    return render_template(
        "face_capture.html",
        git_version=GIT_VERSION,
        fc_expressions=json.dumps(FC_CORE_EXPRESSIONS),
        fc_captures_dir=FC_CAPTURES_DIR,
    )


@face_capture_bp.route("/fc_presign")
@require_auth
def fc_presign():
    filename = request.args.get("filename", "video.mp4")
    filename = os.path.basename(filename) or "video.mp4"
    s3_key = f"uploads/{filename}"
    try:
        s3 = boto3.client("s3", region_name=FC_AWS_REGION)
        url = s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": FC_S3_BUCKET, "Key": s3_key, "ContentType": "video/mp4"},
            ExpiresIn=3600,
        )
        return jsonify({"url": url, "s3_key": s3_key})
    except Exception as e:
        return jsonify({"error": f"Failed to generate presigned URL: {str(e)[:200]}"})


@face_capture_bp.route("/fc_status")
@require_auth
def fc_status_route():
    with status_lock:
        fc_safe = {
            "running": fc_status["running"],
            "log": list(fc_status["log"]),
            "done": fc_status["done"],
            "output_dir": fc_status["output_dir"],
            "result_dirs": dict(fc_status["result_dirs"]),
        }
    return jsonify(fc_safe)


@face_capture_bp.route("/fc_run")
@require_auth
def fc_run():
    params = dict(request.args)
    with status_lock:
        if not fc_status["running"]:
            fc_status["running"] = True
            fc_status["done"] = False
            fc_status["log"] = []
            fc_status["output_dir"] = ""
            fc_status["result_dirs"] = {}
            t = threading.Thread(target=_fc_run_capture, args=(params,), daemon=True)
            t.start()
    return jsonify({"ok": True})


@face_capture_bp.route("/fc_list_results")
@require_auth
def fc_list_results():
    d = request.args.get("dir", "")
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
                        "expr_score": entry.get("expression_score", 0),
                        "qual_score": entry.get("quality_score", 0),
                    })
        else:
            for f_name in sorted(os.listdir(d)):
                if f_name.endswith("_crop.png"):
                    items.append({"crop_path": os.path.join(d, f_name), "full_path": "", "timestamp": "", "score": 0})
    return jsonify(items)


@face_capture_bp.route("/fc_image")
@require_auth
def fc_image():
    img_path = request.args.get("path", "")
    if os.path.isfile(img_path) and img_path.endswith(".png"):
        with open(img_path, "rb") as f:
            data = f.read()
        headers = {"Content-Type": "image/png", "Content-Length": str(len(data))}
        if request.args.get("download") == "1":
            fname = os.path.basename(img_path)
            headers["Content-Disposition"] = f'attachment; filename="{fname}"'
        return Response(data, status=200, headers=headers)
    return "Not found", 404


@face_capture_bp.route("/fc_download_zip")
@require_auth
def fc_download_zip():
    d = request.args.get("dir", "")
    if not os.path.isdir(d):
        return "Not found", 404
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(d):
            for fname in files:
                if fname.endswith(".png") or fname == "metadata.json":
                    full = os.path.join(root, fname)
                    arcname = os.path.relpath(full, os.path.dirname(d))
                    zf.write(full, arcname)
    zip_data = zip_buf.getvalue()
    folder_name = os.path.basename(d)
    return Response(
        zip_data, status=200,
        headers={
            "Content-Type": "application/zip",
            "Content-Length": str(len(zip_data)),
            "Content-Disposition": f'attachment; filename="captures-{folder_name}.zip"',
        },
    )


@face_capture_bp.route("/fc_resolve_file")
@require_auth
def fc_resolve_file():
    name = request.args.get("name", "")
    size = int(request.args.get("size", "0"))
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
    return jsonify({"path": resolved})


@face_capture_bp.route("/fc_open_image")
@require_auth
def fc_open_image():
    img_path = request.args.get("path", "")
    if os.path.isfile(img_path) and img_path.endswith(".png"):
        os.system(f'open "{img_path}"')
    return jsonify({"ok": True})


@face_capture_bp.route("/fc_open_folder")
@require_auth
def fc_open_folder():
    d = request.args.get("dir", "")
    if os.path.isdir(d):
        os.system(f'open "{d}"')
    return jsonify({"ok": True})


@face_capture_bp.route("/fc_upload", methods=["POST"])
@require_auth
def fc_upload():
    """Stream video upload directly to disk."""
    content_type = request.content_type or ""

    if "multipart/form-data" not in content_type:
        return jsonify({"error": "Expected multipart/form-data"})

    # Extract boundary from Content-Type header
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[9:].strip().strip('"')
    if not boundary:
        return jsonify({"error": "No boundary in Content-Type"})

    upload_dir = os.path.join(tempfile.gettempdir(), "fc_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, "upload.mp4")

    boundary_bytes = ("--" + boundary).encode()

    raw = request.get_data()

    parts = raw.split(boundary_bytes)
    filename = "upload.mp4"
    file_written = False

    for part in parts:
        if b'name="video"' in part or b'name="video";' in part:
            header_end = part.find(b"\r\n\r\n")
            if header_end == -1:
                continue
            header_section = part[:header_end].decode("utf-8", errors="replace")
            for line in header_section.split("\r\n"):
                if "filename=" in line:
                    fn_start = line.find('filename="')
                    if fn_start != -1:
                        fn_start += 10
                        fn_end = line.find('"', fn_start)
                        if fn_end != -1:
                            filename = os.path.basename(line[fn_start:fn_end]) or "upload.mp4"

            file_data = part[header_end + 4:]
            if file_data.endswith(b"\r\n"):
                file_data = file_data[:-2]

            save_path = os.path.join(upload_dir, filename)
            with open(save_path, "wb") as f:
                f.write(file_data)
            file_written = True
            break
        elif b'name="filename"' in part:
            header_end = part.find(b"\r\n\r\n")
            if header_end != -1:
                val = part[header_end + 4:].strip().decode("utf-8", errors="replace").strip()
                if val:
                    filename = os.path.basename(val) or "upload.mp4"

    del raw

    if not file_written:
        return jsonify({"error": "No video file found in upload"})

    return jsonify({"path": save_path})
