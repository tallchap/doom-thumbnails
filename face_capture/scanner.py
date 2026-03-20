"""Face capture scanning via Amazon Rekognition Video API."""

import datetime
import json
import os
import subprocess
import tempfile
import threading
import time as _time
from dataclasses import dataclass

import boto3
import cv2
import numpy as np

from config import FC_S3_BUCKET, FC_SNS_TOPIC_ARN, FC_IAM_ROLE_ARN, FC_AWS_REGION, FC_CAPTURES_DIR
from shared.state import status_lock, fc_status


@dataclass
class _FCVideoMeta:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int


@dataclass
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


def _fc_scan_video(meta, expressions, min_face_size, num_samples, log_fn=None, existing_s3_key=None):
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

    if existing_s3_key:
        s3_key = existing_s3_key
        emit(f"  Video already on S3 ({s3_key})")
    else:
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
    tmp_video = None
    try:
        video = params.get("video", "")
        s3_key = params.get("s3_key", "")
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

        if s3_key and not video:
            _fc_log(f"Downloading video from S3...")
            s3 = boto3.client("s3", region_name=FC_AWS_REGION)
            tmp_dir = os.path.join(tempfile.gettempdir(), "fc_downloads")
            os.makedirs(tmp_dir, exist_ok=True)
            filename = os.path.basename(s3_key) or "video.mp4"
            video = os.path.join(tmp_dir, filename)
            tmp_video = video
            s3.download_file(FC_S3_BUCKET, s3_key, video)
            _fc_log(f"  Download complete ({os.path.getsize(video) / (1024*1024):.0f} MB)")

        if not os.path.isfile(video):
            _fc_log(f"ERROR: File not found: {video}")
            return

        t0 = _time.time()
        _fc_log(f"Analyzing: {os.path.basename(video)}")
        meta = _fc_get_video_meta(video)
        _fc_log(f"  {_fc_fmt_time(meta.duration)} | {meta.width}x{meta.height} @ {meta.fps:.1f}fps")
        _fc_log(f"")
        _fc_log(f"Scanning for: {', '.join(expressions)} (via Amazon Rekognition)")

        all_scored = _fc_scan_video(meta, expressions, 0.10, 500, log_fn=_fc_log,
                                     existing_s3_key=s3_key if s3_key else None)

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
        if tmp_video and os.path.isfile(tmp_video):
            try:
                os.remove(tmp_video)
            except OSError:
                pass
        with status_lock:
            fc_status["running"] = False
            fc_status["done"] = True
