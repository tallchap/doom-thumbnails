"""Google Drive upload helper for the Save & Clear feature.

Uses a service account (credentials in GOOGLE_APPLICATION_CREDENTIALS_JSON env
var) to create a subfolder inside a user-owned parent folder (specified by
DOOM_DRIVE_PARENT_FOLDER_ID) and upload files into it. The parent folder must
be shared with the service account's client_email as Editor.
"""

import io
import json
import os
from typing import List, Tuple, Dict

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaFileUpload


# Scoped so we can only touch files this app creates, not the whole drive
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
FOLDER_MIME = "application/vnd.google-apps.folder"


def _build_drive_service():
    """Build a drive_v3 service from the GOOGLE_APPLICATION_CREDENTIALS_JSON env var."""
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
    if not creds_json:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS_JSON not set. "
            "Paste the service account JSON as a string into this env var."
        )
    try:
        info = json.loads(creds_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON: {e}")
    creds = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _unique_folder_name(drive, base_name: str, parent_id: str) -> str:
    """If a folder with base_name already exists under parent, return base_name-2, -3, etc."""
    def exists(name: str) -> bool:
        # Escape single quotes in the name for the query
        safe = name.replace("'", "\\'")
        q = (
            f"name = '{safe}' "
            f"and '{parent_id}' in parents "
            f"and mimeType = '{FOLDER_MIME}' "
            f"and trashed = false"
        )
        res = drive.files().list(q=q, fields="files(id)", pageSize=1).execute()
        return bool(res.get("files"))

    if not exists(base_name):
        return base_name
    suffix = 2
    while exists(f"{base_name}-{suffix}"):
        suffix += 1
    return f"{base_name}-{suffix}"


def _create_folder(drive, name: str, parent_id: str) -> dict:
    """Create a folder and return its metadata (id, name, webViewLink)."""
    body = {
        "name": name,
        "mimeType": FOLDER_MIME,
        "parents": [parent_id],
    }
    return drive.files().create(
        body=body,
        fields="id, name, webViewLink",
    ).execute()


def upload_episode_folder(
    folder_name: str,
    text_content: str,
    images: List[Tuple[str, str]],
) -> Dict:
    """Upload an ideas.txt file + images to a new Drive subfolder.

    Args:
        folder_name: Desired subfolder name (e.g. 'my-episode-2026-04-09').
                     If a folder with that name already exists in the parent,
                     the name is bumped to folder_name-2, -3, etc.
        text_content: String contents of ideas.txt.
        images: List of (filename, local_path) tuples. Filenames should already
                be in the target form (e.g. 'idea1a.png').

    Returns:
        {
            "folder_id": str,
            "folder_name": str,   # may differ from input if bumped
            "folder_url": str,    # web URL viewable in browser
            "file_count": int,    # includes ideas.txt
        }

    Raises:
        RuntimeError if env vars are missing or Drive API rejects.
    """
    parent_id = os.environ.get("DOOM_DRIVE_PARENT_FOLDER_ID", "")
    if not parent_id:
        raise RuntimeError(
            "DOOM_DRIVE_PARENT_FOLDER_ID not set. "
            "Create a folder in your Drive, share it with the service account "
            "email as Editor, and set the env var to the folder ID."
        )

    drive = _build_drive_service()

    # Resolve collision-free folder name under parent
    final_name = _unique_folder_name(drive, folder_name, parent_id)
    folder = _create_folder(drive, final_name, parent_id)
    folder_id = folder["id"]

    # Upload ideas.txt first
    txt_media = MediaIoBaseUpload(
        io.BytesIO(text_content.encode("utf-8")),
        mimetype="text/plain",
        resumable=False,
    )
    drive.files().create(
        body={"name": "ideas.txt", "parents": [folder_id]},
        media_body=txt_media,
        fields="id",
    ).execute()

    # Upload each image
    for filename, local_path in images:
        if not os.path.isfile(local_path):
            continue
        media = MediaFileUpload(local_path, mimetype="image/png", resumable=False)
        drive.files().create(
            body={"name": filename, "parents": [folder_id]},
            media_body=media,
            fields="id",
        ).execute()

    # Web URL. folder["webViewLink"] would also work but the plain format is stable.
    folder_url = folder.get("webViewLink") or f"https://drive.google.com/drive/folders/{folder_id}"

    return {
        "folder_id": folder_id,
        "folder_name": final_name,
        "folder_url": folder_url,
        "file_count": len(images) + 1,  # +1 for ideas.txt
    }
