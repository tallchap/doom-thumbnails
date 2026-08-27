#!/usr/bin/env python
"""In-place Google Doc content tool — same doc, same URL, forever.

Usage:
  gdoc_update.py export  <fileId-or-URL> [--format html|markdown|txt] [-o FILE]
  gdoc_update.py replace <fileId-or-URL> <content-file(.html|.md|.txt)>

export  — dump the doc's current content (run this BEFORE any rebuild so
          Ori's hand-edits are captured; AGENTS.md re-read-before-overwrite rule).
replace — swap the doc's entire body for the given file's content while
          keeping the file ID, URL, and sharing settings.

Auth: the youtubetranscripts service account acting AS ITSELF (no DWD, no
impersonation) with the Drive scope. This works on any doc shared
"Anyone with the link -> Editor" — which every doc is required to be per
AGENTS.md. A 404 on a doc usually means it was never link-shared.

(Do NOT try user ADC / gcloud OAuth instead: Drive is a restricted scope and
Google hard-blocks the gcloud client for it — "This app is blocked".)

Run with: /Users/orinagel2/Desktop/ClaudeCode/google-ads-env-new/bin/python
"""

import argparse
import os
import re
import sys

import google.auth.exceptions
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
SA_KEY = "/Users/orinagel2/Desktop/ClaudeCode/gcp-service-account.json"

EXPORT_MIME = {
    "html": "text/html",
    "markdown": "text/markdown",
    "md": "text/markdown",
    "txt": "text/plain",
}
UPLOAD_MIME = {
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}

ACCESS_HINT = (
    "The service account can only touch docs shared 'Anyone with the link -> "
    "Editor' (the AGENTS.md standard). A 404/403 here usually means this doc "
    "was never link-shared — fix its sharing first."
)


def extract_id(s: str) -> str:
    for pat in (r"/d/([A-Za-z0-9_-]{20,})", r"[?&]id=([A-Za-z0-9_-]{20,})"):
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return s


def drive_service():
    creds = service_account.Credentials.from_service_account_file(SA_KEY, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def cmd_export(args) -> int:
    fid = extract_id(args.doc)
    mime = EXPORT_MIME[args.format]
    svc = drive_service()
    data = svc.files().export(fileId=fid, mimeType=mime).execute()
    if args.output:
        with open(args.output, "wb") as f:
            f.write(data)
        print(f"Exported {fid} ({mime}) -> {args.output}")
    else:
        sys.stdout.buffer.write(data)
    return 0


def cmd_replace(args) -> int:
    fid = extract_id(args.doc)
    ext = os.path.splitext(args.content_file)[1].lower()
    mime = UPLOAD_MIME.get(ext)
    if not mime:
        print(f"Unsupported content type '{ext}' — use .html, .md, or .txt", file=sys.stderr)
        return 1
    svc = drive_service()
    media = MediaFileUpload(args.content_file, mimetype=mime, resumable=False)
    result = (
        svc.files()
        .update(
            fileId=fid,
            media_body=media,
            supportsAllDrives=True,
            fields="id,name,webViewLink,modifiedTime",
        )
        .execute()
    )
    print(f"Replaced content of \"{result['name']}\" in place")
    print(f"URL (unchanged): {result['webViewLink']}")
    print(f"Modified: {result['modifiedTime']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="dump current doc content")
    pe.add_argument("doc", help="Google Doc file ID or URL")
    pe.add_argument("--format", choices=sorted(EXPORT_MIME), default="html")
    pe.add_argument("-o", "--output", help="write to file instead of stdout")
    pe.set_defaults(func=cmd_export)

    pr = sub.add_parser("replace", help="replace doc body in place (URL preserved)")
    pr.add_argument("doc", help="Google Doc file ID or URL")
    pr.add_argument("content_file", help="path to .html/.md/.txt content")
    pr.set_defaults(func=cmd_replace)

    args = p.parse_args()
    try:
        return args.func(args)
    except HttpError as e:
        print(f"Drive API error: {e}", file=sys.stderr)
        if e.resp.status in (403, 404):
            print(ACCESS_HINT, file=sys.stderr)
        return 1
    except google.auth.exceptions.RefreshError as e:
        print(f"Auth refresh failed (service account key at {SA_KEY}): {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
