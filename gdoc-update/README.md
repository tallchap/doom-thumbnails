# gdoc-update

In-place Google Doc content replacement — revisions go into the **same doc at the
same URL**, never a new doc. Retires the create-new-doc-per-revision dance that the
claude.ai Drive connector forces (its `update_file` can only rename/move).

## How auth works (and why)

Uses the **youtubetranscripts service account acting as itself** (key:
`~/Desktop/ClaudeCode/gcp-service-account.json`) with the Drive scope — no OAuth
flow, no browser, no admin console. It can edit any doc shared
**"Anyone with the link → Editor"**, which is already the mandatory sharing state
for every doc per AGENTS.md (docs in the clip-tweets Drive folder inherit it
automatically).

Dead ends, so nobody retries them (verified 2026-08-26):
- `gcloud auth application-default login --scopes=...drive...` → Google hard-blocks
  it ("This app is blocked"): Drive is a restricted scope and the gcloud OAuth
  client isn't verified for it.
- Service account **with DWD impersonation** (`subject=ori@ygrowth.co`) → only the
  datastudio scope is authorized; drive returns `unauthorized_client`. As-itself
  needs neither.

## Usage

```bash
PY=/Users/orinagel2/Desktop/ClaudeCode/google-ads-env-new/bin/python

# 1. ALWAYS export first — captures Ori's hand-edits before any rebuild
$PY gdoc_update.py export "https://docs.google.com/document/d/<ID>/edit" -o current.html

# 2. Rebuild content locally, then replace in place (URL/ID/sharing preserved)
$PY gdoc_update.py replace "https://docs.google.com/document/d/<ID>/edit" new-content.html
```

Accepts full doc URLs or bare file IDs. Content files: `.html` (preferred), `.md`,
`.txt` — Drive converts them to native Google Doc format on upload. A 404/403 from
the tool almost always means the target doc was never link-shared as Editor.

## How it works

`files().update(fileId, media_body=MediaFileUpload(..., mimetype="text/html"))` swaps
the doc body while keeping the file ID, URL, and sharing settings. Export uses
`files().export`.
