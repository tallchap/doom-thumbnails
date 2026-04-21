#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SHA=$(git rev-parse --short HEAD)
gcloud run deploy doom-thumbnails \
  --project=segment-446404 \
  --source=. \
  --region=us-central1 \
  --memory=4Gi --cpu=2 \
  --max-instances=1 --concurrency=80 --timeout=3600 --min-instances=1 \
  --allow-unauthenticated \
  --update-env-vars="GIT_VERSION=$SHA,MAX_CONCURRENT=30" \
  --update-secrets="DOOM_DRIVE_PARENT_FOLDER_ID=doom-drive-parent-folder-id:latest"
