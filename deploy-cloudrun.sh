#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SHA=$(git rev-parse --short HEAD)
gcloud run deploy doom-thumbnails \
  --project=segment-446404 \
  --source=. \
  --region=us-central1 \
  --memory=4Gi --cpu=2 \
  --max-instances=1 --concurrency=80 --timeout=3600 --min-instances=0 \
  --allow-unauthenticated \
  --set-env-vars="GIT_VERSION=$SHA"
