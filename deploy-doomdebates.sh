#!/usr/bin/env bash
set -euo pipefail

GIT_SHA=$(git rev-parse --short HEAD)
echo "Deploying doom-thumbnails commit $GIT_SHA to doomdebates Cloud Run..."

gcloud run deploy doom-thumbnails \
  --project=soy-surge-490220-m8 \
  --account=ori@doomdebates.com \
  --region=us-central1 \
  --source=. \
  --memory=4Gi --cpu=2 \
  --min-instances=0 --max-instances=1 \
  --concurrency=80 --timeout=3600 \
  --no-invoker-iam-check \
  --update-env-vars="GIT_VERSION=$GIT_SHA,MAX_CONCURRENT=30" \
  --update-secrets="GEMINI_API_KEY=gemini-api-key:latest,GEMINI_API_KEY_2=gemini-api-key-2:latest,GOOGLE_APPLICATION_CREDENTIALS_JSON=drive-credentials:latest,DOOM_DRIVE_PARENT_FOLDER_ID=doom-drive-parent-folder-id:latest,OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,BRAVE_API_KEY=brave-api-key:latest,AWS_ACCESS_KEY_ID=aws-access-key-id:latest,AWS_SECRET_ACCESS_KEY=aws-secret-access-key:latest"

echo "Deployed commit $GIT_SHA to doomdebates Cloud Run."
