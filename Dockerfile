FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpng16-16 libjpeg62-turbo libfreetype6 zlib1g \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV NO_BROWSER=1 \
    WEB_CONCURRENCY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 300 \
    --worker-class gthread --threads 32 "app:create_app()"
