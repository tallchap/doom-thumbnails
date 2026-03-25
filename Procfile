web: find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; gunicorn --bind 0.0.0.0:$PORT --timeout 300 --worker-class gthread --threads 4 "app:create_app()"
