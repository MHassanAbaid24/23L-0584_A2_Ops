# syntax=docker/dockerfile:1
# VisionGuard web app (Task 6.2). Build: docker build -t 23l-0584-a2-ops-web .
#
# Layer order = least → most frequently changed, so an edit to the source code reuses the cached ~1 GB dependency
# layer and rebuilds in seconds:  base image → OS user → requirements file → pip install → app source.

# Slim Debian base: glibc (needed by the TensorFlow wheel; Alpine's musl is not supported), no compilers/docs.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONPATH=/app/src \
    KERAS_HOME=/tmp/.keras \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Unprivileged runtime user (fixed UID so mounted files can be made readable to it).
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home --shell /usr/sbin/nologin app

# 1) Dependencies first: this layer is rebuilt only when requirements-serve.txt changes.
#    --no-cache-dir keeps pip's download cache (~the size of the wheels) out of the image (Task 6.4).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 2) Application code last: changes often, cheap to rebuild.
COPY gunicorn.conf.py .
COPY src ./src
RUN mkdir -p /app/instance && chown app:app /app/instance

# The model is NOT baked in: Compose mounts ./models read-only at /app/models (see compose.yaml).
# No secrets in the image: everything sensitive arrives as environment variables at run time.
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "visionguard.web:create_app()"]
