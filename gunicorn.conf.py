"""Gunicorn settings (Task 4.4). Values come from the environment so Docker/cloud can tune them without code changes.

One worker process: each worker would load its own copy of the TensorFlow model (~400 MB RSS), and TF does not
survive fork() after initialisation. Concurrency comes from threads; the model call itself is serialized by a lock.
"""

import os

bind = os.environ.get("BIND", "0.0.0.0:8000")
workers = int(os.environ.get("WEB_WORKERS", "1"))
threads = int(os.environ.get("WEB_THREADS", "4"))
timeout = int(os.environ.get("WEB_TIMEOUT", "60"))   # model load + first request fit comfortably
graceful_timeout = 20
accesslog = "-"                                     # access log to stdout → `docker compose logs`
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
