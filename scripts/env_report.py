"""Task 1.1 — print exact library versions and whether TensorFlow can see a GPU."""

import getpass
import platform
import socket
import sys
from datetime import datetime
from importlib.metadata import version

PACKAGES = [
    "tensorflow", "keras", "keras-tuner", "mlflow", "dagshub", "dvc", "numpy", "pandas",
    "pillow", "matplotlib", "scikit-learn", "flask", "gunicorn", "sqlalchemy",
]


def main() -> None:
    print(f"user@host : {getpass.getuser()}@{socket.gethostname()}")
    print(f"timestamp : {datetime.now().isoformat(timespec='seconds')}")
    print(f"python    : {sys.version.split()[0]} ({sys.executable})")
    print(f"platform  : {platform.platform()}")
    print("-" * 48)
    for pkg in PACKAGES:
        print(f"{pkg:<14}: {version(pkg)}")
    print("-" * 48)

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TF built with CUDA : {tf.test.is_built_with_cuda()}")
    print(f"GPUs visible to TF : {len(gpus)} {[g.name for g in gpus]}")
    print(f"CPU devices        : {[d.name for d in tf.config.list_physical_devices('CPU')]}")


if __name__ == "__main__":
    main()
