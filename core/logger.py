import logging
import os
import sys
from pathlib import Path

_INITIALIZED = False


def _ensure_log_dir(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)


def _setup_root_logger(log_dir: str) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    _ensure_log_dir(log_dir)
    log_path = os.path.join(log_dir, "app.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    _INITIALIZED = True


def get_logger(name: str, log_dir: str = "storage/logs") -> logging.Logger:
    _setup_root_logger(log_dir)
    return logging.getLogger(name)
