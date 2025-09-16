import os
import json
import logging
from typing import Dict

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj: Dict, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def setup_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger
