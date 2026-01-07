import yaml
from typing import Any, Dict

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
