import os
import json
from typing import Any, Dict

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
