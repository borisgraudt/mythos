from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
