"""
config_utils.py
---------------
Handles configuration management and global settings for ExplainDL.
"""

import json
import os

DEFAULT_CONFIG = {
    "random_seed": 42,
    "test_size": 0.2,
    "auto_mode": True,
    "tuning": {
        "enabled": False,
        "max_trials": 10
    },
    "paths": {
        "temp_dir": "temp/",
        "reports_dir": "reports/"
    }
}

def load_config(path: str = "config.json"):
    """
    Loads configuration from a JSON file if present,
    else returns the default config.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        return DEFAULT_CONFIG


def save_config(config: dict, path: str = "config.json"):
    """
    Saves configuration dictionary to JSON file.
    """
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
