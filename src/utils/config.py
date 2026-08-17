import os
import sys
import types
import yaml
from pathlib import Path
from typing import Dict, Any

# Patch missing _bz2 on Windows embedded Python distributions
if '_bz2' not in sys.modules:
    try:
        import _bz2
    except ImportError:
        dummy_bz2 = types.ModuleType('_bz2')
        dummy_bz2.BZ2Compressor = object
        dummy_bz2.BZ2Decompressor = object
        sys.modules['_bz2'] = dummy_bz2

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Safely load a YAML configuration file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> Dict[str, Any]:
    """Load main system configuration."""
    return load_yaml(Path(config_dir) / "config.yaml")

def load_emissions_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> Dict[str, Any]:
    """Load vehicle pollution and noise emission factors."""
    return load_yaml(Path(config_dir) / "emissions.yaml")

def load_model_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> Dict[str, Any]:
    """Load machine learning model hyperparameters."""
    return load_yaml(Path(config_dir) / "model.yaml")
